#!/usr/bin/env bash
# Verify and atomically provision an LFM LoRA ORT Training bundle over adb.
set -euo pipefail

PACKAGE="ai.onlinesdft.router.debug"
BUNDLE_DIR=""
SERIAL=""
ADB_BIN="${ADB:-adb}"
DEVICE_DIR=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: provision_lfm_lora.sh --bundle-dir DIR [options]

Options:
  --bundle-dir DIR         export_lfm_lora_ort.py output (required)
  --serial SERIAL          adb device serial (recommended with >1 device)
  --package APPLICATION_ID Override debug applicationId
  --device-dir PATH        Override app-private files destination
  --adb PATH               adb executable
  --dry-run                Verify host artifacts and print the deployment plan
  -h, --help               Show help
EOF
}

die() { echo "error: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir) [[ $# -ge 2 ]] || die "--bundle-dir requires a value"; BUNDLE_DIR="$2"; shift 2 ;;
    --serial) [[ $# -ge 2 ]] || die "--serial requires a value"; SERIAL="$2"; shift 2 ;;
    --package) [[ $# -ge 2 ]] || die "--package requires a value"; PACKAGE="$2"; shift 2 ;;
    --device-dir) [[ $# -ge 2 ]] || die "--device-dir requires a value"; DEVICE_DIR="$2"; shift 2 ;;
    --adb) [[ $# -ge 2 ]] || die "--adb requires a value"; ADB_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$BUNDLE_DIR" && -d "$BUNDLE_DIR" ]] || die "--bundle-dir must name a directory"
BUNDLE_DIR="$(cd "$BUNDLE_DIR" && pwd -P)"
[[ -f "$BUNDLE_DIR/manifest.json" ]] || die "missing manifest.json"
[[ "$PACKAGE" =~ ^[A-Za-z0-9._]+$ ]] || die "unsafe package"
if [[ -z "$DEVICE_DIR" ]]; then
  DEVICE_DIR="files/lfm_lora"
fi
DEVICE_DIR="${DEVICE_DIR%/}"
[[ "$DEVICE_DIR" =~ ^[A-Za-z0-9._/-]+$ && "$DEVICE_DIR" != /* && "$DEVICE_DIR" != *".."* ]] || die "unsafe device path"
DEVICE_PARENT="${DEVICE_DIR%/*}"
DEVICE_NAME="${DEVICE_DIR##*/}"
STAGE_DIR="$DEVICE_PARENT/.${DEVICE_NAME}.provisioning"
BACKUP_DIR="$DEVICE_PARENT/.${DEVICE_NAME}.previous"
TRANSFER_DIR="/data/local/tmp/.lfm-lora-${PACKAGE}"
TRANSFER_FILE="$TRANSFER_DIR/payload"
ADB_CMD=("$ADB_BIN")
[[ -z "$SERIAL" ]] || ADB_CMD+=("-s" "$SERIAL")
RECORDS="$(mktemp "${TMPDIR:-/tmp}/lfm-lora-records.XXXXXX")"
cleanup() {
  rm -f "$RECORDS"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${ADB_CMD[@]}" shell "rm -rf '$TRANSFER_DIR'" </dev/null >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

python3 - "$BUNDLE_DIR" "$PACKAGE" >"$RECORDS" <<'PY'
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
package = sys.argv[2]
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text())
assert manifest["schema"] == "ai.onlinesdft.lfm_lora_ort_bundle"
assert manifest["schema_version"] == 1
assert manifest["model"]["id"] == "LiquidAI/LFM2.5-230M"
assert manifest["model"]["revision"] == "13a53837c4906b4f7405932532ba85d182bb013b"
assert manifest["model"]["trainable_parameters"] == 172032
assert manifest["model"]["trainable_tensors"] == 48
assert manifest["deploy_contract"]["android_application_id"] == package
deploy = manifest["deploy_contract"]
assert deploy.get("private_files_subdirectory", deploy.get("external_files_subdirectory")) == "lfm_lora"
expected_roles = {
    "zero_adapter_base_model", "lora_training_model", "lora_eval_model",
    "adamw_optimizer_model", "initial_lora_checkpoint", "tokenizer",
}
records = [row for row in manifest["artifacts"] if row.get("deploy")]
assert {row["role"] for row in records} == expected_roles
safe = re.compile(r"^[A-Za-z0-9._/-]+$")

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024): value.update(chunk)
    return value.hexdigest()

seen = set()
for row in records:
    relative = row["path"]
    assert safe.fullmatch(relative) and ".." not in Path(relative).parts and relative not in seen
    seen.add(relative)
    source = root / relative
    assert row["kind"] == "file" and source.is_file()
    actual = digest(source)
    assert source.stat().st_size == row["bytes"] and actual == row["sha256"]
    print(f"{source}\t{relative}\t{actual}\t{row['bytes']}")
print(f"{manifest_path}\tmanifest.json\t{digest(manifest_path)}\t{manifest_path.stat().st_size}")
PY

TOTAL_BYTES="$(awk -F $'\t' '{sum += $4} END {printf "%.0f", sum}' "$RECORDS")"
echo "verified_host_bundle=$BUNDLE_DIR bytes=$TOTAL_BYTES"
echo "device_destination=run-as:$PACKAGE/$DEVICE_DIR"
[[ "$DRY_RUN" -eq 0 ]] || exit 0

"${ADB_CMD[@]}" get-state >/dev/null
"${ADB_CMD[@]}" shell "run-as '$PACKAGE' id >/dev/null" </dev/null ||
  die "$PACKAGE must be installed as a debuggable application"
"${ADB_CMD[@]}" shell "rm -rf '$TRANSFER_DIR' && mkdir -p '$TRANSFER_DIR' && chmod 755 '$TRANSFER_DIR'" </dev/null
"${ADB_CMD[@]}" shell "run-as '$PACKAGE' sh -c \"mkdir -p '$DEVICE_PARENT' && rm -rf '$STAGE_DIR' '$BACKUP_DIR' && mkdir -p '$STAGE_DIR'\"" </dev/null

while IFS=$'\t' read -r source relative expected bytes; do
  remote="$STAGE_DIR/$relative"
  remote_parent="${remote%/*}"
  # adb may read its own stdin even for non-interactive commands. Keep it
  # detached from the records stream or the first push consumes later rows.
  "${ADB_CMD[@]}" push "$source" "$TRANSFER_FILE" </dev/null >/dev/null
  "${ADB_CMD[@]}" shell "chmod 644 '$TRANSFER_FILE' && run-as '$PACKAGE' sh -c \"mkdir -p '$remote_parent' && cp '$TRANSFER_FILE' '$remote'\"" </dev/null
  actual="$("${ADB_CMD[@]}" shell "run-as '$PACKAGE' sha256sum '$remote'" </dev/null | tr -d '\r' | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || die "device hash mismatch: $relative"
  echo "provisioned=$relative bytes=$bytes"
done <"$RECORDS"

"${ADB_CMD[@]}" shell "run-as '$PACKAGE' sh -c \"if [ -e '$DEVICE_DIR' ]; then mv '$DEVICE_DIR' '$BACKUP_DIR'; fi; mv '$STAGE_DIR' '$DEVICE_DIR'; rm -rf '$BACKUP_DIR'\"" </dev/null
echo "provision_complete=run-as:$PACKAGE/$DEVICE_DIR"
echo "force_stop_required=$PACKAGE"
