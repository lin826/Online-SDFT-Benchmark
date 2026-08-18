#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PYTHON_BIN="${ONLINE_SDFT_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is not executable: $PYTHON_BIN" >&2
  echo "Set ONLINE_SDFT_PYTHON to the benchmark virtualenv interpreter." >&2
  exit 2
fi
LOG="$ROOT/logs/mps_seeds_realistic.log"
LOCK="$ROOT/logs/mps_seeds.lock"
mkdir -p "$ROOT/logs"

if [[ -f "$LOCK" ]]; then
  old=$(cat "$LOCK" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "already running pid=$old" >>"$LOG"
    exit 0
  fi
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/online-sdft-mps-seeds.XXXXXX")"
export STAGE
SOURCE_FINGERPRINT_FILE="$STAGE/source_fingerprint.json"
export SOURCE_FINGERPRINT_FILE
exec >>"$LOG" 2>&1
echo "==== launcher $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$ ===="
echo "staging=$STAGE"

capture_source_fingerprint() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import hashlib
import json
import os

root = Path.cwd()
source_files = (
    "online_sdft/__init__.py",
    "online_sdft/config.py",
    "online_sdft/environment.py",
    "online_sdft/experiment.py",
    "online_sdft/methods.py",
    "online_sdft/privilege.py",
    "online_sdft/reporting.py",
    "requirements.txt",
    "run.py",
    "scripts/run_mps_seeds_and_merge.sh",
)


def fingerprint():
    files = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in source_files
    }
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"algorithm": "sha256", "sha256": aggregate, "files": files}


snapshot = fingerprint()
if fingerprint() != snapshot:
    raise RuntimeError("benchmark source changed while capturing its fingerprint")
path = Path(os.environ["SOURCE_FINGERPRINT_FILE"])
path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
print("SOURCE_FINGERPRINT_CAPTURED", snapshot["sha256"], flush=True)
PY
}

verify_source_fingerprint() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import hashlib
import json
import os

root = Path.cwd()
snapshot = json.loads(
    Path(os.environ["SOURCE_FINGERPRINT_FILE"]).read_text()
)
if snapshot.get("algorithm") != "sha256":
    raise RuntimeError("unknown benchmark source fingerprint algorithm")
expected_files = snapshot.get("files")
if not isinstance(expected_files, dict) or not expected_files:
    raise RuntimeError("benchmark source fingerprint has no file manifest")
observed_files = {
    relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
    for relative in expected_files
}
aggregate = hashlib.sha256(
    json.dumps(observed_files, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
changed = sorted(
    relative
    for relative, expected_hash in expected_files.items()
    if observed_files[relative] != expected_hash
)
if changed or aggregate != snapshot.get("sha256"):
    raise RuntimeError(
        "benchmark source changed after the pre-run snapshot: "
        + ", ".join(changed or ["aggregate mismatch"])
    )
print("SOURCE_FINGERPRINT_VERIFIED", aggregate, flush=True)
PY
}

capture_source_fingerprint

# Pin the canonical Online-SDFT configuration before expensive model work.
"$PYTHON_BIN" - <<'PY'
from online_sdft.config import (
    DATASET_NUMPY_VERSION,
    DATASET_VERSION,
    METHODS,
    OBSERVED_OUTCOME_REWARDS,
    PREFERENCE_SAMPLING_TEMPERATURE,
    REINFORCE_BATCH_SIZE,
    REINFORCE_BASELINE_STEP,
    REINFORCE_ENTROPY_COEF,
    REINFORCE_LR,
    REINFORCE_MAX_GRAD_NORM,
    REINFORCE_TRAINING_OUTCOME_REWARDS,
    RFT_PROTOCOL_VERSION,
    RFT_CANDIDATE_COUNT,
    RFT_LR,
    RFT_SAMPLING_MODE,
    RFT_SAMPLING_TEMPERATURE,
    RFT_SETTINGS_PROVENANCE,
    TEACHER_PROMPT_VERSION,
)
from online_sdft.experiment import RFT_CANDIDATE_RNG_OFFSET
from online_sdft.methods import (
    DEFAULT_REINFORCE_SETTINGS,
    DEFAULT_RFT_SETTINGS,
    DEFAULT_SDFT_SETTINGS,
)

expected_version = "concise-causal-v3"
expected_settings = {
    "learning_rate": 0.001,
    "replay_size": 64,
    "replay_prompt_examples": 0,
    "batch_size": 8,
    "update_steps": 2,
    "warmup_examples": 4,
    "lora_rank": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "lora_target_modules": (
        "q_proj", "k_proj", "v_proj", "self_attn.out_proj"
    ),
    "lora_layers_to_transform": (2, 4, 6, 8, 10, 12),
    "lora_a_learning_rate_scale": 1.0,
    "lm_head_lora_a_learning_rate_scale": None,
    "optimizer_weight_decay": 0.0,
    "optimizer_beta1": 0.9,
    "max_grad_norm": 1.0,
    "lm_head_learning_rate": None,
    "ambiguous_replay_group_weight": 0.05,
    "teacher_temperature": 1.0,
    "reasoning_tokens": 0,
    "target_mode": "causal_fusion",
    "reliable_teacher_weight": 0.05,
    "reliable_decision_weight": 0.05,
    "reliable_behavior_weight": 0.90,
    "ambiguous_teacher_weight": 0.0,
    "ambiguous_decision_weight": 1.0,
    "ambiguous_behavior_weight": 0.0,
    "ambiguous_projection": "causal_support",
    "replay_strategy": "selection_balanced",
    "replay_recency_half_life": 32.0,
    "ambiguous_update_mode": "immediate",
    "force_newest_every_step": True,
    "base_kl_weight": 0.0,
    "behavior_mode": "uncertainty_interrupt_probe",
    "behavior_epsilon": 0.02,
    "behavior_epsilon_half_life": None,
    "exploration_taper_start_step": 160,
    "exploration_taper_half_life": 5.0,
    "archive_probe_mix": 0.0,
    "archive_policy_min_feedback": 0.0,
    "interrupt_probe_mix": 0.15,
    "interrupt_probe_half_life": 80.0,
    "interrupt_probe_max_confidence": 0.60,
    "propensity_weight_mode": "none",
    "propensity_weight_cap": 4.0,
}
expected_rft_student_settings = {
    "learning_rate": 0.0007,
    "replay_size": 32,
    "replay_prompt_examples": 0,
    "batch_size": 8,
    "update_steps": 2,
    "warmup_examples": 4,
    "lora_rank": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "lora_target_modules": (
        "q_proj", "k_proj", "v_proj", "self_attn.out_proj"
    ),
    "lora_layers_to_transform": (2, 4, 6, 8, 10, 12),
    "lora_a_learning_rate_scale": 1.0,
    "lm_head_lora_a_learning_rate_scale": None,
    "optimizer_weight_decay": 0.0,
    "optimizer_beta1": 0.9,
    "max_grad_norm": 1.0,
    "lm_head_learning_rate": None,
    "ambiguous_replay_group_weight": 0.05,
    "teacher_temperature": 1.0,
    "reasoning_tokens": 0,
    "target_mode": "causal_fusion",
    "reliable_teacher_weight": 0.05,
    "reliable_decision_weight": 0.05,
    "reliable_behavior_weight": 0.90,
    "ambiguous_teacher_weight": 0.0,
    "ambiguous_decision_weight": 1.0,
    "ambiguous_behavior_weight": 0.0,
    "ambiguous_projection": "causal_support",
    "replay_strategy": "selection_balanced",
    "replay_recency_half_life": None,
    "ambiguous_update_mode": "immediate",
    "force_newest_every_step": True,
    "base_kl_weight": 0.0,
    "behavior_mode": "epsilon_greedy",
    "behavior_epsilon": 0.06,
    "behavior_epsilon_half_life": None,
    "exploration_taper_start_step": None,
    "exploration_taper_half_life": None,
    "archive_probe_mix": 0.0,
    "archive_policy_min_feedback": 0.0,
    "interrupt_probe_mix": 0.0,
    "interrupt_probe_half_life": None,
    "interrupt_probe_max_confidence": 1.0,
    "propensity_weight_mode": "none",
    "propensity_weight_cap": 4.0,
}
expected_rft_settings = {
    "student_settings": expected_rft_student_settings,
    "candidate_count": 1,
    "sampling_temperature": 8.0,
    "sampling_mode": "categorical",
}
expected_reinforce_settings = {
    "learning_rate": 0.0001,
    "batch_size": 8,
    "baseline_step": 0.0,
    "entropy_coef": 1.0,
    "max_grad_norm": 1.0,
    "reward_outcome_map": {
        "OPENED_IMMEDIATELY": 5.0,
        "OPENED_AFTER_DELAY": -1.0,
        "DELETED_NOTIFICATION": -5.0,
        "OPENED_DIGEST": 0.0,
        "DELETED_FROM_DIGEST": -5.0,
        "NO_OBSERVABLE_SELECTION": 0.0,
    },
}
assert TEACHER_PROMPT_VERSION == expected_version, TEACHER_PROMPT_VERSION
assert DATASET_VERSION == "semantic-title-body-sharp-t001", DATASET_VERSION
assert DATASET_NUMPY_VERSION == "2.4.6", DATASET_NUMPY_VERSION
assert PREFERENCE_SAMPLING_TEMPERATURE == 0.01, (
    PREFERENCE_SAMPLING_TEMPERATURE
)
assert METHODS == (
    "Base",
    "ICL",
    "RAG",
    "REINFORCE",
    "RFT",
    "Online-SDFT",
), METHODS
assert DEFAULT_SDFT_SETTINGS.to_dict() == expected_settings, (
    DEFAULT_SDFT_SETTINGS.to_dict()
)
assert DEFAULT_REINFORCE_SETTINGS.to_dict() == expected_reinforce_settings, (
    DEFAULT_REINFORCE_SETTINGS.to_dict()
)
assert DEFAULT_RFT_SETTINGS.to_dict() == expected_rft_settings, (
    DEFAULT_RFT_SETTINGS.to_dict()
)
assert (
    len(expected_settings["lora_target_modules"])
    * len(expected_settings["lora_layers_to_transform"])
    * 2
    == 48
)
assert REINFORCE_BATCH_SIZE == 8
assert REINFORCE_LR == 0.0001
assert REINFORCE_BASELINE_STEP == 0.0
assert REINFORCE_ENTROPY_COEF == 1.0
assert REINFORCE_MAX_GRAD_NORM == 1.0
assert REINFORCE_TRAINING_OUTCOME_REWARDS == (
    expected_reinforce_settings["reward_outcome_map"]
)
assert RFT_LR == 0.0007
assert RFT_PROTOCOL_VERSION == "teacher-categorical-k1-temperature8-singleton"
assert RFT_SETTINGS_PROVENANCE == (
    "fixed-temperature8-lr7e-4-same-lora-architecture"
)
assert RFT_CANDIDATE_COUNT == 1
assert RFT_SAMPLING_TEMPERATURE == 8.0
assert RFT_SAMPLING_MODE == "categorical"
assert RFT_CANDIDATE_RNG_OFFSET == 83
assert OBSERVED_OUTCOME_REWARDS["OPENED_IMMEDIATELY"] == 5.0
print("CONFIG_PIN", expected_version, expected_settings)
PY
verify_source_fingerprint

# Wait up to ~30 min for MPS to become healthy (post-reboot / post-unwedge)
healthy=0
for i in $(seq 1 60); do
  if "$PYTHON_BIN" -u - <<'PY'
import torch, sys
x = torch.ones(4, device="mps")
torch.mps.synchronize()
assert float(x.sum().cpu()) == 4.0
print("MPS_HEALTHY", torch.__version__, flush=True)
PY
  then
    healthy=1
    break
  fi
  echo "mps unhealthy attempt=$i $(date -u +%H:%M:%S); sleep 30"
  sleep 30
done
if [[ "$healthy" -ne 1 ]]; then
  echo "MPS never recovered; abort"
  exit 2
fi

for s in 0 1 2; do
  verify_source_fingerprint
  echo "==== seed $s start $(date -u +%H:%M:%S) ===="
  "$PYTHON_BIN" -u "$ROOT/run.py" \
    --seeds 1 --seed-start "$s" --device mps --local-files-only \
    --prompt-style causal_demos --icl-examples 3 --rag-examples 3 \
    --rag-text-weight 0.5 \
    --output-dir "$STAGE/seeds/s$s" \
    --figure-dir "$STAGE/seed_figures/s$s"
  verify_source_fingerprint
  echo "==== seed $s done $(date -u +%H:%M:%S) ===="
done

verify_source_fingerprint
echo "==== merge $(date -u +%H:%M:%S) ===="
"$PYTHON_BIN" -u - <<'PY'
from pathlib import Path
from collections import Counter
import csv, hashlib, json, math, os, shutil
from online_sdft.reporting import write_compact_results, write_figures, summarize_metrics
from online_sdft.config import (
    ACTIONS,
    DATASET_NUMPY_VERSION,
    DATASET_VERSION,
    METHODS,
    OBSERVED_OUTCOME_REWARDS,
    PREFERENCE_SAMPLING_TEMPERATURE,
    REINFORCE_BATCH_SIZE,
    REINFORCE_BASELINE_STEP,
    REINFORCE_ENTROPY_COEF,
    REINFORCE_LR,
    REINFORCE_MAX_GRAD_NORM,
    REINFORCE_TRAINING_OUTCOME_REWARDS,
    RFT_CANDIDATE_COUNT,
    RFT_LR,
    RFT_PROTOCOL_VERSION,
    RFT_SAMPLING_MODE,
    RFT_SAMPLING_TEMPERATURE,
    RFT_SETTINGS_PROVENANCE,
    TEACHER_PROMPT_VERSION,
)
from online_sdft.experiment import (
    RFT_CANDIDATE_RNG_OFFSET,
    RFT_CANDIDATE_SAMPLER,
)
from online_sdft.environment import DEFAULT_ENVIRONMENT
from online_sdft.methods import (
    DEFAULT_REINFORCE_SETTINGS,
    DEFAULT_RFT_SETTINGS,
    DEFAULT_SDFT_SETTINGS,
)
from online_sdft.reporting import summarize_rft_diagnostics

repo = Path.cwd()
stage = Path(os.environ["STAGE"])
source_fingerprint_file = Path(os.environ["SOURCE_FINGERPRINT_FILE"])
source_fingerprint = json.loads(source_fingerprint_file.read_text())
seed_dirs = [stage / "seeds" / f"s{s}" for s in (0, 1, 2)]
out = stage / "merged" / "outputs"
fig = stage / "merged" / "figures"
out.mkdir(parents=True, exist_ok=True)

configs = []
metrics = []
curves = []
rollouts = []
for d in seed_dirs:
    cfg = json.loads((d / "summary.json").read_text())["config"]
    configs.append(cfg)
    with (d / "per_seed_metrics.csv").open() as f:
        metrics.extend(csv.DictReader(f))
    with (d / "learning_curves.csv").open() as f:
        curves.extend(csv.DictReader(f))
    with (d / "rollouts.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))

# coerce numeric fields for summarize
def numify(row):
    out = dict(row)
    for k, v in row.items():
        if k in {"seed", "t", "step_correct"}:
            try: out[k] = int(float(v))
            except Exception: pass
        else:
            try: out[k] = float(v)
            except Exception: pass
    return out

metrics_n = []
for row in metrics:
    r = dict(row)
    for key in list(r):
        if key in {"seed"}:
            r[key] = int(float(r[key]))
        elif key not in {"method"}:
            try:
                r[key] = float(r[key])
            except Exception:
                pass
    metrics_n.append(r)

expected_settings = {
    "learning_rate": 0.001,
    "replay_size": 64,
    "replay_prompt_examples": 0,
    "batch_size": 8,
    "update_steps": 2,
    "warmup_examples": 4,
    "lora_rank": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "lora_target_modules": (
        "q_proj", "k_proj", "v_proj", "self_attn.out_proj"
    ),
    "lora_layers_to_transform": (2, 4, 6, 8, 10, 12),
    "lora_a_learning_rate_scale": 1.0,
    "lm_head_lora_a_learning_rate_scale": None,
    "optimizer_weight_decay": 0.0,
    "optimizer_beta1": 0.9,
    "max_grad_norm": 1.0,
    "lm_head_learning_rate": None,
    "ambiguous_replay_group_weight": 0.05,
    "teacher_temperature": 1.0,
    "reasoning_tokens": 0,
    "target_mode": "causal_fusion",
    "reliable_teacher_weight": 0.05,
    "reliable_decision_weight": 0.05,
    "reliable_behavior_weight": 0.90,
    "ambiguous_teacher_weight": 0.0,
    "ambiguous_decision_weight": 1.0,
    "ambiguous_behavior_weight": 0.0,
    "ambiguous_projection": "causal_support",
    "replay_strategy": "selection_balanced",
    "replay_recency_half_life": 32.0,
    "ambiguous_update_mode": "immediate",
    "force_newest_every_step": True,
    "base_kl_weight": 0.0,
    "behavior_mode": "uncertainty_interrupt_probe",
    "behavior_epsilon": 0.02,
    "behavior_epsilon_half_life": None,
    "exploration_taper_start_step": 160,
    "exploration_taper_half_life": 5.0,
    "archive_probe_mix": 0.0,
    "archive_policy_min_feedback": 0.0,
    "interrupt_probe_mix": 0.15,
    "interrupt_probe_half_life": 80.0,
    "interrupt_probe_max_confidence": 0.60,
    "propensity_weight_mode": "none",
    "propensity_weight_cap": 4.0,
}
serialized_expected_settings = json.loads(json.dumps(expected_settings))
expected_rft_student_settings = {
    "learning_rate": 0.0007,
    "replay_size": 32,
    "replay_prompt_examples": 0,
    "batch_size": 8,
    "update_steps": 2,
    "warmup_examples": 4,
    "lora_rank": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "lora_target_modules": (
        "q_proj", "k_proj", "v_proj", "self_attn.out_proj"
    ),
    "lora_layers_to_transform": (2, 4, 6, 8, 10, 12),
    "lora_a_learning_rate_scale": 1.0,
    "lm_head_lora_a_learning_rate_scale": None,
    "optimizer_weight_decay": 0.0,
    "optimizer_beta1": 0.9,
    "max_grad_norm": 1.0,
    "lm_head_learning_rate": None,
    "ambiguous_replay_group_weight": 0.05,
    "teacher_temperature": 1.0,
    "reasoning_tokens": 0,
    "target_mode": "causal_fusion",
    "reliable_teacher_weight": 0.05,
    "reliable_decision_weight": 0.05,
    "reliable_behavior_weight": 0.90,
    "ambiguous_teacher_weight": 0.0,
    "ambiguous_decision_weight": 1.0,
    "ambiguous_behavior_weight": 0.0,
    "ambiguous_projection": "causal_support",
    "replay_strategy": "selection_balanced",
    "replay_recency_half_life": None,
    "ambiguous_update_mode": "immediate",
    "force_newest_every_step": True,
    "base_kl_weight": 0.0,
    "behavior_mode": "epsilon_greedy",
    "behavior_epsilon": 0.06,
    "behavior_epsilon_half_life": None,
    "exploration_taper_start_step": None,
    "exploration_taper_half_life": None,
    "archive_probe_mix": 0.0,
    "archive_policy_min_feedback": 0.0,
    "interrupt_probe_mix": 0.0,
    "interrupt_probe_half_life": None,
    "interrupt_probe_max_confidence": 1.0,
    "propensity_weight_mode": "none",
    "propensity_weight_cap": 4.0,
}
serialized_expected_rft_student_settings = json.loads(
    json.dumps(expected_rft_student_settings)
)
expected_rft_settings = {
    "student_settings": serialized_expected_rft_student_settings,
    "candidate_count": 1,
    "sampling_temperature": 8.0,
    "sampling_mode": "categorical",
}
expected_reinforce_settings = {
    "learning_rate": 0.0001,
    "batch_size": 8,
    "baseline_step": 0.0,
    "entropy_coef": 1.0,
    "max_grad_norm": 1.0,
    "reward_outcome_map": {
        "OPENED_IMMEDIATELY": 5.0,
        "OPENED_AFTER_DELAY": -1.0,
        "DELETED_NOTIFICATION": -5.0,
        "OPENED_DIGEST": 0.0,
        "DELETED_FROM_DIGEST": -5.0,
        "NO_OBSERVABLE_SELECTION": 0.0,
    },
}
expected_reinforce_training_reward = {
    "source": "matured executed-surface factual outcome only",
    "outcome_map": expected_reinforce_settings["reward_outcome_map"],
    "unknown_selection": (
        "censored before outcome mapping; no gradient target"
    ),
    "reported_metric": (
        "learner-only shaping; rollout observed_feedback_reward and "
        "cumulative_observed_reward use the shared observed_reward map"
    ),
}
expected_reinforce_provenance = {
    "selection_seeds": [0, 1, 2],
    "disjoint_confirmation": False,
    "selection_rule": (
        "strict candidate gate: pooled exact-action correct > Base and "
        "pooled total regret < Base; rank accuracy first, then regret"
    ),
    "interpretation": (
        "user-requested in-sample tuning on the canonical streams; no "
        "disjoint confirmation set"
    ),
}
expected_gold_action_sampling = {
    "distribution": (
        "probability-power temperature scaling of normalized "
        "evaluator-utility weights"
    ),
    "negative_utility_handling": (
        "subtract the event minimum only when it is negative"
    ),
    "temperature": 0.01,
    "exponent": 100.0,
    "formula": (
        "q_i = p_i**(1 / temperature) / "
        "sum_j p_j**(1 / temperature)"
    ),
    "numerical_implementation": (
        "stable log space with exact zero support preserved"
    ),
    "draw_timing": "once per event from an isolated seeded RNG",
}
expected_peft_architecture = {
    "implementation": "peft.LoraConfig + peft.get_peft_model",
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "r": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "self_attn.out_proj"
    ],
    "layers_to_transform": [2, 4, 6, 8, 10, 12],
    "layers_pattern": "layers",
    "bias": "none",
    "init_lora_weights": True,
    "ensure_weight_tying": False,
    "merged_for_serving": False,
}
assert TEACHER_PROMPT_VERSION == "concise-causal-v3", TEACHER_PROMPT_VERSION
assert DATASET_VERSION == "semantic-title-body-sharp-t001", DATASET_VERSION
assert DATASET_NUMPY_VERSION == "2.4.6", DATASET_NUMPY_VERSION
assert PREFERENCE_SAMPLING_TEMPERATURE == 0.01, (
    PREFERENCE_SAMPLING_TEMPERATURE
)
assert METHODS == (
    "Base",
    "ICL",
    "RAG",
    "REINFORCE",
    "RFT",
    "Online-SDFT",
), METHODS
assert DEFAULT_SDFT_SETTINGS.to_dict() == expected_settings, (
    DEFAULT_SDFT_SETTINGS.to_dict()
)
assert DEFAULT_REINFORCE_SETTINGS.to_dict() == expected_reinforce_settings, (
    DEFAULT_REINFORCE_SETTINGS.to_dict()
)
assert (
    DEFAULT_RFT_SETTINGS.student_settings.to_dict()
    == expected_rft_student_settings
)
assert json.loads(json.dumps(DEFAULT_RFT_SETTINGS.to_dict())) == (
    expected_rft_settings
)
assert expected_settings["update_steps"] == 2, expected_settings
assert (
    len(expected_settings["lora_target_modules"])
    * len(expected_settings["lora_layers_to_transform"])
    * 2
    == 48
)
assert all(c.get("device") == "mps" for c in configs), configs
assert all(
    c.get("teacher_prompt_version") == TEACHER_PROMPT_VERSION
    for c in configs
), configs
assert all(
    c.get("online_sdft_settings") == serialized_expected_settings
    for c in configs
), configs
assert all(c.get("sdft_update_steps") == 2 for c in configs), configs
assert all(c.get("sdft_replay_prompt_examples") == 0 for c in configs), configs
assert all(
    c.get("sdft_replay_prompt_policy")
    == "disabled; serving uses only learned finite parameters"
    for c in configs
), configs
assert all(c.get("prompt_style") == "causal_demos" for c in configs), configs
assert all(c.get("icl_examples") == 3 for c in configs), configs
assert all(c.get("rag_examples") == 3 for c in configs), configs
assert all(c.get("rag_text_weight") == 0.5 for c in configs), configs
assert REINFORCE_BATCH_SIZE == 8
assert all(c.get("reinforce_batch_size") == 8 for c in configs), configs
assert REINFORCE_LR == 0.0001
assert all(c.get("reinforce_lr") == REINFORCE_LR for c in configs), configs
assert REINFORCE_BASELINE_STEP == 0.0
assert REINFORCE_ENTROPY_COEF == 1.0
assert REINFORCE_MAX_GRAD_NORM == 1.0
assert REINFORCE_TRAINING_OUTCOME_REWARDS == (
    expected_reinforce_settings["reward_outcome_map"]
)
assert all(
    c.get("online_reinforce_settings") == expected_reinforce_settings
    for c in configs
), configs
assert all(
    c.get("reinforce_baseline") == "fixed zero causal baseline; step=0.0"
    and c.get("reinforce_entropy_coef") == REINFORCE_ENTROPY_COEF
    and c.get("reinforce_max_grad_norm") == REINFORCE_MAX_GRAD_NORM
    for c in configs
), configs
assert all(
    c.get("reinforce_training_reward")
    == expected_reinforce_training_reward
    for c in configs
), configs
assert all(
    c.get("adaptive_default_configuration_provenance", {}).get("REINFORCE")
    == expected_reinforce_provenance
    for c in configs
), configs
assert all(
    c.get("reinforce_initialization")
    == "the common PEFT LoRA initialization restored before this arm"
    for c in configs
), configs
assert RFT_LR == 0.0007
assert RFT_PROTOCOL_VERSION == "teacher-categorical-k1-temperature8-singleton"
assert RFT_SETTINGS_PROVENANCE == (
    "fixed-temperature8-lr7e-4-same-lora-architecture"
)
assert RFT_CANDIDATE_COUNT == 1
assert RFT_SAMPLING_TEMPERATURE == 8.0
assert RFT_SAMPLING_MODE == "categorical"
assert RFT_CANDIDATE_RNG_OFFSET == 83
assert all(c.get("rft_protocol_version") == RFT_PROTOCOL_VERSION for c in configs)
assert all(
    c.get("rft_settings_provenance") == RFT_SETTINGS_PROVENANCE
    for c in configs
)
assert all(
    c.get("online_rft_settings") == expected_rft_settings
    for c in configs
)
assert all(
    c.get("online_rft_peft_architecture") == expected_peft_architecture
    and c.get("online_reinforce_peft_architecture") == expected_peft_architecture
    and c.get("online_sdft_peft_architecture") == expected_peft_architecture
    for c in configs
), configs
assert all(
    c.get("online_rft_trainable_parameters")
    == c.get("reinforce_trainable_parameters")
    == c.get("online_sdft_trainable_parameters")
    == 172_032
    for c in configs
), configs
assert all(
    c.get("reinforce_capacity_match", {}).get("same_physical_model") is True
    and c.get("reinforce_capacity_match", {}).get("same_frozen_base_model") is True
    and c.get("reinforce_capacity_match", {}).get("same_lora_architecture") is True
    and c.get("reinforce_capacity_match", {}).get(
        "same_adapter_parameter_count"
    ) is True
    and c.get("reinforce_capacity_match", {}).get(
        "same_adapter_initialization"
    ) is True
    and c.get("reinforce_capacity_match", {}).get(
        "adapter_reset_before_arm"
    ) is True
    and c.get("reinforce_capacity_match", {}).get("uses_hindsight_teacher")
    is False
    for c in configs
), configs
assert all(
    "adapter is reset to the identical initialization before every method"
    in c.get("adaptive_model_sharing", "")
    for c in configs
), configs
assert all(
    c.get("rft_candidate_policy", "").startswith(
        "K=1 categorical sample at temperature 8"
    )
    and "adapter disabled" in c.get("rft_candidate_policy", "")
    for c in configs
), configs
assert all(
    c.get("rft_candidate_sampler") == {
        "scheme": RFT_CANDIDATE_SAMPLER,
        "key_fields": [
            "seed",
            "event_id",
            "t",
            "rft_candidate_rng_offset",
        ],
        "inverse_cdf": True,
        "stateful_rng": False,
    }
    for c in configs
), configs
assert all(
    "reliable singleton" in c.get("rft_acceptance_filter", "")
    and "ambiguous digest open" in c.get("rft_acceptance_filter", "")
    and "UNKNOWN is censored" in c.get("rft_acceptance_filter", "")
    for c in configs
), configs
assert all(
    "one-hot cross-entropy" in c.get("rft_target", "")
    and "never enter replay" in c.get("rft_target", "")
    for c in configs
), configs
assert all(
    c.get("rft_capacity_match", {}).get("settings_source")
    == "online_rft_settings.student_settings"
    and
    c.get("rft_capacity_match", {}).get("same_frozen_base_model") is True
    and c.get("rft_capacity_match", {}).get("same_lora_architecture") is True
    and c.get("rft_capacity_match", {}).get("same_adapter_parameter_count") is True
    and c.get("rft_capacity_match", {}).get("same_adapter_initialization") is True
    and c.get("rft_capacity_match", {}).get("same_replay_schedule") is False
    and c.get("rft_capacity_match", {}).get(
        "same_optimizer_hyperparameters"
    ) is False
    and c.get("rft_capacity_match", {}).get("same_adapter_disabled_teacher") is True
    and c.get("rft_candidate_rng_offset") == RFT_CANDIDATE_RNG_OFFSET
    for c in configs
), configs
assert all(
    c["online_sdft_settings"]["learning_rate"] == 0.001
    and c["online_rft_settings"]["student_settings"]["learning_rate"]
    == 0.0007
    and {
        key
        for key, value in c["online_sdft_settings"].items()
        if c["online_rft_settings"]["student_settings"][key] != value
    }
    == {
        "learning_rate",
        "replay_size",
        "replay_recency_half_life",
        "behavior_mode",
        "behavior_epsilon",
        "exploration_taper_start_step",
        "exploration_taper_half_life",
        "interrupt_probe_mix",
        "interrupt_probe_half_life",
        "interrupt_probe_max_confidence",
    }
    for c in configs
), configs
assert all(
    c.get("student_backbone") == "frozen Liquid LFM base weights"
    and c.get("student_backbone_trainable_parameters") == 0
    and c.get("online_sdft_student", "").startswith(
        "PEFT LoRA adapter trained online"
    )
    for c in configs
), configs
assert all(
    "exactly 8 newly matured known factual-outcome callbacks" in c.get(
        "reinforce_optimizer", ""
    )
    and "learner reward from reinforce_training_reward outcome map"
    in c.get("reinforce_optimizer", "")
    and "no replay" in c.get("reinforce_optimizer", "")
    and "incomplete horizon batch not flushed" in c.get(
        "reinforce_optimizer", ""
    )
    for c in configs
), configs
assert all(
    c.get("observed_reward") == OBSERVED_OUTCOME_REWARDS
    for c in configs
), configs
assert all(c.get("dataset_version") == DATASET_VERSION for c in configs), configs
assert all(
    c.get("gold_action_sampling") == expected_gold_action_sampling
    for c in configs
), configs
assert all(
    c.get("dataset_numpy_version") == DATASET_NUMPY_VERSION
    and c.get("runtime_numpy_version") == DATASET_NUMPY_VERSION
    for c in configs
), configs
assert all(
    c.get("dataset_fingerprint") == DEFAULT_ENVIRONMENT.stream_fingerprint([seed])
    for seed, c in enumerate(configs)
), configs
assert all(
    set(c.get("method_dataset_fingerprints", {}).values())
    == {c["dataset_fingerprint"]}
    and set(c.get("method_dataset_fingerprints", {})) == set(METHODS)
    for c in configs
), configs
normalized_configs = [
    {
        key: value
        for key, value in c.items()
        if key
        not in {
            "seeds",
            "seed_start",
            "dataset_fingerprint",
            "method_dataset_fingerprints",
        }
    }
    for c in configs
]
assert all(c == normalized_configs[0] for c in normalized_configs[1:]), configs
config = dict(configs[0])
config["seeds"] = 3
config["seed_start"] = 0
combined_fingerprint = DEFAULT_ENVIRONMENT.stream_fingerprint(range(3))
assert combined_fingerprint == (
    "986cdf1a7d5fcc04c2b33f1bf90a1fc4f24a97ee85e663370382d8a67e4c932d"
), combined_fingerprint
config["dataset_fingerprint"] = combined_fingerprint
config["method_dataset_fingerprints"] = {
    method: combined_fingerprint for method in METHODS
}
config["source_fingerprint"] = source_fingerprint

assert len(metrics) == 3 * len(METHODS), len(metrics)
assert len(curves) == 3 * len(METHODS) * 240, len(curves)
assert len(rollouts) == 3 * len(METHODS) * 240, len(rollouts)
expected_seed_methods = {
    (seed, method) for seed in (0, 1, 2) for method in METHODS
}
assert Counter(
    (int(float(row["seed"])), row["method"]) for row in metrics
) == Counter({key: 1 for key in expected_seed_methods}), metrics
assert Counter(
    (int(float(row["seed"])), row["method"]) for row in curves
) == Counter({key: 240 for key in expected_seed_methods}), curves
assert Counter(
    (int(row["seed"]), row["method"]) for row in rollouts
) == Counter({key: 240 for key in expected_seed_methods}), rollouts


def assert_finite(value, path="root"):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        assert math.isfinite(float(value)), (path, value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


curves_n = [numify(c) for c in curves]
assert_finite(metrics_n, "metrics")
assert_finite(curves_n, "curves")
assert_finite(rollouts, "rollouts")

# Audit the raw paired Base/REINFORCE decisions before summarization. Accuracy
# is strictly action == sampled hidden preference; utility enters only regret.
expected_rollout_keys = {
    (seed, step) for seed in (0, 1, 2) for step in range(1, 241)
}
base_rollouts = [row for row in rollouts if row["method"] == "Base"]
reinforce_rollouts = [
    row for row in rollouts if row["method"] == "REINFORCE"
]
assert len(base_rollouts) == 720, len(base_rollouts)
assert len(reinforce_rollouts) == 720, len(reinforce_rollouts)


def unique_seed_step(rows, method):
    counts = Counter((int(row["seed"]), int(row["t"])) for row in rows)
    assert counts == Counter({key: 1 for key in expected_rollout_keys}), (
        method,
        counts,
    )
    return {
        (int(row["seed"]), int(row["t"])): row
        for row in rows
    }


base_by_key = unique_seed_step(base_rollouts, "Base")
reinforce_by_key = unique_seed_step(reinforce_rollouts, "REINFORCE")
raw_correct = {"Base": 0, "REINFORCE": 0}
raw_total_regret = {"Base": 0.0, "REINFORCE": 0.0}
for key in sorted(expected_rollout_keys):
    paired = (base_by_key[key], reinforce_by_key[key])
    assert paired[0]["event_id"] == paired[1]["event_id"], (key, paired)
    assert (
        paired[0]["gold_action_scoring_only"]
        == paired[1]["gold_action_scoring_only"]
    ), (key, paired)
    for row in paired:
        method = row["method"]
        assert row["action"] in ACTIONS, row
        assert row["gold_action_scoring_only"] in ACTIONS, row
        expected_correct = int(
            row["action"] == row["gold_action_scoring_only"]
        )
        assert row["correct_online"] in {0, 1}, row
        assert int(row["correct_online"]) == expected_correct, row
        raw_correct[method] += expected_correct
        raw_total_regret[method] += float(row["step_regret"])


def metric_for(method, seed):
    matches = [
        row
        for row in metrics_n
        if row["method"] == method and int(row["seed"]) == seed
    ]
    assert len(matches) == 1, (method, seed, matches)
    return matches[0]


for method, rows in (("Base", base_rollouts), ("REINFORCE", reinforce_rollouts)):
    for seed in (0, 1, 2):
        seed_rows = sorted(
            (row for row in rows if int(row["seed"]) == seed),
            key=lambda row: int(row["t"]),
        )
        metric = metric_for(method, seed)
        seed_correct = sum(int(row["correct_online"]) for row in seed_rows)
        seed_regret = sum(float(row["step_regret"]) for row in seed_rows)
        assert math.isclose(
            float(metric["online_accuracy"]),
            seed_correct / 240,
            rel_tol=0.0,
            abs_tol=1e-12,
        ), (method, seed, metric, seed_correct)
        assert math.isclose(
            float(metric["cum_regret"]),
            seed_regret,
            rel_tol=0.0,
            abs_tol=1e-10,
        ), (method, seed, metric, seed_regret)
        assert math.isclose(
            float(seed_rows[-1]["cum_regret"]),
            seed_regret,
            rel_tol=0.0,
            abs_tol=1e-10,
        ), (method, seed, seed_rows[-1], seed_regret)

# The promoted arm must pass the same strict, user-requested in-sample gate
# used by the tuner. These are pooled raw counts/totals, not rounded means.
assert raw_correct["REINFORCE"] > raw_correct["Base"], raw_correct
assert (
    raw_total_regret["REINFORCE"] < raw_total_regret["Base"]
), raw_total_regret

# The shaped scalar consumed by REINFORCE is keyed only by a matured factual
# executed-surface outcome. Public observed-reward metrics retain the shared
# environment map and are audited independently below.
training_reward_map = expected_reinforce_training_reward["outcome_map"]
assert training_reward_map != OBSERVED_OUTCOME_REWARDS
assert set(training_reward_map) == set(OBSERVED_OUTCOME_REWARDS)
for seed in (0, 1, 2):
    seed_rows = sorted(
        (row for row in reinforce_rollouts if int(row["seed"]) == seed),
        key=lambda row: int(row["t"]),
    )
    cumulative_shared_reward = 0.0
    completed_batches = {}
    unflushed = []
    for row in seed_rows:
        feedback = row["feedback"]
        outcome = feedback["outcome"]
        selection = feedback["observed_user_selection"]
        assert outcome in OBSERVED_OUTCOME_REWARDS, row
        shared_reward = float(OBSERVED_OUTCOME_REWARDS[outcome])
        assert math.isclose(
            float(feedback["reward"]),
            shared_reward,
            rel_tol=0.0,
            abs_tol=1e-15,
        ), row
        assert math.isclose(
            float(row["observed_feedback_reward"]),
            shared_reward,
            rel_tol=0.0,
            abs_tol=1e-15,
        ), row
        cumulative_shared_reward += shared_reward
        assert math.isclose(
            float(row["cum_observed_reward"]),
            cumulative_shared_reward,
            rel_tol=0.0,
            abs_tol=1e-12,
        ), row

        status = row["lesson_status"]
        training_reward = row["reinforce_training_reward"]
        released = row["feedback_released_at_minute"] is not None
        if status == "pending_after_horizon":
            assert not released, row
            assert training_reward is None, row
            assert row["reinforce_batch_position"] is None, row
            assert row["reinforce_update_index"] is None, row
            continue
        assert released, row
        if selection == "UNKNOWN":
            assert outcome == "NO_OBSERVABLE_SELECTION", row
            assert status == "censored_no_update", row
            assert training_reward is None, row
            assert row["reinforce_batch_position"] is None, row
            assert row["reinforce_update_index"] is None, row
            continue

        assert selection in ACTIONS, row
        assert outcome != "NO_OBSERVABLE_SELECTION", row
        assert math.isclose(
            float(training_reward),
            float(training_reward_map[outcome]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ), row
        position = int(row["reinforce_batch_position"])
        if status == "feedback_applied":
            update_index = int(row["reinforce_update_index"])
            completed_batches.setdefault(update_index, []).append(row)
        else:
            assert status == "feedback_gradient_unflushed_at_horizon", row
            assert row["reinforce_update_index"] is None, row
            unflushed.append(row)

    assert sorted(completed_batches) == list(
        range(1, len(completed_batches) + 1)
    ), completed_batches
    for update_index, batch_rows in completed_batches.items():
        assert len(batch_rows) == 8, (update_index, batch_rows)
        assert sorted(
            int(row["reinforce_batch_position"]) for row in batch_rows
        ) == list(range(1, 9)), (update_index, batch_rows)
    assert len(unflushed) < 8, unflushed
    assert sorted(
        int(row["reinforce_batch_position"]) for row in unflushed
    ) == list(range(1, len(unflushed) + 1)), unflushed
    metric = metric_for("REINFORCE", seed)
    assert math.isclose(
        float(metric["cumulative_observed_reward"]),
        cumulative_shared_reward,
        rel_tol=0.0,
        abs_tol=1e-12,
    ), (metric, cumulative_shared_reward)

# write merged raw artifacts
with (out / "per_seed_metrics.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()), lineterminator="\n")
    w.writeheader(); w.writerows(metrics)
with (out / "learning_curves.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(curves[0].keys()), lineterminator="\n")
    w.writeheader(); w.writerows(curves)
with (out / "rollouts.jsonl").open("w") as f:
    for row in rollouts:
        f.write(json.dumps(row) + "\n")

summary = write_compact_results(out, config, metrics_n, rollouts)
assert_finite(summary, "summary")
for method in ("Base", "REINFORCE"):
    assert math.isclose(
        float(summary[method]["online_accuracy"]["mean"]),
        raw_correct[method] / 720,
        rel_tol=0.0,
        abs_tol=1e-12,
    ), (method, summary[method], raw_correct)
    assert math.isclose(
        float(summary[method]["cum_regret"]["mean"]),
        raw_total_regret[method] / 3,
        rel_tol=0.0,
        abs_tol=1e-10,
    ), (method, summary[method], raw_total_regret)
write_figures(summary, curves_n, fig)

expected_fusion = {
    "reliable_singleton": {
        "teacher": expected_settings["reliable_teacher_weight"],
        "decision": expected_settings["reliable_decision_weight"],
        "behavior": expected_settings["reliable_behavior_weight"],
    },
    "ambiguous_digest_open": {
        "teacher": expected_settings["ambiguous_teacher_weight"],
        "decision": expected_settings["ambiguous_decision_weight"],
        "behavior": expected_settings["ambiguous_behavior_weight"],
    },
}
sdft_rollouts = [row for row in rollouts if row["method"] == "Online-SDFT"]
assert len(sdft_rollouts) == 720, len(sdft_rollouts)
assert Counter(
    (int(row["seed"]), int(row["t"])) for row in sdft_rollouts
) == Counter(
    {(seed, step): 1 for seed in (0, 1, 2) for step in range(1, 241)}
), sdft_rollouts

online_correct = 0
for row in sdft_rollouts:
    step = int(row["t"])
    assert row["behavior_mode"] == expected_settings["behavior_mode"], row
    assert math.isclose(
        float(row["behavior_epsilon"]),
        expected_settings["behavior_epsilon"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ), row

    taper_start = expected_settings["exploration_taper_start_step"]
    taper_half_life = expected_settings["exploration_taper_half_life"]
    expected_taper_weight = (
        1.0
        if step <= taper_start
        else 2.0 ** (-(step - taper_start) / taper_half_life)
    )
    assert math.isclose(
        float(row["exploration_taper_weight"]),
        expected_taper_weight,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ), row

    student_values = [float(row["student_probs"][action]) for action in ACTIONS]
    assert all(math.isfinite(value) for value in student_values), row
    greedy_index = max(
        range(len(ACTIONS)),
        key=student_values.__getitem__,
    )
    epsilon = expected_settings["behavior_epsilon"]
    expected_behavior = [epsilon / len(ACTIONS)] * len(ACTIONS)
    expected_behavior[greedy_index] += 1.0 - epsilon

    normalized_student = [max(value, 1e-8) for value in student_values]
    student_total = sum(normalized_student)
    normalized_student = [value / student_total for value in normalized_student]
    if max(normalized_student) <= expected_settings[
        "interrupt_probe_max_confidence"
    ]:
        effective_probe_mix = expected_settings[
            "interrupt_probe_mix"
        ] * 2.0 ** (
            -(step - 1) / expected_settings["interrupt_probe_half_life"]
        )
        expected_behavior = [
            (1.0 - effective_probe_mix) * value
            for value in expected_behavior
        ]
        expected_behavior[ACTIONS.index("INTERRUPT")] += effective_probe_mix
        behavior_total = sum(expected_behavior)
        expected_behavior = [
            value / behavior_total for value in expected_behavior
        ]

    if step > taper_start:
        expected_behavior = [
            expected_taper_weight * value
            + (1.0 - expected_taper_weight) * int(index == greedy_index)
            for index, value in enumerate(expected_behavior)
        ]
        behavior_total = sum(expected_behavior)
        expected_behavior = [
            value / behavior_total for value in expected_behavior
        ]

    observed_behavior = [
        float(row["behavior_probs"][action]) for action in ACTIONS
    ]
    assert all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in observed_behavior
    ), row
    assert math.isclose(
        sum(observed_behavior),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ), row
    assert all(
        math.isclose(
            observed,
            expected,
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
        for observed, expected in zip(observed_behavior, expected_behavior)
    ), row

    expected_correct = int(
        row["action"] == row["gold_action_scoring_only"]
    )
    assert row["correct_online"] in {0, 1}, row
    assert int(row["correct_online"]) == expected_correct, row
    online_correct += expected_correct

assert online_correct >= 505, online_correct
sdft_metrics = {
    int(row["seed"]): row
    for row in metrics_n
    if row["method"] == "Online-SDFT"
}
assert set(sdft_metrics) == {0, 1, 2}, sdft_metrics
for seed, metric_row in sdft_metrics.items():
    seed_rollouts = [row for row in sdft_rollouts if int(row["seed"]) == seed]
    seed_accuracy = sum(
        int(row["correct_online"]) for row in seed_rollouts
    ) / len(seed_rollouts)
    assert math.isclose(
        float(metric_row["online_accuracy"]),
        seed_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ), (metric_row, seed_accuracy)
assert math.isclose(
    float(summary["Online-SDFT"]["online_accuracy"]["mean"]),
    online_correct / len(sdft_rollouts),
    rel_tol=0.0,
    abs_tol=1e-12,
), (summary["Online-SDFT"], online_correct)

for row in sdft_rollouts:
    status = row["lesson_status"]
    reliability = row["sdft_evidence_reliability"]
    weights = row["sdft_fusion_weights"]
    assert row["sdft_prompt_examples_used"] == 0, row
    if status in {"soft_target_buffered", "soft_target_applied"}:
        assert reliability in expected_fusion, row
        assert weights == expected_fusion[reliability], row
        expected_support_size = 1 if reliability == "reliable_singleton" else 2
        assert len(row["causal_support"]) == expected_support_size, row
        expected_updates = (
            expected_settings["update_steps"]
            if status == "soft_target_applied"
            else 0
        )
        assert row["sdft_updates_applied"] == expected_updates, row
    elif status == "censored_no_update":
        assert reliability == "censored_unknown", row
        assert weights is None, row
        assert row["sdft_updates_applied"] == 0, row
    else:
        assert status == "pending_after_horizon", row
        assert reliability is None and weights is None, row
        assert row["sdft_updates_applied"] is None, row

for seed in (0, 1, 2):
    buffered = [
        row
        for row in sdft_rollouts
        if row["seed"] == seed and row["lesson_status"] == "soft_target_buffered"
    ]
    assert len(buffered) == expected_settings["warmup_examples"] - 1, buffered

rft_rollouts = [row for row in rollouts if row["method"] == "RFT"]
for row in rft_rollouts:
    status = row["lesson_status"]
    reason = row["rft_reason"]
    candidate = row["rft_candidate_action"]
    accepted = row["rft_accepted"]
    if status in {"rft_target_buffered", "rft_target_applied"}:
        assert accepted is True and reason == "accepted", row
        assert candidate in row["causal_support"], row
        assert len(row["causal_support"]) == 1, row
        expected_updates = (
            expected_rft_student_settings["update_steps"]
            if status == "rft_target_applied"
            else 0
        )
        assert row["rft_updates_applied"] == expected_updates, row
    elif status == "rft_rejected_ambiguous_support":
        assert accepted is False and reason == "ambiguous_unverified", row
        assert candidate is not None and len(row["causal_support"]) == 2, row
        assert row["rft_updates_applied"] == 0, row
    elif status == "rft_rejected_teacher_candidate":
        assert accepted is False and reason == "teacher_mismatch", row
        assert candidate not in row["causal_support"], row
        assert len(row["causal_support"]) == 1, row
        assert row["rft_updates_applied"] == 0, row
    elif status == "censored_no_update":
        assert accepted is None and reason == "censored_unknown", row
        assert candidate is None and len(row["causal_support"]) == 3, row
        assert row["rft_updates_applied"] == 0, row
    else:
        assert status == "pending_after_horizon", row
        assert accepted is None and reason is None and candidate is None, row
        assert row["rft_updates_applied"] is None, row

rft_diagnostics = summarize_rft_diagnostics(rollouts)
rft_total = rft_diagnostics["total"]
assert rft_total["attempted"] == (
    rft_total["accepted"] + rft_total["rejected"]
)
assert rft_total["rejected"] == sum(
    rft_total["rejection_reasons"].values()
)
assert set(rft_total["rejection_reasons"]) <= {
    "ambiguous_unverified",
    "teacher_mismatch",
}
assert rft_total["accepted"] > 0
assert 0.0 < rft_total["acceptance_rate"] < 1.0
for seed in (0, 1, 2):
    seed_rows = [row for row in rft_rollouts if row["seed"] == seed]
    accepted_rows = [row for row in seed_rows if row["rft_accepted"] is True]
    buffered = [
        row for row in accepted_rows if row["lesson_status"] == "rft_target_buffered"
    ]
    assert len(buffered) == min(
        len(accepted_rows),
        expected_rft_student_settings["warmup_examples"] - 1,
    ), buffered

def verify_source_fingerprint():
    assert source_fingerprint.get("algorithm") == "sha256", source_fingerprint
    expected_files = source_fingerprint.get("files")
    assert isinstance(expected_files, dict) and expected_files, source_fingerprint
    observed_files = {
        relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        for relative in expected_files
    }
    aggregate = hashlib.sha256(
        json.dumps(
            observed_files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    changed = sorted(
        relative
        for relative, expected_hash in expected_files.items()
        if observed_files[relative] != expected_hash
    )
    assert not changed, f"benchmark source changed before promotion: {changed}"
    assert aggregate == source_fingerprint["sha256"], (
        aggregate,
        source_fingerprint,
    )
    print("SOURCE_FINGERPRINT_PROMOTION_VERIFIED", aggregate)


# Promote only after the complete staged set has passed every assertion.
verify_source_fingerprint()
canonical_out = repo / "outputs" / "bandit"
canonical_fig = repo / "figures"
canonical_out.mkdir(parents=True, exist_ok=True)
canonical_fig.mkdir(parents=True, exist_ok=True)
for name in (
    "rollouts.jsonl",
    "learning_curves.csv",
    "per_seed_metrics.csv",
    "summary.json",
    "qualitative_examples.json",
):
    shutil.copy2(out / name, canonical_out / name)
for name in ("bandit_accuracy.png", "bandit_learning_curves.png"):
    shutil.copy2(fig / name, canonical_fig / name)
print("MERGED", {m: summary[m]["online_accuracy"] for m in METHODS})
print("REINFORCE_PROMOTION_GATE", {
    "raw_correct": raw_correct,
    "raw_total_regret": raw_total_regret,
})
print("device", config["device"])
print("promoted_from", stage)
PY

echo "ALL_SEEDS_DONE $(date -u +%H:%M:%SZ)"
