#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "usage: $0 <publisher-package> <case-id> <event-id> <category> <importance> <regime> <title> [body] [surface] [timeout-after-ms] [semantic-delay-minutes]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
ADB="${ADB:-$ANDROID_SDK_ROOT/platform-tools/adb}"

PACKAGE_NAME="$1"
CASE_ID="$2"
EVENT_ID="$3"
CATEGORY="$4"
IMPORTANCE="$5"
REGIME="$6"
TITLE="$7"
BODY="${8:-Real Android notification posted by a separate demo source app.}"
SURFACE="${9:-standard}"
TIMEOUT_AFTER_MILLIS="${10:-0}"
SEMANTIC_DELAY_MINUTES="${11:-0}"

# `adb shell` joins its argv into a remote shell command. Preserve spaces and
# punctuation by passing literal single-quoted values through that shell.
quote_for_adb_shell() {
  local value="$1"
  value="${value//\'/\'\\\'\'}"
  printf "'%s'" "$value"
}

COMMON_EXTRAS=(
  --es case_id "$(quote_for_adb_shell "$CASE_ID")"
  --es event_id "$(quote_for_adb_shell "$EVENT_ID")"
  --es category "$(quote_for_adb_shell "$CATEGORY")"
  --es importance "$(quote_for_adb_shell "$IMPORTANCE")"
  --es regime "$(quote_for_adb_shell "$REGIME")"
  --es surface "$(quote_for_adb_shell "$SURFACE")"
  --es title "$(quote_for_adb_shell "$TITLE")"
  --es body "$(quote_for_adb_shell "$BODY")"
  --el timeout_after_millis "$TIMEOUT_AFTER_MILLIS"
  --ei semantic_delay_minutes "$SEMANTIC_DELAY_MINUTES"
)

if [[ "$SURFACE" == "foreground-service" || "$SURFACE" == "call" ]]; then
  # Android 12+ blocks foreground-service launch from a background receiver.
  # A shell-launched visible fixture activity is an honest user-visible start.
  "$ADB" shell am start -W \
    --activity-clear-top \
    -n "$PACKAGE_NAME/ai.onlinesdft.publisher.MainActivity" \
    -a ai.onlinesdft.publisher.START_SURFACE \
    "${COMMON_EXTRAS[@]}"
else
  "$ADB" shell am broadcast \
    --include-stopped-packages \
    -a ai.onlinesdft.publisher.POST_CASE \
    -p "$(quote_for_adb_shell "$PACKAGE_NAME")" \
    "${COMMON_EXTRAS[@]}"
fi
