#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
ADB="${ADB:-$ANDROID_SDK_ROOT/platform-tools/adb}"
RUN_SUFFIX="${1:-$(date +%s)}"
PROOF_TAG="OnlineSdftProof"
PUBLISHER_PACKAGES=(
  ai.onlinesdft.publisher.chat
  ai.onlinesdft.publisher.calendar
  ai.onlinesdft.publisher.mail
)

cleanup_fixtures() {
  local package_name
  for package_name in "${PUBLISHER_PACKAGES[@]}"; do
    "$ADB" shell am broadcast --include-stopped-packages \
      -a ai.onlinesdft.publisher.CLEANUP \
      -p "$package_name" >/dev/null 2>&1 || true
  done
}

cleanup_on_exit() {
  if [[ "${KEEP_NOTIFICATION_FIXTURES:-0}" != "1" ]]; then
    cleanup_fixtures
  fi
}

SDK_LEVEL="$($ADB shell getprop ro.build.version.sdk | tr -d '\r')"
if [[ ! "$SDK_LEVEL" =~ ^[0-9]+$ ]] || (( SDK_LEVEL < 31 )); then
  printf 'All-types proof requires Android API 31+ for genuine CallStyle (found %s)\n' \
    "$SDK_LEVEL" >&2
  exit 1
fi

# Make repeated runs hermetic. Set KEEP_NOTIFICATION_FIXTURES=1 to retain the
# final surfaces for manual notification-shade inspection after a passing run.
cleanup_fixtures
trap cleanup_on_exit EXIT INT TERM

ONGOING_EVENT="all-types-ongoing-$RUN_SUFFIX"
FGS_EVENT="all-types-fgs-$RUN_SUFFIX"
CALL_EVENT="all-types-call-$RUN_SUFFIX"
MEDIA_EVENT="all-types-media-$RUN_SUFFIX"
STANDARD_EVENT="all-types-standard-$RUN_SUFFIX"

wait_for_route() {
  local event_id="$1"
  local deadline=$(( $(date +%s) + 20 ))
  while (( $(date +%s) <= deadline )); do
    if "$ADB" logcat -d -v brief -s "$PROOF_TAG:I" '*:S' |
      grep -Fq "ROUTE_COMMITTED event_id=$event_id "; then
      return 0
    fi
    sleep 0.25
  done
  printf 'Timed out waiting for %s\n' "$event_id" >&2
  return 1
}

wait_for_listener() {
  local deadline=$(( $(date +%s) + 20 ))
  while (( $(date +%s) <= deadline )); do
    if "$ADB" shell dumpsys activity services ai.onlinesdft.router.debug |
      grep -Fq 'hasBound=true'; then
      return 0
    fi
    sleep 0.25
  done
  printf 'Timed out waiting for the router notification listener\n' >&2
  return 1
}

notification_block() {
  local event_id="$1"
  "$ADB" shell dumpsys notification --noredact | awk -v needle="event_id=String ($event_id)" '
    /^    NotificationRecord/ {
      if (matched) { print record; exit }
      record = $0 ORS
      matched = index($0, needle) > 0
      next
    }
    {
      record = record $0 ORS
      if (index($0, needle) > 0) matched = 1
    }
    END { if (matched) print record }
  '
}

"$ADB" shell am start -n \
  ai.onlinesdft.router.debug/ai.onlinesdft.router.MainActivity >/dev/null
wait_for_listener
# Binding becomes visible just before Android finishes the listener's active-set
# replay. Let that ordered replay drain so new fixtures retain their event IDs.
sleep 2
"$ADB" logcat -c

"$SCRIPT_DIR/post_case.sh" \
  ai.onlinesdft.publisher.chat ongoing-sync "$ONGOING_EVENT" status 0.55 weekday \
  "Chat backup in progress" "Uploading encrypted conversation history." ongoing

"$SCRIPT_DIR/post_case.sh" \
  ai.onlinesdft.publisher.mail foreground-index "$FGS_EVENT" status 0.60 weekday \
  "Mail index is updating" "A genuine Android foreground service is running." foreground-service

"$SCRIPT_DIR/post_case.sh" \
  ai.onlinesdft.publisher.calendar incoming-call "$CALL_EVENT" call 0.98 on-call \
  "Maya is calling" "Incoming design-review call." call

"$SCRIPT_DIR/post_case.sh" \
  ai.onlinesdft.publisher.chat media-playback "$MEDIA_EVENT" media 0.70 off-hours \
  "Design Systems podcast" "Playing episode 42." media

"$SCRIPT_DIR/post_case.sh" \
  ai.onlinesdft.publisher.mail ordinary-message "$STANDARD_EVENT" mail 0.45 weekday \
  "Ordinary project update" "A controllable comparison notification." standard

"$ADB" shell am start -n ai.onlinesdft.router.debug/ai.onlinesdft.router.MainActivity >/dev/null

for event_id in \
  "$ONGOING_EVENT" "$FGS_EVENT" "$CALL_EVENT" "$MEDIA_EVENT" "$STANDARD_EVENT"
do
  wait_for_route "$event_id"
done

PROOF_LOGS="$("$ADB" logcat -d -v brief -s "$PROOF_TAG:I" OnlineSdftPublisher:I '*:S')"
grep -Eq "ROUTE_COMMITTED event_id=$ONGOING_EVENT .*constraint=ongoing .*ongoing=true .*thread=sdft-notification-events" <<<"$PROOF_LOGS"
grep -Eq "ROUTE_COMMITTED event_id=$FGS_EVENT .*constraint=foreground_service .*fgs=true .*thread=sdft-notification-events" <<<"$PROOF_LOGS"
grep -Eq "ROUTE_COMMITTED event_id=$CALL_EVENT .*constraint=call .*call=true .*thread=sdft-notification-events" <<<"$PROOF_LOGS"
grep -Eq "ROUTE_COMMITTED event_id=$MEDIA_EVENT .*constraint=media .*media=true .*thread=sdft-notification-events" <<<"$PROOF_LOGS"
grep -Eq "ROUTE_COMMITTED event_id=$STANDARD_EVENT .*constraint=none .*thread=sdft-notification-events" <<<"$PROOF_LOGS"
if grep -Eq 'POST_FAILED|FATAL EXCEPTION|Not posted|startForegroundService\(\) not allowed' <<<"$PROOF_LOGS"; then
  printf 'Android rejected an all-types fixture\n' >&2
  exit 1
fi

ONGOING_BLOCK="$(notification_block "$ONGOING_EVENT")"
FGS_BLOCK="$(notification_block "$FGS_EVENT")"
CALL_BLOCK="$(notification_block "$CALL_EVENT")"
MEDIA_BLOCK="$(notification_block "$MEDIA_EVENT")"
grep -Fq 'flags=0x2' <<<"$ONGOING_BLOCK"
grep -Fq 'FLAG_FOREGROUND_SERVICE' <<<"$("$ADB" shell dumpsys activity services ai.onlinesdft.publisher.mail)" ||
  grep -Fq 'isForeground=true' <<<"$("$ADB" shell dumpsys activity services ai.onlinesdft.publisher.mail)"
grep -Fq 'android.app.Notification$CallStyle' <<<"$CALL_BLOCK"
grep -Fq 'android.app.Notification$MediaStyle' <<<"$MEDIA_BLOCK"
grep -Fq 'android.mediaSession=Token' <<<"$MEDIA_BLOCK"

printf '%s\n' 'ALL_NOTIFICATION_TYPES_PROOF_BEGIN'
grep -E "ROUTE_COMMITTED event_id=all-types-(ongoing|fgs|call|media|standard)-$RUN_SUFFIX" <<<"$PROOF_LOGS"
printf '%s\n' 'ALL_NOTIFICATION_TYPES_PROOF_END'
