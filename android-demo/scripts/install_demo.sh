#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env.sh"

ADB="${ADB:-$ANDROID_SDK_ROOT/platform-tools/adb}"
"$ADB" wait-for-device
"$ADB" install -r "$PROJECT_DIR/router/build/outputs/apk/debug/router-debug.apk"
"$ADB" install -r "$PROJECT_DIR/publisher/build/outputs/apk/chat/debug/publisher-chat-debug.apk"
"$ADB" install -r "$PROJECT_DIR/publisher/build/outputs/apk/calendar/debug/publisher-calendar-debug.apk"
"$ADB" install -r "$PROJECT_DIR/publisher/build/outputs/apk/mail/debug/publisher-mail-debug.apk"

for package_name in \
  ai.onlinesdft.router.debug \
  ai.onlinesdft.publisher.chat \
  ai.onlinesdft.publisher.calendar \
  ai.onlinesdft.publisher.mail
do
  "$ADB" shell pm grant "$package_name" android.permission.POST_NOTIFICATIONS || true
done

"$ADB" shell cmd notification allow_listener \
  ai.onlinesdft.router.debug/ai.onlinesdft.router.notification.RouterNotificationListenerService

# The shared capture AVD may still have the earlier prototype listener enabled;
# leave its APK installed but prevent it from racing this demo for notifications.
"$ADB" shell cmd notification disallow_listener \
  com.onlinesdft.triage/com.onlinesdft.triage.service.NotificationRouterService || true
"$ADB" shell am start -n ai.onlinesdft.router.debug/ai.onlinesdft.router.MainActivity
