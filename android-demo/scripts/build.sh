#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env.sh"

cd "$PROJECT_DIR"
./gradlew \
  :router:testDebugUnitTest \
  :router:assembleDebug \
  :publisher:assembleChatDebug \
  :publisher:assembleCalendarDebug \
  :publisher:assembleMailDebug

find router/build/outputs/apk publisher/build/outputs/apk -type f -name '*.apk' -print | sort
