# Android LFM Online-SDFT with real LoRA training

This directory contains an installable Android app that performs real
forward/backward/AdamW updates to a rank-4 LoRA adapter inside
`LiquidAI/LFM2.5-230M`. It does **not** train a residual policy head.

The app uses `onnxruntime-training-android:1.19.2`. The immutable FP32 base
weights remain frozen. Exactly 48 LoRA tensors (172,032 parameters) are
trainable:

- rank `4`, alpha `8`, dropout `0`;
- `q_proj`, `k_proj`, `v_proj`, and `self_attn.out_proj`;
- LFM attention layers `2, 4, 6, 8, 10, 12`.

The LFM A/B/C next-token logits map to `INTERRUPT`, `LATER`, and `ARCHIVE`.
Student decisions use the live LoRA checkpoint. Base comparisons and the
factual-hindsight teacher use a separately exported zero-adapter graph.

Each accepted callback produces the same causal soft target used by the demo:

```text
reliable target = 0.05 × fixed teacher
                + 0.05 × sealed student decision
                + 0.90 × factual causal support
```

An ambiguous digest open projects the sealed decision onto
`{INTERRUPT, LATER}`. `UNKNOWN` creates no update. The learner retains 32
accepted rows, warms up for four, then performs one ORT AdamW step on the newest
accepted row. This single-row Android contract avoids non-finite activations
observed with long dynamically padded LFM batches in ORT 1.19.2 on ARM and
reduces peak phone memory. Both LoRA/optimizer state and replay/RNG metadata use
crash-recoverable two-slot checkpoints.

Model-selected `ARCHIVE` is reversible. After Android confirms source removal,
the app keeps an app-private copy plus the exact sealed decision under **Saved &
archived**. **Open it** launches the source app and teaches `LATER`, **Show
next time** explicitly teaches `INTERRUPT`, and **Keep these quiet** confirms
`ARCHIVE`. The record survives process recreation, while Reset and Android
lockdown purge its notification content.

Unread model-selected archives are also projected into Android's notification
shade on a separate low-importance, soundless channel. Android collapses them
under one **Tact** group; expand its count/arrow to inspect each
child. The router excludes its own package from scoring, so these projection
notifications cannot feed back into the routing loop. Tapping the group or a
child only opens the **Saved** tab and is not treated as feedback. The item
leaves the unread shade group only after the customer chooses **Open it**,
**Show next time**, or **Keep these quiet** and the corresponding callback finishes.
Shade content is redacted on the lock screen.

## What is generated off-device

The phone does not download model code or build gradient graphs. A Linux host
exports and verifies six immutable artifacts:

```text
base_model.onnx       zero-adapter base/teacher inference
training_model.onnx   forward + soft CE + backward graph
eval_model.onnx       live LoRA evaluation graph
optimizer_model.onnx  AdamW graph
checkpoint            initial model/LoRA checkpoint
tokenizer.json        pinned LFM tokenizer
```

The exporter refuses to publish a bundle until a host ORT session has completed
one forward/backward/AdamW step and observed a changed LoRA norm. The phone then
runs the same training artifacts through the Android training AAR.

## Build the APKs

The project requires JDK 17 and Android SDK 36. The checked-in helper expects
the workspace defaults from `scripts/env.sh`:

```bash
cd android-demo
./scripts/build.sh
```

Outputs:

```text
router/build/outputs/apk/debug/router-debug.apk
publisher/build/outputs/apk/chat/debug/publisher-chat-debug.apk
publisher/build/outputs/apk/calendar/debug/publisher-calendar-debug.apk
publisher/build/outputs/apk/mail/debug/publisher-mail-debug.apk
```

## Export the LoRA training bundle

Use Linux x86_64 with Python 3.11. `onnxruntime-training==1.19.2` is a host
artifact-generation dependency; the phone uses the matching Android AAR.

```bash
python3.11 -m venv /tmp/lfm-lora-export
/tmp/lfm-lora-export/bin/pip install \
  torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
/tmp/lfm-lora-export/bin/pip install \
  transformers==5.13.1 peft==0.19.1 \
  numpy==2.4.6 onnx==1.16.2 onnxruntime-training==1.19.2

/tmp/lfm-lora-export/bin/python scripts/export_lfm_lora_ort.py \
  --output-dir "$HOME/.cache/online-sdft/lfm-lora-ort-1.19.2"
```

Add `--local-files-only` when the pinned Hugging Face revision already exists
in the host cache. The export is CPU/RAM/disk intensive. Start with at least
16 GB host RAM and several GB of free disk. The generated bundle is deliberately
outside Git and outside the APK.

The final output includes a `host_optimizer_proof` in `manifest.json`. Confirm
that `adapter_norm_changed` is `true` before deployment:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path.home()/'.cache/online-sdft/lfm-lora-ort-1.19.2/manifest.json'
print(json.loads(p.read_text())['parity']['host_optimizer_proof'])
PY
```

## Deploy to a real Android phone

Recommended first device:

- Android 12/API 31 or newer;
- ARM64;
- 12 GB RAM minimum, 16 GB preferred;
- at least 8 GB free storage;
- developer options and USB debugging enabled;
- phone connected to power, screen on, and thermal/battery-saver restrictions
  disabled during the first proof run.

The full FP32 training graph is intentionally a correctness implementation, not
a claim of practical background training on low-memory phones. A device that
cannot allocate the graph fails closed and leaves notifications untouched.

Install all four debug APKs:

```bash
export PHONE_SERIAL='replace-with-adb-serial'
adb -s "$PHONE_SERIAL" install -r router/build/outputs/apk/debug/router-debug.apk
adb -s "$PHONE_SERIAL" install -r publisher/build/outputs/apk/chat/debug/publisher-chat-debug.apk
adb -s "$PHONE_SERIAL" install -r publisher/build/outputs/apk/calendar/debug/publisher-calendar-debug.apk
adb -s "$PHONE_SERIAL" install -r publisher/build/outputs/apk/mail/debug/publisher-mail-debug.apk
adb -s "$PHONE_SERIAL" shell pm grant \
  ai.onlinesdft.router.debug android.permission.POST_NOTIFICATIONS
```

Provision the verified bundle. The script hashes every host and device file,
copies through `run-as` so the real app process owns every artifact, and swaps
the app-private files directory only after all copies validate:

```bash
./scripts/provision_lfm_lora.sh \
  --serial "$PHONE_SERIAL" \
  --bundle-dir "$HOME/.cache/online-sdft/lfm-lora-ort-1.19.2"

adb -s "$PHONE_SERIAL" shell am force-stop ai.onlinesdft.router.debug
adb -s "$PHONE_SERIAL" shell monkey -p ai.onlinesdft.router.debug 1
```

On the phone, open:

```text
Settings → Apps → Special app access → Notification access
```

Enable **Tact**, then allow its notification permission. Android
calls a notification listener after posting, so the app demonstrates post-time
rerouting rather than guaranteed pre-alert suppression. Protected calls,
foreground services, media playback, group summaries, ongoing, and non-clearable
notifications are scored but passed through safely.

## Make the phone really learn

1. Open the router app and wait for the model panel to show
   `Ready · real A/B/C logits + local ORT Training`.
2. Post four genuine test notifications, for example:

   ```bash
   for n in 1 2 3 4; do
     ANDROID_SERIAL="$PHONE_SERIAL" ./scripts/post_case.sh \
       ai.onlinesdft.publisher.chat promo-lora-$n promo-lora-$n \
       promo 0.10 off-hours "Weekend offer $n" \
       "Browse this offer later" standard
   done
   ```

3. For each notification card in the router, tap the same explicit correction,
   such as **Archive**. The first three accepted callbacks are warmup lessons.
   The fourth invokes `trainStep`, `optimizerStep`, `lazyResetGrad`, and a
   checkpoint commit on the phone.
4. Post a fifth semantically similar notification. Its decision is evaluated
   from the new LoRA checkpoint.

The synthetic **Run 10 corrections + 10 probes** button can also produce
accepted lessons, but genuine publisher notifications are the stronger device
proof.

## Verify that Android changed model parameters

Watch the proof logs:

```bash
adb -s "$PHONE_SERIAL" logcat -c
adb -s "$PHONE_SERIAL" logcat -s OnlineSdftProof:I LiquidOrtLoRA:I '*:S'
```

Successful learning emits both warmup rows and an update row similar to:

```text
ON_DEVICE_WARMUP ... replay_size=3 remaining=1 ...
ON_DEVICE_UPDATE ... update=1 loss_before=... loss_after=... \
  checksum_before=<sha256> checksum_after=<different-sha256> \
  thread=sdft-learning
```

The update is accepted only if the serialized ORT checkpoint hash changed.
Inspect the app-private two-slot model state:

```bash
adb -s "$PHONE_SERIAL" shell run-as ai.onlinesdft.router.debug \
  ls -l files/model/lora-ort-v1

adb -s "$PHONE_SERIAL" exec-out run-as ai.onlinesdft.router.debug \
  cat files/telemetry/audit-v1.jsonl | grep training_update
```

Then force-stop and reopen the app. The update index, LoRA norm, and checkpoint
checksum should remain unchanged across restart; the fifth similar notification
should still use the adapted distribution. **Reset** deletes both committed ORT
slots and reloads the exported initial checkpoint.

## Demo the reversible archive group

1. Produce two notifications that the live model routes to `ARCHIVE`.
2. Swipe down to Android's **Silent** section and expand the numbered
   **Tact** group.
3. Tap one archived child. Confirm that the **Saved** tab opens and the LoRA
   update index has not changed merely from inspecting it.
4. Choose **Show next time**, **Open it**, or **Keep these quiet** according to
   the desired future behavior, then wait for the learning status to finish.
5. Open the shade again. The resolved child is gone and the group count is one
   lower. **Local scores** and the `ON_DEVICE_UPDATE` proof log show a new LoRA
   update once warmup is complete.

This separates recoverability from preference evidence: viewing an archived
notification is safe, while an explicit choice closes the learning loop.

## Integrate Online-SDFT into your own app

The learning loop lives in
`router/src/main/java/ai/onlinesdft/router/model/`. Six files carry it:
`OnlineSdftLearner` (the loop), `FrozenFoundationRuntime` and
`LiquidOrtFoundationModel` (ORT inference and training), `SdftTargetBuilder`
(the hindsight target), `LoraReplayStore` (checkpointed replay), and `Domain`
(the shared types). Copy that package, keep the ORT Training dependency from
`router/build.gradle.kts`, and provision a training bundle with the export
step above.

Your app then makes two calls. At decision time:

```kotlin
val decision = learner.decide(context)   // DecisionSnapshot: route + frozen distribution
```

Execute `decision.route` however your app delivers it, and keep the snapshot.
When the outcome arrives, minutes or hours later:

```kotlin
val feedback = FactualFeedback(
    eventId = context.eventId,
    executedRoute = decision.route,
    outcome = observedOutcome,           // what the surface actually reported
    observedSelection = selectionOrNull, // null when the surface cannot reveal one
    delayMinutes = elapsedMinutes,
    source = FeedbackSource.USER,
    observedAtMillis = System.currentTimeMillis(),
)
val metrics = learner.learn(decision, feedback)   // null when nothing was learnable
```

`learn` returns null for censored feedback and during warmup, which is the
expected path rather than an error. [`state/DemoRuntime.kt`](router/src/main/java/ai/onlinesdft/router/state/DemoRuntime.kt)
is the reference wiring: `decide` at line 204, `learn` at line 1233.

A few obligations come with it:

- **Run `learn` off the main thread.** It performs ORT AdamW steps. Schedule it
  with WorkManager under charging, idle, and battery-not-low constraints if you
  batch updates rather than applying them on arrival.
- **Persist through `LoraReplayStore`.** Point it at a file in app-private
  storage; it checkpoints adapter and replay state with a checksum so a crash
  mid-update cannot leave a torn adapter. `learner.status()` reports the update
  index and LoRA norm, and `learner.reset()` restores the shipped adapter.
- **Keep inspection separate from evidence.** Reading a routed item must not
  call `learn`. Only an explicit user choice closes the loop.

For a different task, three places change: the `Route` enum in `Domain.kt` for
your action set, `LfmCompactPromptCodec` for how context is rendered, and
`SdftTargetBuilder.support` for which outcomes identify one action and which
leave several open. The fusion weights and the censoring rule stay as they are.

## Emulator

The same install/provision flow works with the API 34 `OnlineSdft34` AVD:

```bash
$HOME/.local/android-sdk/emulator/emulator -avd OnlineSdft34 -no-snapshot-save
./scripts/install_demo.sh
./scripts/provision_lfm_lora.sh --bundle-dir \
  "$HOME/.cache/online-sdft/lfm-lora-ort-1.19.2"
```

An emulator proves integration and numerical behavior. Because it runs on host
RAM and CPU, ARM performance, phone memory pressure, battery, and thermals come
from a run on real hardware.

## Failure checks

- `The app was not packaged with onnxruntime-training-android`: rebuild the
  router APK from this branch; an inference-only AAR was installed.
- `Graph inputs/outputs do not match`: exporter and APK are from different
  commits; regenerate and reprovision the whole bundle.
- `Replay checkpoint does not match the committed LoRA checkpoint`: one side
  of a previous commit was interrupted. Reset, or allow the runtime to keep the
  LoRA checkpoint and discard incompatible replay metadata.
- Process death or OOM while training: the previous slot remains valid. The app
  never treats a partially written checkpoint as committed.
- No update after a callback: `UNKNOWN` and unsupported causal outcomes are
  deliberately censored; four accepted lessons are required for warmup.

Official background: [ONNX Runtime on-device training](https://onnxruntime.ai/docs/get-started/training-on-device.html)
and the [Android training tutorial](https://onnxruntime.ai/docs/tutorials/on-device-training/android-app.html).
