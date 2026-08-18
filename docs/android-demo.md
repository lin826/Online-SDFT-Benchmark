# Android LoRA demo

The Android branch now trains a rank-4 LoRA adapter inside
`LiquidAI/LFM2.5-230M` with ONNX Runtime Training. The old 524×3 residual-head
experiment is not present in this branch.

See [`android-demo/README.md`](../android-demo/README.md) for:

- the exact 172,032-parameter LoRA shell;
- host export and mandatory optimizer proof;
- real-phone APK installation and model provisioning;
- notification-access setup;
- the silent, expandable Android shade group for reversible archives;
- four-callback warmup and genuine on-phone learning;
- checkpoint/log/audit verification across process restart;
- emulator and physical-device limitations.
