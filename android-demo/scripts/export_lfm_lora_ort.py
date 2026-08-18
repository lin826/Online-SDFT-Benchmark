#!/usr/bin/env python3
"""Export real LFM2.5-230M LoRA artifacts for ORT Training on Android.

The forward graph contains rank-4 LoRA in q/k/v/self-attention-out projections
of LFM attention layers 2/4/6/8/10/12. Only those 48 tensors require gradients
(172,032 parameters). The graph computes a three-action soft-target CE loss;
ORT generates backward, eval, AdamW, and checkpoint artifacts. A separate
zero-adapter graph supplies the immutable base/hindsight teacher.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence


MODEL_ID = "LiquidAI/LFM2.5-230M"
HF_REVISION = "13a53837c4906b4f7405932532ba85d182bb013b"
ORT_VERSION = "1.19.2"
OPSET_VERSION = 17
PRECISION = "fp32"
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "self_attn.out_proj")
LORA_LAYERS = (2, 4, 6, 8, 10, 12)
TRAINABLE_PARAMETERS = 172_032
TRAINABLE_TENSORS = 48
LEARNING_RATE = 1e-3
MAX_SEQUENCE_LENGTH = 256
MAX_BATCH_SIZE = 1
ACTION_CODES = ("A", "B", "C")
ACTION_TOKEN_IDS = (542, 543, 544)
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 7
TOKENIZER_SIZE = 64_402
MODEL_VOCAB_SIZE = 65_536
LOSS_OUTPUT = "loss"
LOGITS_OUTPUT = "action_logits"
PROBABILITIES_OUTPUT = "action_probabilities"
ADAPTER_NORM_OUTPUT = "adapter_l2_norm"
TARGETS_INPUT = "target_probabilities"

EXPECTED_VERSIONS = {
    "torch": "2.13.0+cpu",
    "transformers": "5.13.1",
    "peft": "0.19.1",
    "numpy": "2.4.6",
    "onnx": "1.16.2",
    "onnxruntime-training": ORT_VERSION,
}
SOURCE_HASHES = {
    "model.safetensors": "f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd",
    "config.json": "f7d0bcc454b7a30fa471b1e7b9e359e11fb25b56f5b4ffd59bb18248e3c2ea3d",
    "tokenizer.json": "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd",
}
REPO_ROOT = Path(__file__).resolve().parents[2]

STUDENT_SYSTEM_PROMPT = """You are an on-device notification router.
Assess the partial evidence, then choose exactly one route:
A = INTERRUPT now
B = LATER in a digest
C = ARCHIVE without a notification
Use the current notification and any past completed interactions. Do not add explanation."""


class ExportError(RuntimeError):
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LFM2.5-230M rank-4 LoRA ORT Training artifacts.",
        epilog=f"""Reference Linux x86_64 environment (the ORT training wheel is host-side):
  python3.11 -m venv /tmp/lfm-lora-export
  /tmp/lfm-lora-export/bin/pip install \\
    torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
  /tmp/lfm-lora-export/bin/pip install \\
    transformers==5.13.1 peft==0.19.1 \\
    numpy==2.4.6 onnx==1.16.2 onnxruntime-training=={ORT_VERSION}

The exporter performs a real host ORT forward/backward/AdamW proof before it
publishes the bundle. Budget at least 16 GB RAM and several GB of free disk.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise ExportError(f"missing package {name!r}; see --help") from error


def dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from huggingface_hub import snapshot_download
        from onnxruntime.training import artifacts
        from onnxruntime.training.api import CheckpointState, Module, Optimizer
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ExportError(f"missing package {error.name!r}; see --help") from error
    versions = {name: package_version(name) for name in EXPECTED_VERSIONS}
    mismatches = {
        name: {"expected": EXPECTED_VERSIONS[name], "actual": actual}
        for name, actual in versions.items()
        if actual != EXPECTED_VERSIONS[name]
    }
    if mismatches:
        raise ExportError(f"reference package versions are mandatory: {mismatches}")
    return locals()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def output_path(raw: Path) -> Path:
    result = raw.expanduser().resolve()
    try:
        result.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise ExportError("large generated bundles must be outside the repository")
    if result.exists():
        raise ExportError(f"output already exists: {result}")
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def snapshot(args: argparse.Namespace, deps: dict[str, Any]) -> Path:
    kwargs: dict[str, Any] = {
        "repo_id": MODEL_ID,
        "revision": HF_REVISION,
        "local_files_only": args.local_files_only,
    }
    if args.cache_dir:
        kwargs["cache_dir"] = str(args.cache_dir.expanduser())
    result = Path(deps["snapshot_download"](**kwargs)).resolve()
    for name, expected in SOURCE_HASHES.items():
        path = result / name
        if not path.is_file() or sha256(path) != expected:
            raise ExportError(f"pinned source validation failed: {name}")
    return result


def load_tokenizer(source: Path, deps: dict[str, Any]):
    tokenizer = deps["AutoTokenizer"].from_pretrained(str(source), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    action_ids = tuple(
        tokenizer.encode(code, add_special_tokens=False)[0] for code in ACTION_CODES
    )
    if action_ids != ACTION_TOKEN_IDS:
        raise ExportError(f"A/B/C token ids changed: {action_ids}")
    if (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) != (
        PAD_TOKEN_ID,
        BOS_TOKEN_ID,
        EOS_TOKEN_ID,
    ):
        raise ExportError("tokenizer special ids changed")
    if len(tokenizer) != TOKENIZER_SIZE:
        raise ExportError("tokenizer size changed")
    return tokenizer


def probe(tokenizer: Any) -> list[int]:
    context = (
        "The notification title is Payment failed. The message says update your "
        "payment method. This is a commerce notification that arrived at 09:15 "
        "local time during the weekday period. Its on-device importance score is 0.87 out of 1."
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Notification: {context}\nRoute:"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or ids[0] != BOS_TOKEN_ID or len(ids) > MAX_SEQUENCE_LENGTH:
        raise ExportError("validation prompt violates the Android token contract")
    return ids


def build_models(source: Path, deps: dict[str, Any]):
    torch = deps["torch"]
    torch.manual_seed(0)
    base = deps["AutoModelForCausalLM"].from_pretrained(
        str(source), local_files_only=True, dtype=torch.float32
    ).cpu()
    base.config.use_cache = False
    config = deps["LoraConfig"](
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        layers_to_transform=list(LORA_LAYERS),
        layers_pattern="layers",
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True,
        ensure_weight_tying=False,
    )
    peft_model = deps["get_peft_model"](base, config).cpu()
    peft_model.eval()
    causal_lm = peft_model.get_base_model()
    trainable = [(name, p) for name, p in peft_model.named_parameters() if p.requires_grad]
    if len(trainable) != TRAINABLE_TENSORS or sum(p.numel() for _, p in trainable) != TRAINABLE_PARAMETERS:
        raise ExportError(
            f"LoRA shell changed: tensors={len(trainable)}, parameters={sum(p.numel() for _, p in trainable)}"
        )
    action_rows = causal_lm.lm_head.weight.detach().index_select(
        0, torch.tensor(ACTION_TOKEN_IDS)
    ).float().clone()

    class LoraActionTraining(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = causal_lm.model
            self.register_buffer("action_lm_head_rows", action_rows)

        def forward(self, input_ids, attention_mask, target_probabilities):
            hidden = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1, :].float()
            logits = torch.nn.functional.linear(hidden, self.action_lm_head_rows)
            probabilities = torch.softmax(logits, dim=-1)
            loss = -(target_probabilities * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
            lora = [p.float().square().sum() for n, p in self.named_parameters() if "lora_" in n]
            adapter_norm = torch.sqrt(torch.stack(lora).sum())
            return loss, logits, probabilities, adapter_norm

    class BaseActionInference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = causal_lm.model
            self.register_buffer("action_lm_head_rows", action_rows)

        def forward(self, input_ids, attention_mask):
            hidden = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1, :].float()
            logits = torch.nn.functional.linear(hidden, self.action_lm_head_rows)
            return logits, torch.softmax(logits, dim=-1)

    return peft_model, LoraActionTraining().train(), BaseActionInference().eval()


def export_forward(stage: Path, wrapper: Any, ids: list[int], deps: dict[str, Any]) -> Path:
    torch = deps["torch"]
    path = stage / "forward_lora.onnx"
    # Keep the dynamic batch axis visible by tracing LFM's batch>1 mask branch.
    # The Android deploy contract caps optimizer steps at one row because ORT
    # 1.19.2 on ARM can yield non-finite activations for long padded batches.
    # Dynamic batch=1 is exercised again by prove_optimizer.
    input_ids = torch.tensor([ids, ids], dtype=torch.long)
    attention = torch.ones_like(input_ids)
    targets = torch.tensor(
        [[0.05, 0.90, 0.05], [0.90, 0.05, 0.05]],
        dtype=torch.float32,
    )
    torch.onnx.export(
        wrapper,
        (input_ids, attention, targets),
        str(path),
        input_names=["input_ids", "attention_mask", TARGETS_INPUT],
        output_names=[LOSS_OUTPUT, LOGITS_OUTPUT, PROBABILITIES_OUTPUT, ADAPTER_NORM_OUTPUT],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            TARGETS_INPUT: {0: "batch"},
            LOGITS_OUTPUT: {0: "batch"},
            PROBABILITIES_OUTPUT: {0: "batch"},
        },
        export_params=True,
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.TRAINING,
        opset_version=OPSET_VERSION,
        keep_initializers_as_inputs=False,
        external_data=False,
        dynamo=False,
    )
    deps["onnx"].checker.check_model(str(path))
    return path


def export_base(stage: Path, peft_model: Any, wrapper: Any, ids: list[int], deps: dict[str, Any]) -> Path:
    torch = deps["torch"]
    path = stage / "base_model.onnx"
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention = torch.ones_like(input_ids)
    with peft_model.disable_adapter():
        torch.onnx.export(
            wrapper,
            (input_ids, attention),
            str(path),
            input_names=["input_ids", "attention_mask"],
            output_names=[LOGITS_OUTPUT, PROBABILITIES_OUTPUT],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                LOGITS_OUTPUT: {0: "batch"},
                PROBABILITIES_OUTPUT: {0: "batch"},
            },
            export_params=True,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            opset_version=OPSET_VERSION,
            external_data=False,
            dynamo=False,
        )
    deps["onnx"].checker.check_model(str(path))
    return path


def make_training_artifacts(stage: Path, forward: Path, deps: dict[str, Any]) -> list[str]:
    model = deps["onnx"].load(str(forward), load_external_data=False)
    initializer_names = [item.name for item in model.graph.initializer]
    trainable = [name for name in initializer_names if ".lora_A." in name or ".lora_B." in name]
    frozen = [name for name in initializer_names if name not in trainable]
    if len(trainable) != TRAINABLE_TENSORS:
        raise ExportError(f"exported LoRA initializer count changed: {len(trainable)}")
    # ORT Training 1.19.2 is the newest Android training AAR. Its artifact
    # generator does not expose its optimization phase, and its ShapeOptimizer
    # crashes on this valid dynamic LFM graph. Bypass only that optional
    # pre-gradient rewrite and let GradientGraphBuilder consume the checked
    # forward graph directly. The generated graph is checked and executed below.
    from onnxruntime.training.onnxblock import _training_graph_utils

    original_optimizer = _training_graph_utils.get_optimized_model
    _training_graph_utils.get_optimized_model = lambda serialized, _grad, _options: serialized
    try:
        # Reuse the already-loaded proto. Passing the path makes ORT load a
        # second ~877 MB copy while it materializes the backward graph.
        deps["artifacts"].generate_artifacts(
            model,
            requires_grad=trainable,
            frozen_params=frozen,
            loss=None,
            loss_input_names=[LOSS_OUTPUT],
            optimizer=deps["artifacts"].OptimType.AdamW,
            additional_output_names=[LOGITS_OUTPUT, PROBABILITIES_OUTPUT, ADAPTER_NORM_OUTPUT],
            artifact_directory=stage,
        )
    finally:
        _training_graph_utils.get_optimized_model = original_optimizer
    required = ["training_model.onnx", "eval_model.onnx", "optimizer_model.onnx", "checkpoint"]
    for name in required:
        if not (stage / name).is_file():
            raise ExportError(f"ORT did not generate {name}")
    return trainable


def prove_optimizer(stage: Path, ids: list[int], deps: dict[str, Any]) -> dict[str, Any]:
    np = deps["np"]
    state = deps["CheckpointState"].load_checkpoint(str(stage / "checkpoint"))
    module = deps["Module"](
        str(stage / "training_model.onnx"),
        state,
        str(stage / "eval_model.onnx"),
        device="cpu",
    )
    optimizer = deps["Optimizer"](str(stage / "optimizer_model.onnx"), module)
    optimizer.set_learning_rate(LEARNING_RATE)
    input_ids = np.asarray([ids], dtype=np.int64)
    mask = np.ones_like(input_ids)
    target = np.asarray([[0.05, 0.90, 0.05]], dtype=np.float32)
    module.eval()
    before = module(input_ids, mask, target)
    loss_before = float(np.asarray(before[0]))
    norm_before = float(np.asarray(before[3]))
    module.train()
    trained = module(input_ids, mask, target)
    optimizer.step()
    module.lazy_reset_grad()
    module.eval()
    after = module(input_ids, mask, target)
    loss_after = float(np.asarray(after[0]))
    norm_after = float(np.asarray(after[3]))
    if not all(np.isfinite(value) for value in (loss_before, loss_after, norm_before, norm_after)):
        raise ExportError("non-finite ORT training proof")
    if norm_after == norm_before:
        raise ExportError("ORT AdamW proof did not change the LoRA norm")
    # Restore the distributable checkpoint to exported initialization. The
    # proof mutates only this in-memory CheckpointState.
    return {
        "loss_before": loss_before,
        "loss_after": loss_after,
        "train_step_loss": float(np.asarray(trained[0])),
        "adapter_norm_before": norm_before,
        "adapter_norm_after": norm_after,
        "adapter_norm_changed": True,
    }


def artifact(path: Path, root: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.relative_to(root).as_posix(),
        "kind": "file",
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "deploy": True,
    }


def write_manifest(
    stage: Path,
    source: Path,
    ids: list[int],
    trainable: list[str],
    proof: dict[str, Any],
    deps: dict[str, Any],
) -> None:
    roles = {
        "base_model.onnx": "zero_adapter_base_model",
        "training_model.onnx": "lora_training_model",
        "eval_model.onnx": "lora_eval_model",
        "optimizer_model.onnx": "adamw_optimizer_model",
        "checkpoint": "initial_lora_checkpoint",
        "tokenizer.json": "tokenizer",
    }
    records = [artifact(stage / name, stage, role) for name, role in roles.items()]
    manifest = {
        "schema": "ai.onlinesdft.lfm_lora_ort_bundle",
        "schema_version": 1,
        "model": {
            "id": MODEL_ID,
            "revision": HF_REVISION,
            "precision": PRECISION,
            "source_model_sha256": SOURCE_HASHES["model.safetensors"],
            "trainable_parameters": TRAINABLE_PARAMETERS,
            "trainable_tensors": TRAINABLE_TENSORS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW",
            "lora": {
                "rank": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_modules": list(LORA_TARGET_MODULES),
                "layers": list(LORA_LAYERS),
                "initializer_names": trainable,
            },
        },
        "tokenizer": {
            "tokenizer_json_sha256": SOURCE_HASHES["tokenizer.json"],
            "tokenizer_size": TOKENIZER_SIZE,
            "model_vocab_size": MODEL_VOCAB_SIZE,
            "pad_token_id": PAD_TOKEN_ID,
            "bos_token_id": BOS_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
        },
        "graph_contract": {
            "opset": OPSET_VERSION,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "max_batch_size": MAX_BATCH_SIZE,
            "action_codes": list(ACTION_CODES),
            "action_token_ids": list(ACTION_TOKEN_IDS),
            "inputs": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "target_probabilities": TARGETS_INPUT,
            },
            "outputs": {
                "loss": LOSS_OUTPUT,
                "action_logits": LOGITS_OUTPUT,
                "action_probabilities": PROBABILITIES_OUTPUT,
                "adapter_l2_norm": ADAPTER_NORM_OUTPUT,
            },
            "soft_target_cross_entropy": True,
            "full_vocabulary_output": False,
            "training_graph": True,
        },
        "runtime": {
            "onnxruntime_version": ORT_VERSION,
            "android_maven_artifact": "com.microsoft.onnxruntime:onnxruntime-training-android:1.19.2",
            "execution_provider": "CPU",
        },
        "parity": {"probe_token_ids": ids, "host_optimizer_proof": proof},
        "deploy_contract": {
            "android_application_id": "ai.onlinesdft.router.debug",
            "private_files_subdirectory": "lfm_lora",
            "tokenizer_path": "tokenizer.json",
            "base_model_path": "base_model.onnx",
            "training_model_path": "training_model.onnx",
            "eval_model_path": "eval_model.onnx",
            "optimizer_model_path": "optimizer_model.onnx",
            "initial_checkpoint_path": "checkpoint",
        },
        "artifacts": records,
        "exporter": {
            "packages": {name: package_version(name) for name in EXPECTED_VERSIONS},
            "script_sha256": sha256(Path(__file__).resolve()),
        },
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> Path:
    destination = output_path(args.output_dir)
    deps = dependencies()
    source = snapshot(args, deps)
    tokenizer = load_tokenizer(source, deps)
    ids = probe(tokenizer)
    peft_model, training_wrapper, base_wrapper = build_models(source, deps)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        forward = export_forward(stage, training_wrapper, ids, deps)
        del training_wrapper
        gc.collect()
        export_base(stage, peft_model, base_wrapper, ids, deps)
        del base_wrapper, peft_model
        gc.collect()
        trainable = make_training_artifacts(stage, forward, deps)
        proof = prove_optimizer(stage, ids, deps)
        shutil.copyfile(source / "tokenizer.json", stage / "tokenizer.json")
        if sha256(stage / "tokenizer.json") != SOURCE_HASHES["tokenizer.json"]:
            raise ExportError("copied tokenizer changed")
        forward.unlink()
        write_manifest(stage, source, ids, trainable, proof, deps)
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"bundle={destination}")
    print(f"manifest_sha256={sha256(destination / 'manifest.json')}")
    print(json.dumps(proof, sort_keys=True))
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (ExportError, AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
