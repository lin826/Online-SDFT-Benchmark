"""Static gates for the standalone Colab workflow."""

import ast
import base64
import hashlib
import json
import math
import zlib
from dataclasses import fields
from pathlib import Path

import nbformat
import numpy as np
import pytest

from online_sdft.config import (
    ACTIONS,
    DATASET_VERSION,
    ICL_K,
    MODEL_ID,
    PREFERENCE_SAMPLING_TEMPERATURE,
    PROMPT_STYLE,
    RAG_K,
    RAG_TEXT_WEIGHT,
    REINFORCE_BATCH_SIZE,
    REINFORCE_BASELINE_STEP,
    REINFORCE_ENTROPY_COEF,
    REINFORCE_LR,
    REINFORCE_MAX_GRAD_NORM,
    REINFORCE_TRAINING_OUTCOME_REWARDS,
    REGIMES,
    RFT_CANDIDATE_COUNT,
    RFT_PROTOCOL_VERSION,
    RFT_SAMPLING_MODE,
    RFT_SAMPLING_TEMPERATURE,
    RFT_SETTINGS_PROVENANCE,
    SDFT_UPDATE_STEPS,
    TEACHER_PROMPT_VERSION,
    TEACHER_REASONING_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    TEACHER_TEMPERATURE,
)
from online_sdft.environment import (
    DEFAULT_ENVIRONMENT,
    Event,
    NotificationRoutingEnvironment,
)
from online_sdft.experiment import RFT_CANDIDATE_SAMPLER, experiment_config
from online_sdft.methods import (
    DEFAULT_REINFORCE_SETTINGS,
    DEFAULT_RFT_SETTINGS,
    DEFAULT_SDFT_SETTINGS,
)
from scripts.build_standalone_notebook import (
    CANONICAL_MODEL_REVISION,
    GAME_ENGINE,
    REFERENCE_ARTIFACT_NAMES,
    RUNTIME_GUARD,
    RUN_CONFIGURATION,
    build_notebook,
    canonical_reproduction_manifest,
    embedded_core,
    markdown_cells,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "online_sdft_bandit_demo.ipynb"


def _code_sources() -> list[str]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]


def _embedded_pipeline_source() -> str:
    return next(
        source
        for source in _code_sources()
        if "class OnlineSDFTSettings" in source
    )


def _embedded_literals() -> dict[str, object]:
    values = {}
    for node in ast.parse(_embedded_pipeline_source()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            pass
    return values


def test_embedded_pipeline_exactly_matches_live_production_modules():
    _, separator, embedded = _embedded_pipeline_source().partition("\n")
    assert separator
    assert embedded == RUNTIME_GUARD + "\n\n" + embedded_core()


def test_committed_notebook_content_is_reproducible_from_builder():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    generated = markdown_cells()
    assert [
        (cell.cell_type, cell.source) for cell in notebook.cells
    ] == [
        (cell.cell_type, cell.source) for cell in generated
    ]


def test_notebook_embeds_exact_tracked_compact_artifact_bytes():
    manifest = canonical_reproduction_manifest()
    summary = json.loads(
        (ROOT / "outputs" / "bandit" / "summary.json").read_text()
    )
    assert tuple(manifest["seeds"]) == (0, 1, 2)
    assert manifest["model_revision"] == CANONICAL_MODEL_REVISION
    assert set(manifest["artifact_sha256"]) == set(REFERENCE_ARTIFACT_NAMES)
    assert manifest["source_fingerprint"] == summary["config"][
        "source_fingerprint"
    ]
    assert manifest["dataset_fingerprint"] == summary["config"][
        "dataset_fingerprint"
    ]
    for name in REFERENCE_ARTIFACT_NAMES:
        payload = (ROOT / "outputs" / "bandit" / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == (
            manifest["artifact_sha256"][name]
        )


def test_notebook_embeds_all_canonical_events_with_exact_float_bytes():
    manifest = canonical_reproduction_manifest()
    bundle = manifest["stream_bundle"]
    payload = zlib.decompress(base64.b64decode(bundle["zlib_base64"]))
    assert hashlib.sha256(payload).hexdigest() == bundle["sha256"]

    document = json.loads(payload)
    assert document["format"] == bundle["format"] == (
        "online-sdft-event-stream-v1-float-hex"
    )
    assert [entry["seed"] for entry in document["seeds"]] == [0, 1, 2]
    assert bundle["event_count"] == 720
    assert sum(len(entry["events"]) for entry in document["seeds"]) == 720

    event_fields = {field.name for field in fields(Event)}
    records = [
        record
        for entry in document["seeds"]
        for record in entry["events"]
    ]
    assert records
    assert all(set(record) == event_fields for record in records)

    streams = {}
    for entry in document["seeds"]:
        streams[entry["seed"]] = [
            Event(
                event_id=record["event_id"],
                phase=record["phase"],
                category=record["category"],
                scenario_id=record["scenario_id"],
                scenario_tier=record["scenario_tier"],
                title=record["title"],
                body=record["body"],
                hour=float.fromhex(record["hour"]),
                useful_horizon_minutes=record["useful_horizon_minutes"],
                importance=float.fromhex(record["importance"]),
                deadline=float.fromhex(record["deadline"]),
                affinity=float.fromhex(record["affinity"]),
                busy=float.fromhex(record["busy"]),
                x=np.asarray(
                    [float.fromhex(value) for value in record["x"]],
                    dtype=float,
                ),
                z={
                    key: float.fromhex(value)
                    for key, value in record["z"].items()
                },
                sampled_preference=record["sampled_preference"],
            )
            for record in entry["events"]
        ]

    class BundledEnvironment(NotificationRoutingEnvironment):
        def make_stream(self, seed):
            return streams[seed]

    assert DEFAULT_ENVIRONMENT.stream_fingerprint((0, 1, 2)) == (
        manifest["dataset_fingerprint"]
    )
    assert BundledEnvironment().stream_fingerprint((0, 1, 2)) == (
        manifest["dataset_fingerprint"]
    )


def test_notebook_builder_assigns_deterministic_cell_ids():
    first = build_notebook()
    second = build_notebook()
    assert [cell.id for cell in first.cells] == [cell.id for cell in second.cells]
    assert len({cell.id for cell in first.cells}) == len(first.cells)


def test_notebook_does_not_embed_or_advertise_tuning_assets():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    rendered = "\n".join(cell.source for cell in notebook.cells).lower()
    for stale_marker in (
        "outputs/tuning",
        "scripts/tune_",
        "sdft_visible_rubric",
        "visible-rubric",
        "adapter_warm_start",
        "bundled_lora_warm_start",
        "hyperparameter_selection",
        "no-rft-sweep",
        "pilot sweep",
        "one sweep arm",
        "prompt sweep",
        "selection artifact",
        "partial-dev",
        "unconfirmed",
    ):
        assert stale_marker not in rendered


def test_committed_notebook_code_cells_are_clean_and_unexecuted():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)


def test_interactive_game_examples_have_semantically_defensible_routes():
    namespace = {}
    exec(GAME_ENGINE, namespace)
    actions = namespace["GAME_ACTIONS"]
    routes = []
    for scenario in namespace["GAME_SCENARIOS"]:
        utilities = namespace["_game_utilities"](scenario)
        routes.append(actions[max(range(len(actions)), key=utilities.__getitem__)])
    assert routes == ["INTERRUPT", "INTERRUPT", "LATER", "ARCHIVE"]


def test_interactive_game_examples_are_exact_semantic_dataset_rows():
    namespace = {}
    exec(GAME_ENGINE, namespace)
    scenarios = namespace["GAME_SCENARIOS"]
    environment = NotificationRoutingEnvironment()
    events = {
        event.event_id: event
        for event in environment.make_stream(0)
    }
    for scenario in scenarios:
        event = events[scenario["event_id"]]
        total_minutes = int(math.floor(event.hour * 60)) % (24 * 60)
        hour, minute = divmod(total_minutes, 60)
        expected = {
            "dataset_version": DATASET_VERSION,
            "title": event.title,
            "body": event.body,
            "category": event.category,
            "time": f"{hour:02d}:{minute:02d}",
            "regime": REGIMES[event.phase],
            "importance": event.importance,
            "deadline": event.deadline,
            "affinity": event.affinity,
            "busy": event.busy,
            "incident": event.z["incident_on_call"],
            "manager": event.z["manager_focus"],
            "social": event.z["leisure_social"],
            "quiet_work": event.z["off_hours_quiet"],
            "useful_horizon_minutes": event.useful_horizon_minutes,
            "sampled_preference": ACTIONS[environment.gold_action(event)],
        }
        assert {
            key: scenario[key] for key in expected
        } == expected
        assert namespace["_game_utilities"](scenario) == tuple(
            environment.oracle_utilities(event)
        )

    section = next(
        cell.source
        for cell in markdown_cells()
        if cell.source.startswith("## 1. Play the router")
    )
    for scenario in scenarios:
        assert scenario["title"] in section
        assert scenario["time"] in section
    assert "exact rows from seed 0" in section


def test_notebook_builder_places_colab_badge_below_title():
    cells = markdown_cells()
    intro = cells[0].source
    assert intro.startswith(
        "# On-device Online-SDFT: learn from the route you actually took\n\n"
    )
    assert "colab.research.google.com" in intro
    assert "online_sdft_bandit_demo.ipynb" in intro
    assert 'alt="Open In Colab"' in intro
    assert sum("Open In Colab" in cell.source for cell in cells) == 1


def test_embedded_memory_configuration_and_teacher_prompt_match_live_contract():
    values = _embedded_literals()
    assert values["PROMPT_STYLE"] == PROMPT_STYLE == "causal_demos"
    assert values["DATASET_VERSION"] == DATASET_VERSION
    assert (
        values["PREFERENCE_SAMPLING_TEMPERATURE"]
        == PREFERENCE_SAMPLING_TEMPERATURE
        == 0.01
    )
    assert values["ICL_K"] == ICL_K == 3
    assert values["RAG_K"] == RAG_K == 3
    assert values["RAG_TEXT_WEIGHT"] == RAG_TEXT_WEIGHT == 0.5
    assert values["REINFORCE_LR"] == REINFORCE_LR == 1e-4
    assert values["REINFORCE_BATCH_SIZE"] == REINFORCE_BATCH_SIZE == 8
    assert (
        values["REINFORCE_BASELINE_STEP"]
        == REINFORCE_BASELINE_STEP
        == 0.0
    )
    assert (
        values["REINFORCE_ENTROPY_COEF"]
        == REINFORCE_ENTROPY_COEF
        == 1.0
    )
    assert (
        values["REINFORCE_MAX_GRAD_NORM"]
        == REINFORCE_MAX_GRAD_NORM
        == 1.0
    )
    assert (
        values["REINFORCE_TRAINING_OUTCOME_REWARDS"]
        == REINFORCE_TRAINING_OUTCOME_REWARDS
    )
    assert DEFAULT_REINFORCE_SETTINGS.to_dict() == {
        "learning_rate": 1e-4,
        "batch_size": 8,
        "baseline_step": 0.0,
        "entropy_coef": 1.0,
        "max_grad_norm": 1.0,
        "reward_outcome_map": REINFORCE_TRAINING_OUTCOME_REWARDS,
    }
    assert values["SDFT_UPDATE_STEPS"] == SDFT_UPDATE_STEPS == 2
    assert values["TEACHER_TEMPERATURE"] == TEACHER_TEMPERATURE == 1.0
    assert values["TEACHER_PROMPT_VERSION"] == TEACHER_PROMPT_VERSION
    assert values["TEACHER_SYSTEM_PROMPT"] == TEACHER_SYSTEM_PROMPT
    assert (
        values["TEACHER_REASONING_SYSTEM_PROMPT"]
        == TEACHER_REASONING_SYSTEM_PROMPT
    )
    source = _embedded_pipeline_source()
    settings = DEFAULT_SDFT_SETTINGS
    rendered_defaults = [
        f"reasoning_tokens: int = {settings.reasoning_tokens}",
        f'target_mode: str = "{settings.target_mode}"',
    ]
    for rendered_default in rendered_defaults:
        assert rendered_default in source
    for name in (
        "reliable_teacher_weight",
        "reliable_decision_weight",
        "reliable_behavior_weight",
        "ambiguous_teacher_weight",
        "ambiguous_decision_weight",
        "ambiguous_behavior_weight",
    ):
        prefix = f"{name}: float = "
        literal = next(
            line.split("=", 1)[1].strip()
            for line in source.splitlines()
            if line.strip().startswith(prefix)
        )
        assert math.isclose(float(literal), getattr(settings, name))
    assert "use_cache=True" in source
    assert "def _assessment_crosses_boundary" in source
    assert 'raise ValueError("teacher generated an invalid token id")' in source
    assert "except (OverflowError, RuntimeError, ValueError)" in source
    for forbidden in ("copy exactly", "correct output", "do not add explanation"):
        assert forbidden not in TEACHER_SYSTEM_PROMPT.lower()
        assert forbidden not in TEACHER_REASONING_SYSTEM_PROMPT.lower()


def test_embedded_rft_defaults_and_event_keyed_sampler_match_live_contract():
    values = _embedded_literals()
    assert values["RFT_PROTOCOL_VERSION"] == RFT_PROTOCOL_VERSION
    assert values["RFT_SETTINGS_PROVENANCE"] == RFT_SETTINGS_PROVENANCE
    assert values["RFT_CANDIDATE_COUNT"] == RFT_CANDIDATE_COUNT == 1
    assert (
        values["RFT_SAMPLING_TEMPERATURE"]
        == RFT_SAMPLING_TEMPERATURE
        == DEFAULT_RFT_SETTINGS.sampling_temperature
        == 8.0
    )
    assert (
        values["RFT_SAMPLING_MODE"]
        == RFT_SAMPLING_MODE
        == DEFAULT_RFT_SETTINGS.sampling_mode
        == "categorical"
    )

    source = _embedded_pipeline_source()
    assert f'RFT_CANDIDATE_SAMPLER = "{RFT_CANDIDATE_SAMPLER}"' in source
    assert "def rft_event_uniform(" in source
    assert "def rft_inverse_cdf_sample(" in source
    assert "rft_candidate_uniform" in source
    assert "teacher-categorical-k1-singleton-v1" not in source
    assert "capacity-matched-to-online-sdft-no-rft-sweep" not in source


def test_colab_setup_installs_the_required_frozen_lfm_and_peft_runtime():
    setup = next(
        source
        for source in _code_sources()
        if "packages = [" in source and "Runtime ready" in source
    )
    assert '"transformers==5.13.1"' in setup
    assert '"peft==0.19.1"' in setup
    assert '"numpy==2.4.6"' in setup
    assert "numeric_targets = {} if in_colab else" in setup
    assert 'if importlib.util.find_spec(module_name) is None' in setup
    assert 'packages.append(requirement)' in setup
    assert "else:\n    packages.extend(" in setup
    assert "Restart session" not in setup
    assert '"torch==2.13.0"' in setup
    assert 'if in_colab and importlib.util.find_spec("torch") is None' in setup
    assert "from transformers import AutoTokenizer, Lfm2Config" in setup
    assert "Lfm2ForCausalLM" in setup
    assert "import peft" in setup
    assert 'f"peft={peft.__version__} | "' in setup
    assert 'package_metadata.version("torchao")' in setup
    assert '"pip", "uninstall", "-y", "torchao"' in setup
    assert setup.index('"pip", "uninstall", "-y", "torchao"') < setup.index(
        "import peft"
    )
    assert "torchao>=" not in setup.lower()
    assert "from peft import LoraConfig, get_peft_model" in setup
    assert "smoke_base = Lfm2ForCausalLM(" in setup
    assert "full_attn_idxs=[0]" in setup
    assert 'target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]' in setup
    assert "smoke_lora = get_peft_model(" in setup
    assert '"lora_A" in name and parameter.requires_grad' in setup
    assert "smoke_lora(input_ids=torch.zeros((1, 2), dtype=torch.long))" in setup


def test_colab_setup_and_runner_enforce_a_completed_runtime():
    setup = next(
        source
        for source in _code_sources()
        if "_ONLINE_SDFT_RUNTIME_READY = False" in source
    )
    runner = next(
        source
        for source in _code_sources()
        if "run_canonical_lora_benchmark" in source
    )
    embedded = _embedded_pipeline_source()
    assert "Disconnect and delete runtime" in setup
    assert "Restart session" not in setup
    assert setup.index("_ONLINE_SDFT_RUNTIME_READY = False") < setup.index(
        "subprocess.check_call("
    )
    assert setup.index("_ONLINE_SDFT_RUNTIME_READY = True") > setup.index(
        "from transformers import AutoTokenizer, Lfm2Config"
    )
    assert setup.index("_ONLINE_SDFT_RUNTIME_READY = True") > setup.index(
        "smoke_lora = get_peft_model("
    )
    assert runner.index("_ONLINE_SDFT_RUNTIME_READY") < runner.index(
        "LiquidLLMPolicy("
    )
    assert embedded.index("_ONLINE_SDFT_RUNTIME_READY") < embedded.index(
        "import numpy as np"
    )
    assert RUNTIME_GUARD in embedded
    assert RUNTIME_GUARD in runner

    runner_guard = ast.parse(runner).body[0]
    guard_module = ast.Module(body=[runner_guard], type_ignores=[])
    with pytest.raises(RuntimeError, match="Section 5.1 did not finish"):
        exec(compile(guard_module, "<runner-guard>", "exec"), {})


def test_colab_setup_supports_cpu_fallback_and_reports_the_selected_device():
    setup = next(
        source
        for source in _code_sources()
        if "Runtime ready" in source
    )
    assert 'os.environ.get("COLAB_RELEASE_TAG")' in setup
    assert "CPU-only runtime detected" in setup
    assert "fully supported and uses FP32" in setup
    assert "loaded_conflicts" in setup
    assert "Restart session" not in setup
    assert "Restart the Jupyter kernel" in setup
    assert "Disconnect and delete runtime" in setup
    assert "Dependency smoke test failed" in setup
    assert "Lfm2ForCausalLM" in setup
    assert "LoRA adapter smoke test=passed" in setup
    assert "torch.cuda.get_device_name(0)" in setup
    assert 'device_name = "CPU (FP32)"' in setup


def test_notebook_strict_runner_gates_on_canonical_runtime_and_fingerprint():
    runner = next(
        source
        for source in _code_sources()
        if "def run_canonical_lora_benchmark" in source
    )
    assert CANONICAL_MODEL_REVISION in runner
    assert 'snapshot_download(' in runner
    assert 'revision=CANONICAL_REPRODUCTION["model_revision"]' in runner
    assert "except LocalEntryNotFoundError" in runner
    assert "local_files_only=True" in runner
    assert 'CANONICAL_REPRODUCTION["dataset_fingerprint"]' in runner
    strict_fingerprint_check = (
        'if strict:\n                    assert runtime_dataset_fingerprint == ('
    )
    assert strict_fingerprint_check in runner
    assert runner.index(strict_fingerprint_check) < runner.index(
        'config["dataset_fingerprint"] = canonical_dataset_fingerprint'
    )
    assert 'config["method_dataset_fingerprints"] = {' in runner
    assert 'config["source_fingerprint"]' in runner
    assert 'STRICT_BYTE_REPRODUCTION=False' in runner
    assert 'selected_device = "mps" if strict else "auto"' in runner
    assert (
        'STRICT_BYTE_REPRODUCTION = not bool(globals().get("in_colab", False))'
        in RUN_CONFIGURATION
    )
    assert "Strict byte reproduction requires the audited MPS/FP32 runtime" in runner
    assert "not a bit-for-bit reproduction" not in NOTEBOOK.read_text()


def test_notebook_advertises_the_full_rank_four_lora_adapter():
    settings = DEFAULT_SDFT_SETTINGS
    assert (
        settings.lora_rank,
        settings.lora_alpha,
        settings.lora_dropout,
    ) == (4, 8, 0.0)
    assert settings.lora_target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "self_attn.out_proj",
    )
    assert settings.lora_layers_to_transform == (2, 4, 6, 8, 10, 12)

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    rendered = "\n".join(cell.source for cell in notebook.cells)
    assert (
        "A rank-4 Q/K/V/O LoRA adapter with 172,032 trainable parameters"
        in rendered
    )
    assert 'config["online_sdft_trainable_parameters"] == 172032' in rendered
    assert 'config["online_rft_trainable_parameters"] == 172032' in rendered
    assert 'config["reinforce_trainable_parameters"] == 172032' in rendered
    assert "six paired methods, common rank-4 LoRA" in rendered
    assert "common zero-initialized rank-4 Q/K/V/O LoRA state" in rendered
    assert "replay 64 with a 32-step recency half-life" in rendered
    assert "2% epsilon-greedy baseline" in rendered
    assert "15% `INTERRUPT` probe with an 80-step half-life" in rendered
    assert "after step 160" in rendered
    assert "five-step half-life" in rendered
    assert "semantic-title-body-sharp-t001" in rendered
    assert "$q_i\\propto p_i^{100}$" in rendered
    assert "$T=0.01$" in rendered
    assert (
        "986cdf1a7d5fcc04c2b33f1bf90a1fc4f24a97ee85e663370382d8a67e4c932d"
        in rendered
    )
    assert "sdft_visible_rubric" not in rendered
    assert "adapter_warm_start" not in rendered
    assert "bundled_lora_warm_start" not in rendered
    assert (
        '"same_adapter_initialization": same_rft_architecture'
        in rendered
    )
    assert "peft==0.19.1" in rendered
    assert "get_peft_model" in rendered
    assert "init_lora_weights=True" in rendered

    assert "Lfm2ForCausalLM" in rendered


def test_notebook_runner_executes_all_paired_methods_and_releases_device_cache():
    runner = next(
        source
        for source in _code_sources()
        if "run_canonical_lora_benchmark" in source
    )
    assert 'run_method(' in runner
    assert "for seed_index, seed in enumerate(seeds, start=1)" in runner
    assert "for method in METHODS" in runner
    assert 'seeds = tuple(CANONICAL_REPRODUCTION["seeds"])' in runner
    assert "canonical_streams = _canonical_streams()" in runner
    assert "stream = canonical_streams[seed]" in runner
    assert "stream = DEFAULT_ENVIRONMENT.make_stream(seed)" not in runner
    assert 'bundle = CANONICAL_REPRODUCTION["stream_bundle"]' in runner
    assert 'observed_sha256 = hashlib.sha256(payload).hexdigest()' in runner
    assert "float.fromhex(record[\"hour\"])" in runner
    assert "Event(" in runner
    assert 'len(metrics) == len(seeds) * len(METHODS)' in runner
    assert 'len(rollouts) == len(seeds) * len(METHODS) * STREAM_LENGTH' in runner
    assert "LiquidLLMPolicy(" in runner
    assert "local_files_only=True" in runner
    assert "torch.mps.synchronize()" in runner
    assert "torch.mps.empty_cache()" in runner
    assert "torch.cuda.empty_cache()" in runner
    assert "SEED = 0" not in runner
    assert "_compact_artifact_bytes" in runner
    assert "write_compact_results" in runner


def test_colab_results_cell_is_self_checking():
    results = next(
        source
        for source in _code_sources()
        if "Byte-identical reproduction passed" in source
    )
    assert 'config["teacher_model"] == MODEL_ID' in results
    assert (
        'config["student_backbone"] == "frozen Liquid LFM base weights"'
        in results
    )
    assert 'config["student_backbone_trainable_parameters"] == 0' in results
    assert 'config["online_sdft_trainable_parameters"] == 172032' in results
    assert 'config["online_rft_trainable_parameters"] == 172032' in results
    assert 'config["reinforce_trainable_parameters"] == 172032' in results
    assert 'config["methods"] == METHODS' in results
    assert (
        'config["teacher_student_model_sharing"]["student_forward"] '
        '== "LoRA adapter enabled"'
        in results
    )
    assert (
        'config["teacher_student_model_sharing"]["teacher_forward"] '
        '== "same model with LoRA adapter disabled"'
        in results
    )
    assert "delayed observed user selection" in results
    assert "digest open ambiguous between INTERRUPT and LATER" in results
    assert "UNKNOWN stays censored" in results
    assert "no end-of-horizon flush" in results
    assert 'len(metrics) == 3 * len(METHODS)' in results
    assert 'len(rollouts) == 3 * len(METHODS) * STREAM_LENGTH' in results
    assert 'row["lesson_status"] == "soft_target_applied"' in results
    assert "actual == expected" in results
    assert "expected_sha == manifest_sha" in results
    assert 'all(item["exact"] for item in artifact_comparison.values())' in results
    assert "first differing byte" in results
    assert "beats" not in results

    config = experiment_config(
        3,
        0,
        MODEL_ID,
        type(
            "Policy",
            (),
            {"device": "cpu", "trainable_parameters": 172_032},
        )(),
    )
    config_assertions = []
    for node in ast.parse(results).body:
        if not isinstance(node, ast.Assert):
            continue
        names = {
            child.id
            for child in ast.walk(node.test)
            if isinstance(child, ast.Name)
        }
        if names <= {"config", "MODEL_ID", "METHODS"}:
            config_assertions.append(node)
    assert len(config_assertions) >= 15
    exec(
        compile(
            ast.Module(body=config_assertions, type_ignores=[]),
            "<notebook-config-assertions>",
            "exec",
        ),
        {"config": config, "MODEL_ID": MODEL_ID, "METHODS": config["methods"]},
    )


def test_notebook_explains_delayed_mobile_feedback_without_stale_results():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    assert "immediate open after 1 minute" in markdown
    assert "deletion after 15 minutes" in markdown
    assert "delayed read after 120 minutes" in markdown
    assert "archive remains `UNKNOWN` after 240 minutes" in markdown
    assert "sampled preference" in markdown
    assert "utility-optimal route" in markdown
    assert "this exact delayed-feedback protocol" in markdown
    assert "59.58%" not in markdown


def test_notebook_discloses_promoted_reinforce_without_privileged_feedback():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    rendered = "\n".join(cell.source for cell in notebook.cells)
    compact = " ".join(rendered.split())
    for phrase in (
        "learner-only outcome map",
        "`+5` for an immediate push open",
        "`-1` for a delayed push open",
        "`-5` for a push deletion",
        "`0` for a digest open",
        "`-5` for a digest deletion",
        "`UNKNOWN` is censored before mapping",
        "batch size eight",
        "fixed zero baseline",
        "entropy coefficient `1.0`",
        "gradient-norm clipping at `1.0`",
        "selected in-sample on the same canonical seeds 0–2",
        "no disjoint confirmation seeds were run",
        "same `semantic-title-body-sharp-t001` streams",
        "$T=0.01$ evaluator sampling",
    ):
        assert phrase in compact
    assert (
        "This shaped training signal is separate from the shared "
        "observable-reward metric" in compact
    )
    assert (
        "hidden preference, utilities, deadline, urgency, affinity, and "
        "scenario taxonomy never enter the learner" in compact
    )


def test_notebook_presents_the_full_six_method_byte_reproduction():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    compact = " ".join(markdown.split())
    assert "Follow the six-method pipeline" in markdown
    assert "three paired seeds × six methods × 240 decisions" in compact
    assert "Base, ICL, RAG, REINFORCE, RFT, and Online-SDFT" in compact
    assert "play → understand → reproduce" in markdown
    assert "2–4. Follow the pipeline" in markdown
    assert "6. Inspect" in markdown
    assert "Byte-identical reproduction" in markdown
    assert "One seed × Online-SDFT" not in markdown


def test_notebook_explains_t4_preference_and_cpu_only_fallback():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    assert "T4 GPU is preferred" in markdown
    assert "CPU-only runtime is" in markdown
    assert "GPU budget is exhausted" in markdown
    assert "Apple MPS" in markdown
    assert "CUDA/FP16" in markdown
    assert "cannot honestly claim byte identity with MPS" in markdown
