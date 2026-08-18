"""Configuration and published-result gates for the live protocol."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from online_sdft.config import (
    DATASET_NUMPY_VERSION,
    DATASET_VERSION,
    DECISION_INTERVAL_MINUTES,
    DIGEST_DELIVERY_DELAY_MINUTES,
    FEEDBACK_WINDOWS_MINUTES,
    ICL_K,
    METHODS,
    MODEL_ID,
    OBSERVED_OUTCOME_REWARDS,
    PREFERENCE_SAMPLING_TEMPERATURE,
    PUBLISHED_RESULTS_DATASET_VERSION,
    PROMPT_STYLE,
    PROMPT_TOKEN_BUDGET,
    RAG_K,
    RAG_TEXT_WEIGHT,
    REINFORCE_BASELINE_STEP,
    REINFORCE_BATCH_SIZE,
    REINFORCE_ENTROPY_COEF,
    REINFORCE_LR,
    REINFORCE_MAX_GRAD_NORM,
    REINFORCE_TRAINING_OUTCOME_REWARDS,
    RFT_LR,
    RFT_PROTOCOL_VERSION,
    RFT_SAMPLING_TEMPERATURE,
    RFT_SETTINGS_PROVENANCE,
    SDFT_DISTILL_TEMPERATURE,
    SYSTEM_PROMPT,
    TEACHER_PROMPT_VERSION,
    TEACHER_REASONING_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
)
from online_sdft.experiment import (
    RFT_CANDIDATE_RNG_OFFSET,
    RFT_CANDIDATE_SAMPLER,
    experiment_config,
)
from online_sdft.methods import (
    DEFAULT_REINFORCE_SETTINGS,
    DEFAULT_RFT_SETTINGS,
    DEFAULT_SDFT_SETTINGS,
    OnlineSDFTSettings,
)


class _ConfigStub:
    device = "cpu"
    trainable_parameters = 172_032


_RFT_CONFIGURED_DIFFERENCES = (
    "temperature-8 sample-filter hard targets and a tuned 7e-4 learning rate "
    "with replay-32 epsilon-greedy serving versus Online-SDFT's replay-64 "
    "recency/probe/taper schedule and reliability-conditioned soft targets"
)


def _expected_peft_architecture() -> dict:
    return {
        "implementation": "peft.LoraConfig + peft.get_peft_model",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "self_attn.out_proj",
        ],
        "layers_to_transform": [2, 4, 6, 8, 10, 12],
        "layers_pattern": "layers",
        "bias": "none",
        "init_lora_weights": True,
        "ensure_weight_tying": False,
        "merged_for_serving": False,
    }


def _assert_common_lora_contract(config: dict) -> None:
    architecture = _expected_peft_architecture()
    assert architecture["init_lora_weights"] is True
    assert config["online_sdft_peft_architecture"] == architecture
    assert config["online_rft_peft_architecture"] == architecture
    assert config["online_reinforce_peft_architecture"] == architecture
    assert (
        config["reinforce_trainable_parameters"]
        == config["online_rft_trainable_parameters"]
        == config["online_sdft_trainable_parameters"]
        == _ConfigStub.trainable_parameters
    )


def test_live_protocol_uses_delayed_phone_callbacks_without_shadow_labels():
    assert METHODS == (
        "Base",
        "ICL",
        "RAG",
        "REINFORCE",
        "RFT",
        "Online-SDFT",
    )
    assert DECISION_INTERVAL_MINUTES > 0
    assert FEEDBACK_WINDOWS_MINUTES["INTERRUPT_IMMEDIATE"] > 0
    assert FEEDBACK_WINDOWS_MINUTES["INTERRUPT_DELAYED_READ"] > (
        FEEDBACK_WINDOWS_MINUTES["INTERRUPT_IMMEDIATE"]
    )
    assert FEEDBACK_WINDOWS_MINUTES["LATER"] >= (
        FEEDBACK_WINDOWS_MINUTES["INTERRUPT_DELAYED_READ"]
    )
    assert FEEDBACK_WINDOWS_MINUTES["ARCHIVE"] > FEEDBACK_WINDOWS_MINUTES["LATER"]
    assert TEACHER_PROMPT_VERSION == "concise-causal-v3"
    assert "on-device notification router" in SYSTEM_PROMPT.lower()
    assert "choose exactly one route" in SYSTEM_PROMPT.lower()
    assert "current notification" in SYSTEM_PROMPT.lower()
    assert "past completed interactions" in SYSTEM_PROMPT.lower()
    assert "notification and observed callback" in TEACHER_SYSTEM_PROMPT.lower()
    assert "hidden label or unchosen outcome" in TEACHER_SYSTEM_PROMPT.lower()
    assert "a digest open after later leaves" in TEACHER_SYSTEM_PROMPT.lower()
    assert "interrupt versus later unresolved" in TEACHER_SYSTEM_PROMPT.lower()
    assert "unknown supports no route" in TEACHER_SYSTEM_PROMPT.lower()
    assert "one short paragraph" in TEACHER_REASONING_SYSTEM_PROMPT.lower()
    assert "only its observed behavior" in TEACHER_REASONING_SYSTEM_PROMPT.lower()
    assert "do not invent a hidden label" in TEACHER_REASONING_SYSTEM_PROMPT.lower()
    assert "explain uncertainty" in TEACHER_REASONING_SYSTEM_PROMPT.lower()
    assert "do not add explanation" not in TEACHER_SYSTEM_PROMPT.lower()
    assert "do not add explanation" not in TEACHER_REASONING_SYSTEM_PROMPT.lower()


@pytest.mark.parametrize(
    ("template", "max_characters", "max_words"),
    [
        (SYSTEM_PROMPT, 280, 45),
        (TEACHER_SYSTEM_PROMPT, 380, 65),
        (TEACHER_REASONING_SYSTEM_PROMPT, 320, 55),
    ],
)
def test_prompt_templates_remain_concise(template, max_characters, max_words):
    assert len(template) <= max_characters
    assert len(template.split()) <= max_words


def test_default_online_sdft_and_rft_settings_match_canonical_configuration():
    expected_sdft = {
        "learning_rate": 1e-3,
        "replay_size": 64,
        "replay_prompt_examples": 0,
        "batch_size": 8,
        "update_steps": 2,
        "warmup_examples": 4,
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "lora_target_modules": (
            "q_proj",
            "k_proj",
            "v_proj",
            "self_attn.out_proj",
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
        "reliable_behavior_weight": 0.9,
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
        "interrupt_probe_max_confidence": 0.6,
        "propensity_weight_mode": "none",
        "propensity_weight_cap": 4.0,
    }
    assert DEFAULT_SDFT_SETTINGS.to_dict() == expected_sdft

    expected_rft_student = dict(expected_sdft)
    expected_rft_student.update(
        {
            "learning_rate": 7e-4,
            "replay_size": 32,
            "replay_recency_half_life": None,
            "behavior_mode": "epsilon_greedy",
            "behavior_epsilon": 0.06,
            "exploration_taper_start_step": None,
            "exploration_taper_half_life": None,
            "interrupt_probe_mix": 0.0,
            "interrupt_probe_half_life": None,
            "interrupt_probe_max_confidence": 1.0,
        }
    )
    assert DEFAULT_RFT_SETTINGS.student_settings.to_dict() == expected_rft_student


def test_experiment_config_describes_the_same_mobile_contract():
    config = experiment_config(3, 0, MODEL_ID, _ConfigStub())
    assert config["student_model"] == MODEL_ID
    assert config["dataset_version"] == DATASET_VERSION == (
        "semantic-title-body-sharp-t001"
    )
    assert config["dataset_numpy_version"] == DATASET_NUMPY_VERSION
    assert config["dataset_fingerprint"] == (
        "986cdf1a7d5fcc04c2b33f1bf90a1fc4f24a97ee85e663370382d8a67e4c932d"
    )
    assert set(config["method_dataset_versions"]) == set(METHODS)
    assert set(config["method_dataset_fingerprints"]) == set(METHODS)
    assert set(config["method_dataset_versions"].values()) == {DATASET_VERSION}
    assert set(config["method_dataset_fingerprints"].values()) == {
        config["dataset_fingerprint"]
    }
    assert config["gold_action_sampling"] == {
        "distribution": (
            "probability-power temperature scaling of normalized "
            "evaluator-utility weights"
        ),
        "negative_utility_handling": (
            "subtract the event minimum only when it is negative"
        ),
        "temperature": PREFERENCE_SAMPLING_TEMPERATURE,
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
    assert PREFERENCE_SAMPLING_TEMPERATURE == 0.01
    assert config["teacher_model"] == MODEL_ID
    assert config["methods"] == METHODS
    assert config["decision_interval_minutes"] == DECISION_INTERVAL_MINUTES
    assert config["digest_delivery_delay_minutes"] == (
        DIGEST_DELIVERY_DELAY_MINUTES
    )
    assert config["feedback_windows_minutes"] == FEEDBACK_WINDOWS_MINUTES
    assert config["history_rendering"] == (
        "alternating notification/route demonstrations from reliable singleton "
        "callbacks only; ambiguous and UNKNOWN interactions remain retained but "
        "are not prompted"
    )
    assert config["prompt_style"] == PROMPT_STYLE == "causal_demos"
    assert config["icl_examples"] == ICL_K == 3
    assert config["rag_examples"] == RAG_K == 3
    assert config["rag_text_weight"] == RAG_TEXT_WEIGHT == 0.5
    assert config["online_sdft_settings"] == DEFAULT_SDFT_SETTINGS.to_dict()
    assert config["online_rft_settings"] == DEFAULT_RFT_SETTINGS.to_dict()
    assert config["online_rft_settings"]["student_settings"] == (
        DEFAULT_RFT_SETTINGS.student_settings.to_dict()
    )
    rft_student = config["online_rft_settings"]["student_settings"]
    sdft_student = config["online_sdft_settings"]
    assert rft_student["learning_rate"] == RFT_LR == pytest.approx(7e-4)
    assert {
        key for key in sdft_student if sdft_student[key] != rft_student[key]
    } == {
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
    assert config["online_rft_settings"]["candidate_count"] == 1
    assert config["online_rft_settings"]["sampling_mode"] == "categorical"
    assert config["online_rft_settings"]["sampling_temperature"] == (
        RFT_SAMPLING_TEMPERATURE
    ) == 8.0
    _assert_common_lora_contract(config)
    assert config["student_backbone"] == "frozen Liquid LFM base weights"
    assert config["student_backbone_trainable_parameters"] == 0
    assert config["rft_protocol_version"] == RFT_PROTOCOL_VERSION == (
        "teacher-categorical-k1-temperature8-singleton"
    )
    assert config["rft_settings_provenance"] == RFT_SETTINGS_PROVENANCE == (
        "fixed-temperature8-lr7e-4-same-lora-architecture"
    )
    assert config["rft_candidate_rng_offset"] == RFT_CANDIDATE_RNG_OFFSET == 83
    assert config["rft_candidate_policy"].startswith("K=1 categorical sample")
    assert "temperature 8" in config["rft_candidate_policy"]
    assert config["rft_candidate_sampler"] == {
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
    assert "reliable singleton" in config["rft_acceptance_filter"]
    assert "one-hot cross-entropy" in config["rft_target"]
    assert config["rft_capacity_match"] == {
        "settings_source": "online_rft_settings.student_settings",
        "same_frozen_base_model": True,
        "same_lora_architecture": True,
        "same_adapter_parameter_count": True,
        "same_adapter_initialization": True,
        "same_replay_schedule": False,
        "same_optimizer_hyperparameters": False,
        "same_adapter_disabled_teacher": True,
        "same_teacher_forward_budget": (
            "one hindsight distribution for each matured non-UNKNOWN callback"
        ),
        "configured_differences": _RFT_CONFIGURED_DIFFERENCES,
    }
    assert config["sdft_replay_prompt_examples"] == 0
    assert config["sdft_replay_prompt_policy"] == (
        "disabled; serving uses only learned finite parameters"
    )
    assert config["teacher_prompt_version"] == TEACHER_PROMPT_VERSION
    assert config["prompt_token_budget"] == PROMPT_TOKEN_BUDGET == 768
    assert "delayed observed user selection" in config["teacher_policy"]
    assert "UNKNOWN stays censored" in config["teacher_policy"]
    assert "no scalar reward" in config["teacher_policy"]
    assert config["observed_reward"] == OBSERVED_OUTCOME_REWARDS
    assert "no end-of-horizon flush" in config["update_timing"]
    assert config["sdft_distill_temperature"] == SDFT_DISTILL_TEMPERATURE
    assert "digest open ambiguous" in config["teacher_policy"]


def test_experiment_config_reports_action_token_lora_reinforce():
    config = experiment_config(3, 0, MODEL_ID, _ConfigStub())
    assert config["reinforce_lr"] == REINFORCE_LR == 1e-4
    assert config["reinforce_batch_size"] == REINFORCE_BATCH_SIZE == 8
    assert config["reinforce_policy"] == (
        "action-token REINFORCE trains only the same reset LFM PEFT LoRA "
        "adapter used by RFT and Online-SDFT"
    )
    assert config["reinforce_initialization"] == (
        "the common PEFT LoRA initialization restored before this arm"
    )
    _assert_common_lora_contract(config)
    assert config["online_reinforce_settings"] == (
        DEFAULT_REINFORCE_SETTINGS.to_dict()
    )
    assert config["reinforce_optimizer"] == (
        "AdamW autograd over exactly 8 newly matured known factual-outcome "
        "callbacks; learner reward from reinforce_training_reward outcome "
        "map; each row consumed once; no replay; incomplete horizon batch "
        "not flushed"
    )
    assert config["reinforce_baseline"] == (
        "fixed zero causal baseline; step=0.0"
    )
    assert REINFORCE_BASELINE_STEP == 0.0
    assert config["reinforce_entropy_coef"] == REINFORCE_ENTROPY_COEF == 1.0
    assert (
        config["reinforce_max_grad_norm"]
        == REINFORCE_MAX_GRAD_NORM
        == 1.0
    )
    assert config["reinforce_capacity_match"] == {
        "settings_source": "online_sdft_settings LoRA architecture",
        "same_physical_model": True,
        "same_frozen_base_model": True,
        "same_lora_architecture": True,
        "same_adapter_parameter_count": True,
        "same_adapter_initialization": True,
        "adapter_reset_before_arm": True,
        "uses_hindsight_teacher": False,
    }
    assert config["observed_reward"] == OBSERVED_OUTCOME_REWARDS
    assert config["reinforce_training_reward"] == {
        "source": "matured executed-surface factual outcome only",
        "outcome_map": REINFORCE_TRAINING_OUTCOME_REWARDS,
        "unknown_selection": (
            "censored before outcome mapping; no gradient target"
        ),
        "reported_metric": (
            "learner-only shaping; rollout observed_feedback_reward and "
            "cumulative_observed_reward use the shared observed_reward map"
        ),
    }


def test_experiment_config_reports_the_configured_online_sdft_settings():
    settings = OnlineSDFTSettings(
        learning_rate=2e-4,
        replay_size=12,
        replay_prompt_examples=3,
        batch_size=3,
        update_steps=2,
        teacher_temperature=0.7,
        reasoning_tokens=16,
        target_mode="causal_fusion",
        reliable_teacher_weight=0.3,
        reliable_decision_weight=0.2,
        reliable_behavior_weight=0.5,
        ambiguous_teacher_weight=0.5,
        ambiguous_decision_weight=0.3,
        ambiguous_behavior_weight=0.2,
        replay_strategy="selection_balanced",
    )
    config = experiment_config(
        1,
        0,
        MODEL_ID,
        _ConfigStub(),
        sdft_settings=settings,
    )
    assert config["online_sdft_settings"] == settings.to_dict()
    assert config["online_rft_settings"] == DEFAULT_RFT_SETTINGS.to_dict()
    assert config["online_rft_settings"]["student_settings"] != settings.to_dict()
    assert config["online_rft_trainable_parameters"] == (
        config["online_sdft_trainable_parameters"]
    )
    assert config["sdft_lr"] == settings.learning_rate
    assert config["sdft_replay_size"] == settings.replay_size
    assert config["sdft_replay_prompt_examples"] == 3
    assert "no similarity retrieval" in config["sdft_replay_prompt_policy"]
    assert config["sdft_batch_size"] == settings.batch_size
    assert config["sdft_update_steps"] == settings.update_steps
    assert config["teacher_temperature"] == settings.teacher_temperature
    reliability = config["sdft_evidence_reliability"]
    assert reliability["reliable_singleton"]["behavior_weight"] == 0.5
    assert reliability["ambiguous_digest_open"]["behavior_weight"] == 0.2
    assert reliability["censored_unknown"] == "no target or update"


def test_committed_ablation_contains_the_current_six_method_result_set():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "outputs" / "bandit" / "summary.json").read_text()
    )
    config = payload["config"]
    summary = payload["summary"]
    assert config["seeds"] == 3
    assert config["dataset_version"] == PUBLISHED_RESULTS_DATASET_VERSION == (
        "semantic-title-body-sharp-t001"
    )
    assert config["dataset_fingerprint"] == (
        "986cdf1a7d5fcc04c2b33f1bf90a1fc4f24a97ee85e663370382d8a67e4c932d"
    )
    assert config["dataset_numpy_version"] == DATASET_NUMPY_VERSION == "2.4.6"
    assert config["runtime_numpy_version"] == DATASET_NUMPY_VERSION
    assert config["device"] == "mps"
    assert config["methods"] == list(METHODS)
    assert set(summary) == set(METHODS)
    source_fingerprint = config["source_fingerprint"]
    assert source_fingerprint["algorithm"] == "sha256"
    observed_source_hashes = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in source_fingerprint["files"]
    }
    assert observed_source_hashes == source_fingerprint["files"]
    assert source_fingerprint["sha256"] == hashlib.sha256(
        json.dumps(
            source_fingerprint["files"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert "rft_selection_artifact" not in config
    assert not any(
        relative.startswith("outputs/tuning/")
        or relative.startswith("scripts/tune_")
        for relative in source_fingerprint["files"]
    )
    assert config["method_dataset_fingerprints"] == {
        method: config["dataset_fingerprint"] for method in METHODS
    }
    assert config["gold_action_sampling"] == {
        "distribution": (
            "probability-power temperature scaling of normalized "
            "evaluator-utility weights"
        ),
        "negative_utility_handling": (
            "subtract the event minimum only when it is negative"
        ),
        "temperature": PREFERENCE_SAMPLING_TEMPERATURE,
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
    assert config["prompt_style"] == PROMPT_STYLE == "causal_demos"
    assert config["icl_examples"] == ICL_K == 3
    assert config["rag_examples"] == RAG_K == 3
    assert config["rag_text_weight"] == RAG_TEXT_WEIGHT == 0.5
    assert config["online_sdft_settings"] == json.loads(
        json.dumps(DEFAULT_SDFT_SETTINGS.to_dict())
    )
    assert config["adaptive_default_configuration_provenance"][
        "Online-SDFT"
    ] == {
        "selection_seeds": [0, 1, 2],
        "disjoint_confirmation": False,
        "interpretation": (
            "user-requested in-sample tuning on the canonical streams"
        ),
    }
    assert config["adaptive_default_configuration_provenance"][
        "REINFORCE"
    ] == {
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
    assert "adapter_warm_start" not in config["online_sdft_settings"]
    assert "online_sdft_adapter_initialization" not in config
    serialized_config = json.dumps(config, sort_keys=True).lower()
    assert "rubric" not in serialized_config
    assert "warm_start" not in serialized_config
    assert config["online_rft_settings"] == json.loads(
        json.dumps(DEFAULT_RFT_SETTINGS.to_dict())
    )
    assert config["online_rft_settings"]["student_settings"] == (
        json.loads(json.dumps(DEFAULT_RFT_SETTINGS.student_settings.to_dict()))
    )
    assert config["online_rft_settings"]["student_settings"][
        "learning_rate"
    ] == RFT_LR == pytest.approx(7e-4)
    _assert_common_lora_contract(config)
    assert config["online_reinforce_settings"] == json.loads(
        json.dumps(DEFAULT_REINFORCE_SETTINGS.to_dict())
    )
    assert config["reinforce_lr"] == REINFORCE_LR == 1e-4
    assert config["reinforce_max_grad_norm"] == REINFORCE_MAX_GRAD_NORM
    assert config["reinforce_capacity_match"]["uses_hindsight_teacher"] is False
    assert config["rft_capacity_match"]["same_optimizer_hyperparameters"] is False
    assert config["rft_capacity_match"]["same_replay_schedule"] is False
    assert config["rft_capacity_match"]["configured_differences"] == (
        _RFT_CONFIGURED_DIFFERENCES
    )
    assert config["rft_protocol_version"] == RFT_PROTOCOL_VERSION
    assert config["rft_settings_provenance"] == RFT_SETTINGS_PROVENANCE
    assert config["sdft_replay_prompt_examples"] == 0
    assert config["sdft_replay_prompt_policy"] == (
        "disabled; serving uses only learned finite parameters"
    )
    assert config["observed_reward"] == OBSERVED_OUTCOME_REWARDS
    assert "ICL/RAG retain direct completed interactions" in config[
        "learning_signal"
    ]
    assert "prompt only reliable singleton route evidence" in config[
        "learning_signal"
    ]
    assert "notification title/body" in config["teacher_policy"]
    with (root / "outputs" / "bandit" / "per_seed_metrics.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 3 * len(METHODS)
    qualitative = json.loads(
        (root / "outputs" / "bandit" / "qualitative_examples.json").read_text()
    )
    assert qualitative
    assert all(row["notification_title"] for row in qualitative)
    assert all(row["notification_body"] for row in qualitative)
    assert all(set(row["methods"]) == set(METHODS) for row in qualitative)
    rft_examples = [row["methods"]["RFT"] for row in qualitative]
    assert all(
        set(row) >= {"rft_candidate_action", "rft_accepted", "rft_reason"}
        for row in rft_examples
    )
    sdft_examples = [row["methods"]["Online-SDFT"] for row in qualitative]
    expected_fusion_weights = {
        "reliable_singleton": {
            "teacher": 0.05,
            "decision": 0.05,
            "behavior": 0.9,
        },
        "ambiguous_digest_open": {
            "teacher": 0.0,
            "decision": 1.0,
            "behavior": 0.0,
        },
    }
    applied_sdft_examples = [
        row
        for row in sdft_examples
        if row["sdft_evidence_reliability"] in expected_fusion_weights
    ]
    assert applied_sdft_examples
    assert all(
        row["sdft_fusion_weights"]
        == expected_fusion_weights[row["sdft_evidence_reliability"]]
        for row in applied_sdft_examples
    )
    rft_diagnostics = payload["rft_diagnostics"]
    total = rft_diagnostics["total"]
    assert total["attempted"] == total["accepted"] + total["rejected"]
    assert total["rejected"] == sum(total["rejection_reasons"].values())
    assert set(total["rejection_reasons"]) <= {
        "ambiguous_unverified",
        "teacher_mismatch",
    }
    assert total["accepted"] > 0
    assert total["update_count"] > 0
    assert 0.0 < total["acceptance_rate"] < 1.0
    assert set(rft_diagnostics["per_seed"]) == {"0", "1", "2"}
    for method in ("ICL", "RAG", "REINFORCE", "RFT", "Online-SDFT"):
        assert summary[method]["online_accuracy"]["mean"] > (
            summary["Base"]["online_accuracy"]["mean"]
        )
        assert summary[method]["cum_regret"]["mean"] < (
            summary["Base"]["cum_regret"]["mean"]
        )
    expected_canonical_means = {
        "Base": (0.2819444444, 164.6662596),
        "ICL": (0.4222222222, 131.4833770),
        "RAG": (0.5000000000, 106.6210090),
        "REINFORCE": (0.3208333333, 157.9920452),
        "RFT": (0.5277777778, 105.2600741),
        "Online-SDFT": (0.7027777778, 44.7627261),
    }
    for method, (accuracy, regret) in expected_canonical_means.items():
        method_metrics = [row for row in metrics if row["method"] == method]
        assert len(method_metrics) == config["seeds"]
        per_seed_accuracy_mean = sum(
            float(row["online_accuracy"]) for row in method_metrics
        ) / len(method_metrics)
        per_seed_regret_mean = sum(
            float(row["cum_regret"]) for row in method_metrics
        ) / len(method_metrics)
        assert summary[method]["online_accuracy"]["mean"] == pytest.approx(
            per_seed_accuracy_mean
        )
        assert summary[method]["cum_regret"]["mean"] == pytest.approx(
            per_seed_regret_mean
        )
        assert summary[method]["online_accuracy"]["mean"] == pytest.approx(
            accuracy
        )
        assert summary[method]["cum_regret"]["mean"] == pytest.approx(regret)
    total_decisions = config["seeds"] * config["stream_length"]
    sdft_metrics = [row for row in metrics if row["method"] == "Online-SDFT"]
    exact_matches = sum(
        round(float(row["online_accuracy"]) * config["stream_length"])
        for row in sdft_metrics
    )
    assert total_decisions == 720
    assert exact_matches == 506
    assert exact_matches >= 505
    assert summary["Online-SDFT"]["online_accuracy"]["mean"] == pytest.approx(
        506 / 720
    )
    assert summary["Online-SDFT"]["cum_regret"]["mean"] == pytest.approx(
        44.76272608109617
    )
    base_metrics = [row for row in metrics if row["method"] == "Base"]
    reinforce_metrics = [
        row for row in metrics if row["method"] == "REINFORCE"
    ]
    base_exact_matches = sum(
        round(float(row["online_accuracy"]) * config["stream_length"])
        for row in base_metrics
    )
    reinforce_exact_matches = sum(
        round(float(row["online_accuracy"]) * config["stream_length"])
        for row in reinforce_metrics
    )
    assert base_exact_matches == 203
    assert reinforce_exact_matches == 231
    assert reinforce_exact_matches > base_exact_matches
    assert summary["REINFORCE"]["cum_regret"]["mean"] < (
        summary["Base"]["cum_regret"]["mean"]
    )
    accuracy_leader = max(
        summary,
        key=lambda method: summary[method]["online_accuracy"]["mean"],
    )
    regret_leader = min(
        summary,
        key=lambda method: summary[method]["cum_regret"]["mean"],
    )
    assert accuracy_leader == "Online-SDFT"
    assert regret_leader == "Online-SDFT"
