"""Interaction-order tests for delayed mobile feedback."""

import csv
import io
import json

import numpy as np
import pytest

import online_sdft.experiment as experiment_module
from online_sdft.config import (
    ACTIONS,
    EXPLORATION_EPSILON,
    METHODS,
    MODEL_ID,
    OBSERVED_OUTCOME_REWARDS,
    REINFORCE_BATCH_SIZE,
    REINFORCE_BASELINE_STEP,
    REINFORCE_ENTROPY_COEF,
    REINFORCE_MAX_GRAD_NORM,
    REINFORCE_TRAINING_OUTCOME_REWARDS,
    RFT_PROTOCOL_VERSION,
    RFT_SETTINGS_PROVENANCE,
    STREAM_LENGTH,
)
from online_sdft.environment import (
    NotificationRoutingEnvironment,
    StudentObservation,
    TeacherObservation,
)
from online_sdft.experiment import (
    METHOD_RNG_OFFSETS,
    RFT_CANDIDATE_RNG_OFFSET,
    RFT_CANDIDATE_SAMPLER,
    experiment_config,
    rft_event_uniform,
    rft_inverse_cdf_sample,
    rft_sampling_distribution,
    run_method,
)
from online_sdft.methods import (
    DEFAULT_RFT_SETTINGS,
    DEFAULT_RFT_STUDENT_SETTINGS,
    DEFAULT_SDFT_SETTINGS,
    OnlineSDFTSettings,
    RFTSettings,
    causal_evidence_reliability,
    causal_route_support,
)
from online_sdft.privilege import FactualCallback
from online_sdft.reporting import summarize_rft_diagnostics


class OrderedPolicy:
    def __init__(self, calls):
        self.calls = calls
        self.updates = []
        self.update_weights = []
        self.support_updates = []
        self.support_update_weights = []
        self.reinforce_updates = []
        self.example_counts = []

    def start_run(self, learning_rate):
        self.learning_rate = learning_rate
        self.updates.clear()
        self.update_weights.clear()
        self.support_updates.clear()
        self.support_update_weights.clear()
        self.reinforce_updates.clear()

    def probs(self, context, examples=None):
        del context
        self.calls.append("act")
        self.example_counts.append(len(examples or []))
        return np.array([0.2, 0.6, 0.2])

    def teacher_probs(self, observation, examples=None):
        assert isinstance(observation, TeacherObservation)
        assert examples is None
        self.calls.append("teacher")
        return np.array([0.2, 0.5, 0.3])

    def update(self, batch, sample_weights=None):
        self.calls.append("update")
        self.updates.append(batch)
        self.update_weights.append(
            None if sample_weights is None else np.asarray(sample_weights).copy()
        )
        return 0.0

    def update_support(self, batch, sample_weights=None):
        self.calls.append("update_support")
        self.support_updates.append(batch)
        self.support_update_weights.append(
            None if sample_weights is None else np.asarray(sample_weights).copy()
        )
        return 0.0

    def reinforce_update(self, batch, entropy_coef, max_grad_norm):
        self.calls.append("update")
        self.reinforce_updates.append((batch, entropy_coef, max_grad_norm))
        return 0.0


class OrderedEnvironment(NotificationRoutingEnvironment):
    def __init__(self, calls):
        self.calls = calls
        self.executing = False

    def student_observation(self, event):
        self.calls.append("context")
        return super().student_observation(event)

    def oracle_utilities(self, event):
        if not self.executing:
            self.calls.append("score")
        return super().oracle_utilities(event)

    def execute(self, event, action, rng):
        self.calls.append("execute")
        self.executing = True
        try:
            return super().execute(event, action, rng)
        finally:
            self.executing = False

    def teacher_observation(self, observation, action, callback):
        assert isinstance(observation, StudentObservation)
        assert isinstance(callback, FactualCallback)
        self.calls.append("teacher_view")
        return super().teacher_observation(observation, action, callback)


def run_fast_method(
    method,
    sdft_settings=DEFAULT_SDFT_SETTINGS,
    rft_settings=DEFAULT_RFT_SETTINGS,
):
    calls = []
    environment = OrderedEnvironment(calls)
    stream = environment.make_stream(0)
    calls.clear()  # Ignore evaluator work used to construct the seeded stream.
    policy = OrderedPolicy(calls)
    rollout_buffer = io.StringIO()
    curve_buffer = io.StringIO()
    curve_writer = csv.DictWriter(
        curve_buffer,
        fieldnames=[
            "seed", "method", "t", "phase", "regime", "step_correct",
            "step_feedback_reward", "step_regret", "cum_accuracy",
            "cum_regret", "cum_observed_reward",
        ],
        lineterminator="\n",
    )
    curve_writer.writeheader()
    metrics = run_method(
        0,
        method,
        stream,
        policy,
        rollout_buffer,
        curve_writer,
        environment,
        sdft_settings=sdft_settings,
        rft_settings=rft_settings,
    )
    rollouts = [
        json.loads(line) for line in rollout_buffer.getvalue().splitlines()
    ]
    return calls, policy, metrics, rollouts


def test_experiment_config_records_all_three_adaptive_lora_arms():
    class ConfigPolicy:
        device = "cpu"
        trainable_parameters = 172_032

    config = experiment_config(1, 0, MODEL_ID, ConfigPolicy())
    assert config["rft_protocol_version"] == RFT_PROTOCOL_VERSION
    assert config["rft_settings_provenance"] == RFT_SETTINGS_PROVENANCE == (
        "fixed-temperature8-lr7e-4-same-lora-architecture"
    )
    assert config["online_rft_settings"] == DEFAULT_RFT_SETTINGS.to_dict()
    assert config["online_rft_settings"]["student_settings"] == (
        DEFAULT_RFT_STUDENT_SETTINGS.to_dict()
    )
    assert config["online_rft_settings"]["student_settings"][
        "learning_rate"
    ] == pytest.approx(7e-4)
    assert (
        config["online_rft_trainable_parameters"]
        == config["online_sdft_trainable_parameters"]
        == ConfigPolicy.trainable_parameters
    )
    target_modules = DEFAULT_SDFT_SETTINGS.lora_target_modules
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    else:
        target_modules = list(target_modules)
    layers = DEFAULT_SDFT_SETTINGS.lora_layers_to_transform
    if isinstance(layers, int):
        layers = [layers]
    elif layers is not None:
        layers = list(layers)
    expected_peft_architecture = {
        "implementation": "peft.LoraConfig + peft.get_peft_model",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": DEFAULT_SDFT_SETTINGS.lora_rank,
        "lora_alpha": DEFAULT_SDFT_SETTINGS.lora_alpha,
        "lora_dropout": DEFAULT_SDFT_SETTINGS.lora_dropout,
        "target_modules": target_modules,
        "layers_to_transform": layers,
        "layers_pattern": None if layers is None else "layers",
        "bias": "none",
        "init_lora_weights": True,
        "ensure_weight_tying": False,
        "merged_for_serving": False,
    }
    assert config["online_sdft_peft_architecture"] == (
        expected_peft_architecture
    )
    assert config["online_rft_peft_architecture"] == (
        expected_peft_architecture
    )
    assert config["online_reinforce_peft_architecture"] == (
        expected_peft_architecture
    )
    assert (
        config["reinforce_trainable_parameters"]
        == config["online_rft_trainable_parameters"]
        == config["online_sdft_trainable_parameters"]
        == ConfigPolicy.trainable_parameters
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
    assert "learner-only scalar mapped" in config["learning_signal"]
    assert "reported observed reward retains the shared" in config[
        "learning_signal"
    ]
    assert "K=1 categorical sample at temperature 8" in config[
        "rft_candidate_policy"
    ]
    assert "same model with its adapter disabled" in config[
        "rft_candidate_policy"
    ]
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
    assert "rejected and censored rows never enter replay" in config[
        "rft_target"
    ]
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
        "configured_differences": (
            "temperature-8 sample-filter hard targets and a tuned 7e-4 "
            "learning rate with replay-32 epsilon-greedy serving versus "
            "Online-SDFT's replay-64 recency/probe/taper schedule and "
            "reliability-conditioned soft targets"
        ),
    }
    assert config["student_backbone"] == "frozen Liquid LFM base weights"
    assert config["student_backbone_trainable_parameters"] == 0
    assert config["online_sdft_optimizer"] == {
        "type": "AdamW",
        "learning_rate": DEFAULT_SDFT_SETTINGS.learning_rate,
        "lm_head_learning_rate": DEFAULT_SDFT_SETTINGS.lm_head_learning_rate,
        "lora_a_learning_rate_scale": (
            DEFAULT_SDFT_SETTINGS.lora_a_learning_rate_scale
        ),
        "lm_head_lora_a_learning_rate_scale": (
            DEFAULT_SDFT_SETTINGS.lm_head_lora_a_learning_rate_scale
        ),
        "weight_decay": DEFAULT_SDFT_SETTINGS.optimizer_weight_decay,
        "max_grad_norm": DEFAULT_SDFT_SETTINGS.max_grad_norm,
        "batch_size": DEFAULT_SDFT_SETTINGS.batch_size,
        "update_steps_per_release": DEFAULT_SDFT_SETTINGS.update_steps,
    }
    assert config["teacher_student_model_sharing"] == {
        "model_instances": 1,
        "same_base_parameters": True,
        "student_forward": "LoRA adapter enabled",
        "teacher_forward": "same model with LoRA adapter disabled",
        "teacher_reference": "fixed initial frozen base-model policy",
        "separate_teacher_checkpoint": False,
    }
    assert "student and teacher are one shared" in config["teacher_policy"]
    assert "one physical Liquid model instance" in config[
        "adaptive_model_sharing"
    ]
    assert "same reset-per-arm PEFT LoRA adapter" in config["student_policy"]
    assert config["rft_candidate_rng_offset"] == RFT_CANDIDATE_RNG_OFFSET


def test_rft_temperature_scaling_and_event_sampler_are_deterministic():
    teacher = np.array([0.01, 0.09, 0.90])
    original = teacher.copy()
    expected = np.sqrt(teacher)
    expected /= expected.sum()

    assert rft_sampling_distribution(teacher, 1.0) == pytest.approx(teacher)
    assert rft_sampling_distribution(teacher, 2.0) == pytest.approx(expected)
    assert teacher == pytest.approx(original)

    first = rft_event_uniform(7, "s7-p0-12", 13)
    assert first == rft_event_uniform(7, "s7-p0-12", 13)
    assert 0.0 < first < 1.0
    assert first != rft_event_uniform(7, "s7-p0-13", 14)
    assert first != rft_event_uniform(
        7,
        "s7-p0-12",
        13,
        rng_offset=RFT_CANDIDATE_RNG_OFFSET + 1,
    )
    assert rft_inverse_cdf_sample(np.array([0.2, 0.3, 0.5]), 0.19) == 0
    assert rft_inverse_cdf_sample(np.array([0.2, 0.3, 0.5]), 0.20) == 1
    assert rft_inverse_cdf_sample(np.array([0.2, 0.3, 0.5]), 0.50) == 2


@pytest.mark.parametrize(
    ("teacher", "temperature", "match"),
    [
        ([0.5, 0.5], 1.0, "action count"),
        ([0.5, -0.1, 0.6], 1.0, "finite and non-negative"),
        ([0.0, 0.0, 0.0], 1.0, "positive mass"),
        ([0.2, 0.5, 0.3], 0.0, "positive and finite"),
    ],
)
def test_rft_temperature_scaling_rejects_invalid_inputs(
    teacher,
    temperature,
    match,
):
    with pytest.raises(ValueError, match=match):
        rft_sampling_distribution(np.asarray(teacher), temperature)


def test_experiment_config_records_split_lm_head_lora_a_scale():
    class ConfigPolicy:
        device = "cpu"
        trainable_parameters = 125_952

    settings = OnlineSDFTSettings(
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "lm_head",
        ),
        lora_a_learning_rate_scale=1.0 / 16.0,
        lm_head_lora_a_learning_rate_scale=1.0,
        lm_head_learning_rate=3e-3,
    )

    config = experiment_config(
        1,
        0,
        MODEL_ID,
        ConfigPolicy(),
        sdft_settings=settings,
        rft_settings=RFTSettings(student_settings=settings),
    )

    assert config["online_sdft_settings"]["lora_a_learning_rate_scale"] == (
        pytest.approx(1.0 / 16.0)
    )
    assert config["online_rft_settings"]["student_settings"][
        "lora_a_learning_rate_scale"
    ] == pytest.approx(1.0 / 16.0)
    assert config["online_rft_settings"]["student_settings"][
        "lm_head_lora_a_learning_rate_scale"
    ] == pytest.approx(1.0)
    assert config["online_sdft_optimizer"][
        "lm_head_lora_a_learning_rate_scale"
    ] == pytest.approx(1.0)


def test_experiment_config_records_online_sdft_behavior_epsilon_schedule():
    class ConfigPolicy:
        device = "cpu"
        trainable_parameters = 125_952

    settings = OnlineSDFTSettings(
        behavior_epsilon=0.01,
        behavior_epsilon_half_life=10.0,
        exploration_taper_start_step=160,
        exploration_taper_half_life=20.0,
    )
    config = experiment_config(
        1,
        0,
        MODEL_ID,
        ConfigPolicy(),
        sdft_settings=settings,
    )

    assert config["exploration_epsilon"] == EXPLORATION_EPSILON
    assert config["online_sdft_settings"]["behavior_epsilon"] == 0.01
    assert (
        config["online_sdft_settings"]["behavior_epsilon_half_life"]
        == 10.0
    )
    assert (
        config["online_sdft_settings"]["exploration_taper_start_step"]
        == 160
    )
    assert (
        config["online_sdft_settings"]["exploration_taper_half_life"]
        == 20.0
    )
    assert not config["rft_capacity_match"]["same_replay_schedule"]


def test_main_passes_sdft_settings_to_the_single_model_policy(
    monkeypatch,
    tmp_path,
):
    captured = {}

    class StopAfterConstruction(RuntimeError):
        pass

    def policy_constructor(**kwargs):
        captured.update(kwargs)
        raise StopAfterConstruction

    monkeypatch.setattr(
        experiment_module,
        "LiquidLLMPolicy",
        policy_constructor,
    )
    settings = DEFAULT_SDFT_SETTINGS
    with pytest.raises(StopAfterConstruction):
        experiment_module.main(
            seeds=1,
            output_dir=tmp_path / "outputs",
            figure_dir=tmp_path / "figures",
            sdft_settings=settings,
        )

    assert captured["sdft_settings"] is settings


def test_sdft_replay_prompt_appears_only_after_reliable_feedback_release():
    settings = OnlineSDFTSettings(replay_prompt_examples=1)
    _, policy, _, rollouts = run_fast_method(
        "Online-SDFT",
        sdft_settings=settings,
    )

    recorded_counts = [
        row["sdft_prompt_examples_used"] for row in rollouts
    ]
    assert recorded_counts == policy.example_counts
    first_prompt_index = next(
        index for index, count in enumerate(recorded_counts) if count
    )
    first_prompt_time = rollouts[first_prompt_index]["decision_time_minute"]
    reliable_release_times = [
        row["feedback_released_at_minute"]
        for row in rollouts[:first_prompt_index]
        if row["sdft_evidence_reliability"] == "reliable_singleton"
    ]
    assert reliable_release_times
    assert first_prompt_time >= min(reliable_release_times)
    assert all(count == 0 for count in recorded_counts[:first_prompt_index])


def test_sdft_waits_for_feedback_window_before_teaching():
    calls, policy, metrics, rollouts = run_fast_method("Online-SDFT")
    assert calls[:4] == ["context", "act", "score", "execute"]
    assert "teacher" in calls
    assert metrics["method"] == "Online-SDFT"

    released = [row for row in rollouts if row["teacher_probs"] is not None]
    applied = [
        row for row in released
        if row["lesson_status"] == "soft_target_applied"
    ]
    censored = [
        row for row in rollouts
        if row["lesson_status"] == "censored_no_update"
    ]
    assert applied and censored
    assert 0 < len(applied) < STREAM_LENGTH
    assert policy.learning_rate == DEFAULT_SDFT_SETTINGS.learning_rate
    assert len(policy.updates) == sum(
        row["sdft_updates_applied"] for row in applied
    )
    assert all(
        1 <= len(batch) <= DEFAULT_SDFT_SETTINGS.batch_size
        for batch in policy.updates
    )
    sdft_update_rows = [row for batch in policy.updates for row in batch]
    assert all(
        isinstance(context, str)
        and context
        and np.isclose(target.sum(), 1.0)
        for context, target in sdft_update_rows
    )
    assert any(
        np.count_nonzero(target > 1e-7) > 1
        for _, target in sdft_update_rows
    )
    assert all(
        row["sdft_updates_applied"] == DEFAULT_SDFT_SETTINGS.update_steps
        for row in applied
    )
    assert sorted(row["sdft_update_index"] for row in applied) == list(
        range(
            DEFAULT_SDFT_SETTINGS.update_steps,
            len(applied) * DEFAULT_SDFT_SETTINGS.update_steps + 1,
            DEFAULT_SDFT_SETTINGS.update_steps,
        )
    )
    assert all(
        row["feedback_released_at_minute"]
        >= row["feedback_available_at_minute"]
        > row["decision_time_minute"]
        for row in applied
    )
    assert all(
        row["feedback"]["observed_user_selection"] == "UNKNOWN"
        for row in censored
    )
    assert all(row["teacher_probs"] is None for row in censored)
    expected_weights = {
        "reliable_singleton": {
            "teacher": 0.05,
            "decision": 0.05,
            "behavior": 0.90,
        },
        "ambiguous_digest_open": {
            "teacher": 0.0,
            "decision": 1.0,
            "behavior": 0.0,
        },
    }
    assert {
        row["sdft_evidence_reliability"] for row in applied
    } == set(expected_weights)
    assert all(
        row["sdft_fusion_weights"]
        == expected_weights[row["sdft_evidence_reliability"]]
        for row in applied
    )
    assert all(
        row["sdft_evidence_reliability"] == "censored_unknown"
        and row["sdft_fusion_weights"] is None
        and row["sdft_updates_applied"] == 0
        for row in censored
    )
    assert all(
        row["sdft_evidence_reliability"] is None
        and row["sdft_fusion_weights"] is None
        and row["sdft_update_index"] is None
        and row["sdft_updates_applied"] is None
        for row in rollouts
        if row["feedback_released_at_minute"] is None
    )
    assert any(row["lesson_status"] == "pending_after_horizon" for row in rollouts)


def test_sdft_support_likelihood_trace_and_updates_use_causal_sets():
    settings = OnlineSDFTSettings(
        target_mode="support_likelihood",
        reliable_teacher_weight=0.0,
        reliable_decision_weight=0.0,
        reliable_behavior_weight=0.0,
        ambiguous_teacher_weight=0.0,
        ambiguous_decision_weight=0.0,
        ambiguous_behavior_weight=0.0,
    )
    _, policy, _, rollouts = run_fast_method(
        "Online-SDFT",
        sdft_settings=settings,
    )

    released = [
        row for row in rollouts
        if row["feedback_released_at_minute"] is not None
    ]
    informative = [
        row for row in released
        if row["sdft_evidence_reliability"] != "censored_unknown"
    ]
    censored = [
        row for row in released
        if row["sdft_evidence_reliability"] == "censored_unknown"
    ]

    assert informative and censored
    assert not policy.updates
    assert len(policy.support_updates) == sum(
        row["sdft_updates_applied"] for row in informative
    )
    supports = [
        np.asarray(support)
        for batch in policy.support_updates
        for _, support in batch
    ]
    assert supports
    assert all(
        set(np.unique(support)).issubset({0.0, 1.0})
        and 1 <= support.sum() < len(ACTIONS)
        for support in supports
    )
    assert any(support.tolist() == [1.0, 1.0, 0.0] for support in supports)
    assert all(row["sdft_objective"] == "support_likelihood" for row in released)
    assert all(row["sdft_fusion_weights"] is None for row in released)
    assert all(
        row["lesson_status"]
        in {"support_target_applied", "support_target_buffered"}
        for row in informative
    )
    assert all(row["lesson_status"] == "censored_no_update" for row in censored)


def test_sdft_policy_sampling_behavior_uses_student_distribution():
    settings = OnlineSDFTSettings(behavior_mode="policy_sampling")
    _, _, _, rollouts = run_fast_method(
        "Online-SDFT",
        sdft_settings=settings,
    )

    assert all(
        row["behavior_probs"] == pytest.approx(row["student_probs"])
        for row in rollouts
    )


def test_epsilon_greedy_default_is_bit_exact_and_custom_epsilon_scales():
    probs = np.array([0.2, 0.6, 0.2])
    greedy = np.array([0.0, 1.0, 0.0])
    legacy = (
        (1.0 - EXPLORATION_EPSILON) * greedy
        + EXPLORATION_EPSILON / len(ACTIONS)
    )
    legacy /= legacy.sum()

    default = experiment_module.epsilon_greedy(probs)
    assert default.tobytes() == legacy.tobytes()
    assert experiment_module.epsilon_greedy(
        probs,
        epsilon=0.01,
    ) == pytest.approx([0.01 / 3.0, 0.99 + 0.01 / 3.0, 0.01 / 3.0])


def test_sdft_behavior_epsilon_controls_epsilon_greedy_rollout():
    settings = OnlineSDFTSettings(behavior_epsilon=0.01)
    _, _, _, rollouts = run_fast_method(
        "Online-SDFT",
        sdft_settings=settings,
    )

    expected = dict(
        zip(
            ACTIONS,
            map(
                float,
                experiment_module.epsilon_greedy(
                    np.array([0.2, 0.6, 0.2]),
                    epsilon=0.01,
                ),
            ),
        )
    )
    assert all(
        row["behavior_probs"] == expected
        for row in rollouts
    )
    assert all(row["behavior_epsilon"] == 0.01 for row in rollouts)
    assert all(row["exploration_taper_weight"] == 1.0 for row in rollouts)


def test_sdft_behavior_epsilon_decays_from_step_one_by_half_life():
    settings = OnlineSDFTSettings(
        behavior_epsilon=0.08,
        behavior_epsilon_half_life=10.0,
    )
    _, _, _, rollouts = run_fast_method(
        "Online-SDFT",
        sdft_settings=settings,
    )

    assert rollouts[0]["behavior_epsilon"] == 0.08
    assert rollouts[10]["behavior_epsilon"] == pytest.approx(0.04)
    assert rollouts[10]["behavior_probs"] == pytest.approx(
        {
            "INTERRUPT": 0.04 / 3.0,
            "LATER": 0.96 + 0.04 / 3.0,
            "ARCHIVE": 0.04 / 3.0,
        }
    )


def test_sdft_exploration_taper_is_exact_through_start_then_halves():
    settings = OnlineSDFTSettings(
        behavior_epsilon=0.06,
        exploration_taper_start_step=10,
        exploration_taper_half_life=5.0,
    )
    _, _, _, rollouts = run_fast_method(
        "Online-SDFT",
        sdft_settings=settings,
    )

    baseline = {
        "INTERRUPT": 0.02,
        "LATER": 0.96,
        "ARCHIVE": 0.02,
    }
    assert all(row["behavior_probs"] == baseline for row in rollouts[:10])
    assert all(
        row["exploration_taper_weight"] == 1.0
        for row in rollouts[:10]
    )
    assert rollouts[14]["exploration_taper_weight"] == pytest.approx(0.5)
    assert rollouts[14]["behavior_probs"] == pytest.approx(
        {
            "INTERRUPT": 0.01,
            "LATER": 0.98,
            "ARCHIVE": 0.01,
        }
    )


def test_archive_uniform_probe_has_fixed_feedback_bearing_dose():
    collapsed_archive = np.array([0.0, 0.0, 1.0])

    assert experiment_module.archive_uniform_probe(
        collapsed_archive,
        0.10,
    ) == pytest.approx([0.068, 0.068, 0.864])
    non_archive_top = np.array([0.2, 0.6, 0.2])
    assert experiment_module.archive_uniform_probe(
        non_archive_top,
        0.20,
    ) == pytest.approx(experiment_module.epsilon_greedy(non_archive_top))
    assert experiment_module.archive_uniform_probe(
        collapsed_archive,
        0.10,
        baseline_epsilon=0.01,
    ) == pytest.approx([0.053, 0.053, 0.894])


def test_uncertainty_interrupt_probe_is_decaying_and_has_positive_support():
    uncertain = np.array([0.33, 0.34, 0.33])

    first = experiment_module.uncertainty_interrupt_probe(
        uncertain,
        step=1,
        mix=0.20,
        half_life=10.0,
        max_confidence=0.50,
    )
    one_half_life = experiment_module.uncertainty_interrupt_probe(
        uncertain,
        step=11,
        mix=0.20,
        half_life=10.0,
        max_confidence=0.50,
    )

    assert first == pytest.approx([0.216, 0.768, 0.016])
    assert one_half_life == pytest.approx([0.118, 0.864, 0.018])
    assert (first > 0.0).all()
    assert (one_half_life > 0.0).all()
    lower_epsilon = experiment_module.uncertainty_interrupt_probe(
        uncertain,
        step=1,
        mix=0.20,
        half_life=10.0,
        max_confidence=0.50,
        baseline_epsilon=0.01,
    )
    assert lower_epsilon == pytest.approx(
        [0.20266666666666666, 0.7946666666666667, 0.0026666666666666666]
    )
    assert (lower_epsilon > 0.0).all()
    assert experiment_module.uncertainty_interrupt_probe(
        np.array([0.10, 0.80, 0.10]),
        step=1,
        mix=0.20,
        half_life=10.0,
        max_confidence=0.50,
    ) == pytest.approx([0.02, 0.96, 0.02])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"step": 0}, "positive integer"),
        ({"mix": 0.0}, "mix"),
        ({"half_life": 0.0}, "half-life"),
        ({"max_confidence": 0.2}, "maximum confidence"),
    ],
)
def test_uncertainty_interrupt_probe_rejects_invalid_parameters(
    overrides,
    message,
):
    kwargs = {
        "probs": np.array([0.33, 0.34, 0.33]),
        "step": 1,
        "mix": 0.20,
        "half_life": 10.0,
        "max_confidence": 0.50,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        experiment_module.uncertainty_interrupt_probe(**kwargs)


def test_sdft_uncertainty_interrupt_probe_uses_pre_action_step(monkeypatch):
    def uncertain_probs(self, context, examples=None):
        del context
        self.calls.append("act")
        self.example_counts.append(len(examples or []))
        return np.array([0.33, 0.34, 0.33])

    monkeypatch.setattr(OrderedPolicy, "probs", uncertain_probs)
    settings = OnlineSDFTSettings(
        behavior_mode="uncertainty_interrupt_probe",
        behavior_epsilon=0.01,
        interrupt_probe_mix=0.20,
        interrupt_probe_half_life=10.0,
        interrupt_probe_max_confidence=0.50,
    )
    _, _, _, rollouts = run_fast_method(
        "Online-SDFT",
        sdft_settings=settings,
    )

    assert rollouts[0]["behavior_mode"] == "uncertainty_interrupt_probe"
    assert rollouts[0]["behavior_probs"] == pytest.approx(
        {
            "INTERRUPT": 0.20266666666666666,
            "LATER": 0.7946666666666667,
            "ARCHIVE": 0.0026666666666666666,
        }
    )
    assert rollouts[10]["behavior_probs"] == pytest.approx(
        {
            "INTERRUPT": 0.103,
            "LATER": 0.894,
            "ARCHIVE": 0.003,
        }
    )


def test_sdft_archive_uniform_probe_is_recorded_in_behavior_probs(monkeypatch):
    def archive_probs(self, context, examples=None):
        del context
        self.calls.append("act")
        self.example_counts.append(len(examples or []))
        return np.array([0.0, 0.0, 1.0])

    monkeypatch.setattr(OrderedPolicy, "probs", archive_probs)
    settings = OnlineSDFTSettings(
        behavior_mode="archive_uniform_probe",
        archive_probe_mix=0.20,
    )
    _, _, _, rollouts = run_fast_method(
        "Online-SDFT",
        sdft_settings=settings,
    )

    assert all(
        row["behavior_probs"]
        == pytest.approx(
            {"INTERRUPT": 0.116, "LATER": 0.116, "ARCHIVE": 0.768}
        )
        for row in rollouts
    )


def test_archive_policy_feedback_floor_preserves_raw_policy_and_route_ratio():
    assert experiment_module.archive_policy_feedback_floor(
        np.array([2.0, 3.0, 5.0]),
        0.30,
    ) == pytest.approx([0.2, 0.3, 0.5])
    assert experiment_module.archive_policy_feedback_floor(
        np.array([6.0, 1.0, 3.0]),
        0.50,
    ) == pytest.approx([0.6, 0.1, 0.3])
    assert experiment_module.archive_policy_feedback_floor(
        np.array([1.0, 3.0, 16.0]),
        0.40,
    ) == pytest.approx([0.1, 0.3, 0.6])
    assert experiment_module.archive_policy_feedback_floor(
        np.array([0.0, 0.0, 4.0]),
        0.20,
    ) == pytest.approx([0.1, 0.1, 0.8])


def test_rft_and_sdft_share_the_configured_archive_feedback_floor(monkeypatch):
    def archive_probs(self, context, examples=None):
        del context
        self.calls.append("act")
        self.example_counts.append(len(examples or []))
        return np.array([0.02, 0.08, 0.90])

    monkeypatch.setattr(OrderedPolicy, "probs", archive_probs)
    settings = OnlineSDFTSettings(
        behavior_mode="archive_policy_feedback_floor",
        archive_policy_min_feedback=0.30,
    )

    for method in ("RFT", "Online-SDFT"):
        _, _, _, rollouts = run_fast_method(
            method,
            sdft_settings=settings,
            rft_settings=RFTSettings(student_settings=settings),
        )
        assert all(
            row["behavior_probs"]
            == pytest.approx(
                {"INTERRUPT": 0.06, "LATER": 0.24, "ARCHIVE": 0.70}
            )
            for row in rollouts
        )


def test_sdft_feedback_floor_snips_freezes_decision_time_propensity_through_delay(
    monkeypatch,
):
    def archive_probs(self, context, examples=None):
        del context
        self.calls.append("act")
        self.example_counts.append(len(examples or []))
        return np.array([0.02, 0.08, 0.90])

    monkeypatch.setattr(OrderedPolicy, "probs", archive_probs)
    settings = OnlineSDFTSettings(
        behavior_mode="archive_policy_feedback_floor",
        archive_policy_min_feedback=0.30,
        propensity_weight_mode="feedback_surface_snips",
    )
    _, policy, _, rollouts = run_fast_method(
        "Online-SDFT",
        sdft_settings=settings,
    )

    assert all(
        row["behavior_probs"]
        == pytest.approx(
            {"INTERRUPT": 0.06, "LATER": 0.24, "ARCHIVE": 0.70}
        )
        for row in rollouts
    )
    assert all(
        row["sdft_feedback_propensity"] == pytest.approx(0.30)
        and row["sdft_propensity_weight"] == pytest.approx(10.0 / 3.0)
        for row in rollouts
    )
    assert policy.update_weights
    assert all(
        weights == pytest.approx(np.full(len(weights), 10.0 / 3.0))
        for weights in policy.update_weights
    )
    assert all(
        row["feedback_released_at_minute"]
        >= row["feedback_available_at_minute"]
        > row["decision_time_minute"]
        for row in rollouts
        if row["feedback_released_at_minute"] is not None
    )


def test_rft_samples_then_filters_teacher_routes_after_delayed_feedback():
    calls, policy, metrics, rollouts = run_fast_method("RFT")
    assert calls[:4] == ["context", "act", "score", "execute"]
    assert "teacher" in calls
    assert "teacher_view" in calls
    assert metrics["method"] == "RFT"

    attempted = [
        row for row in rollouts if row["rft_candidate_action"] is not None
    ]
    accepted = [row for row in attempted if row["rft_accepted"] is True]
    rejected = [row for row in attempted if row["rft_accepted"] is False]
    censored = [
        row for row in rollouts if row["rft_reason"] == "censored_unknown"
    ]
    pending = [
        row for row in rollouts if row["lesson_status"] == "pending_after_horizon"
    ]
    assert attempted and accepted and rejected and censored and pending
    assert len(attempted) == len(accepted) + len(rejected)
    assert {row["rft_reason"] for row in accepted} == {"accepted"}
    assert {
        row["rft_reason"] for row in rejected
    } == {"teacher_mismatch", "ambiguous_unverified"}
    assert any(
        row["rft_candidate_action"] != row["teacher_rollout"]
        for row in attempted
    )

    for row in attempted:
        action_index = ACTIONS.index(row["action"])
        candidate_index = ACTIONS.index(row["rft_candidate_action"])
        teacher_probs = np.asarray(
            [row["teacher_probs"][action] for action in ACTIONS]
        )
        candidate_probs = np.asarray(
            [row["rft_candidate_probs"][action] for action in ACTIONS]
        )
        assert candidate_probs == pytest.approx(
            rft_sampling_distribution(
                teacher_probs,
                DEFAULT_RFT_SETTINGS.sampling_temperature,
            )
        )
        assert row["rft_candidate_entropy"] == pytest.approx(
            -(candidate_probs * np.log(candidate_probs)).sum()
        )
        expected_uniform = rft_event_uniform(
            row["seed"],
            row["event_id"],
            row["t"],
        )
        assert row["rft_candidate_uniform"] == expected_uniform
        assert candidate_index == rft_inverse_cdf_sample(
            candidate_probs,
            expected_uniform,
        )
        reliability = causal_evidence_reliability(
            action_index,
            row["feedback"],
        )
        support = causal_route_support(action_index, row["feedback"])
        expected_accepted = bool(
            reliability == "reliable_singleton" and support[candidate_index]
        )
        assert row["rft_accepted"] is expected_accepted
        assert row["feedback_released_at_minute"] >= (
            row["feedback_available_at_minute"]
        ) > row["decision_time_minute"]

    applied = [
        row for row in accepted if row["lesson_status"] == "rft_target_applied"
    ]
    buffered = [
        row for row in accepted if row["lesson_status"] == "rft_target_buffered"
    ]
    assert len(buffered) == DEFAULT_RFT_STUDENT_SETTINGS.warmup_examples - 1
    assert all(
        row["rft_updates_applied"] == DEFAULT_RFT_STUDENT_SETTINGS.update_steps
        for row in applied
    )
    assert policy.learning_rate == DEFAULT_RFT_STUDENT_SETTINGS.learning_rate
    assert len(policy.updates) == sum(
        row["rft_updates_applied"] for row in applied
    )
    rft_update_rows = [row for batch in policy.updates for row in batch]
    assert rft_update_rows
    assert all(
        isinstance(context, str)
        and context
        and np.isclose(target.sum(), 1.0)
        and np.count_nonzero(target) == 1
        for context, target in rft_update_rows
    )
    assert all(row["rft_updates_applied"] == 0 for row in buffered)
    assert all(
        row["rft_updates_applied"] == 0
        and row["lesson_status"]
        in {
            "rft_rejected_ambiguous_support",
            "rft_rejected_teacher_candidate",
        }
        for row in rejected
    )
    assert all(
        row["rft_candidate_action"] is None
        and row["rft_accepted"] is None
        and row["rft_updates_applied"] == 0
        and row["lesson_status"] == "censored_no_update"
        for row in censored
    )
    assert all(
        row["rft_candidate_action"] is None
        and row["rft_accepted"] is None
        and row["rft_reason"] is None
        and row["rft_update_index"] is None
        and row["rft_updates_applied"] is None
        for row in pending
    )

    diagnostics = summarize_rft_diagnostics(rollouts)
    total = diagnostics["total"]
    assert set(diagnostics["per_seed"]) == {"0"}
    assert total["attempted"] == len(attempted)
    assert total["accepted"] == len(accepted)
    assert total["rejected"] == len(rejected)
    assert sum(total["rejection_reasons"].values()) == len(rejected)
    assert sum(total["proposal_counts"].values()) == len(attempted)
    assert sum(total["accepted_counts"].values()) == len(accepted)
    assert total["mean_proposal_entropy"] == pytest.approx(
        np.mean([row["rft_candidate_entropy"] for row in attempted])
    )
    assert total["acceptance_rate"] == len(accepted) / len(attempted)
    assert total["update_count"] == max(
        row["rft_update_index"] for row in rollouts
        if row["rft_update_index"] is not None
    )
    assert total["censored_unknown"] == len(censored)
    assert total["pending_after_horizon"] == len(pending)


def test_rft_candidate_rng_isolated_from_action_feedback_and_replay_rngs(
    monkeypatch,
):
    assert RFT_CANDIDATE_RNG_OFFSET not in {
        1,
        2,
        *METHOD_RNG_OFFSETS.values(),
    }

    def run_with_candidate_offset(offset):
        monkeypatch.setattr(
            experiment_module,
            "RFT_CANDIDATE_RNG_OFFSET",
            offset,
        )
        calls = []
        environment = OrderedEnvironment(calls)
        policy = OrderedPolicy(calls)
        rollout_buffer = io.StringIO()
        curve_buffer = io.StringIO()
        curve_writer = csv.DictWriter(
            curve_buffer,
            fieldnames=[
                "seed",
                "method",
                "t",
                "phase",
                "regime",
                "step_correct",
                "step_feedback_reward",
                "step_regret",
                "cum_accuracy",
                "cum_regret",
                "cum_observed_reward",
            ],
            lineterminator="\n",
        )
        curve_writer.writeheader()
        run_method(
            0,
            "RFT",
            environment.make_stream(0)[:48],
            policy,
            rollout_buffer,
            curve_writer,
            environment,
            sdft_settings=OnlineSDFTSettings(
                replay_size=64,
                warmup_examples=64,
            ),
        )
        return [
            json.loads(line)
            for line in rollout_buffer.getvalue().splitlines()
        ]

    first = run_with_candidate_offset(RFT_CANDIDATE_RNG_OFFSET)
    second = run_with_candidate_offset(RFT_CANDIDATE_RNG_OFFSET + 1)
    assert [row["action"] for row in first] == [row["action"] for row in second]
    assert [row["feedback"] for row in first] == [
        row["feedback"] for row in second
    ]
    assert [row["behavior_probs"] for row in first] == [
        row["behavior_probs"] for row in second
    ]
    first_candidates = [
        row["rft_candidate_action"]
        for row in first
        if row["rft_candidate_action"] is not None
    ]
    second_candidates = [
        row["rft_candidate_action"]
        for row in second
        if row["rft_candidate_action"] is not None
    ]
    assert first_candidates and second_candidates
    assert first_candidates != second_candidates


def test_frozen_baselines_never_update_weights():
    for method in ("Base", "ICL", "RAG"):
        _, policy, _, _ = run_fast_method(method)
        assert not policy.updates
        assert not policy.reinforce_updates


def test_base_never_requests_a_teacher():
    calls, _, _, rollouts = run_fast_method("Base")
    assert "teacher" not in calls
    assert "teacher_view" not in calls
    assert all(row["teacher_probs"] is None for row in rollouts)


def test_icl_and_rag_never_request_a_teacher():
    for method in ("ICL", "RAG"):
        calls, _, _, rollouts = run_fast_method(method)
        assert "teacher" not in calls
        assert "teacher_view" not in calls
        assert all(row["teacher_probs"] is None for row in rollouts)
        assert all(row["teacher_rollout"] is None for row in rollouts)


def test_reinforce_receives_only_matured_factual_rewards():
    calls, policy, metrics, rollouts = run_fast_method("REINFORCE")
    assert calls[:4] == ["context", "act", "score", "execute"]
    assert "teacher" not in calls
    assert "teacher_view" not in calls
    assert policy.reinforce_updates
    assert not policy.updates
    assert metrics["method"] == "REINFORCE"
    applied = [
        row for row in rollouts if row["lesson_status"] == "feedback_applied"
    ]
    unflushed = [
        row
        for row in rollouts
        if row["lesson_status"] == "feedback_gradient_unflushed_at_horizon"
    ]
    released = applied + unflushed
    assert REINFORCE_BATCH_SIZE == 8
    assert REINFORCE_BASELINE_STEP == 0.0
    assert REINFORCE_ENTROPY_COEF == 1.0
    assert REINFORCE_MAX_GRAD_NORM == 1.0
    assert 0 < len(released) < STREAM_LENGTH
    batches = {}
    for row in applied:
        batches.setdefault(row["reinforce_update_index"], []).append(row)
    assert sorted(batches) == list(range(1, len(batches) + 1))
    assert all(len(rows) == REINFORCE_BATCH_SIZE for rows in batches.values())
    assert len(policy.reinforce_updates) == len(batches)
    assert all(
        len(batch) == REINFORCE_BATCH_SIZE
        and entropy_coef == pytest.approx(REINFORCE_ENTROPY_COEF)
        and max_grad_norm == pytest.approx(REINFORCE_MAX_GRAD_NORM)
        and all(
            isinstance(context, str)
            and action in range(len(ACTIONS))
            and np.isfinite(advantage)
            for context, action, advantage in batch
        )
        for batch, entropy_coef, max_grad_norm in policy.reinforce_updates
    )
    for update_index, (batch, _, _) in enumerate(
        policy.reinforce_updates,
        start=1,
    ):
        expected_rewards = [
            row["reinforce_training_reward"]
            for row in sorted(
                batches[update_index],
                key=lambda row: row["reinforce_batch_position"],
            )
        ]
        assert [advantage for _, _, advantage in batch] == expected_rewards
    assert all(
        sorted(row["reinforce_batch_position"] for row in rows)
        == list(range(1, REINFORCE_BATCH_SIZE + 1))
        for rows in batches.values()
    )
    assert len(unflushed) < REINFORCE_BATCH_SIZE
    assert sorted(row["reinforce_batch_position"] for row in unflushed) == list(
        range(1, len(unflushed) + 1)
    )
    assert all(
        row["feedback_released_at_minute"]
        >= row["feedback_available_at_minute"]
        for row in released
    )
    assert all(row["reinforce_update_index"] is None for row in unflushed)
    assert all(
        row["reinforce_batch_position"] is None
        and row["reinforce_update_index"] is None
        and row["reinforce_training_reward"] is None
        for row in rollouts
        if row["lesson_status"] in {"censored_no_update", "pending_after_horizon"}
    )
    assert all(
        row["reinforce_training_reward"]
        == REINFORCE_TRAINING_OUTCOME_REWARDS[row["feedback"]["outcome"]]
        for row in released
    )
    assert all(
        row["observed_feedback_reward"]
        == OBSERVED_OUTCOME_REWARDS[row["feedback"]["outcome"]]
        for row in released
    )
    assert any(
        row["reinforce_training_reward"] != row["observed_feedback_reward"]
        for row in released
    )
    assert all(
        row["feedback"]["reward"]
        in {-2.0, -1.0, 0.0, 0.25, 5.0}
        for row in released
    )


def test_trace_keeps_sampled_preference_out_of_teacher_evidence():
    _, _, _, rollouts = run_fast_method("Online-SDFT")
    assert all("gold_action_scoring_only" in row for row in rollouts)
    assert all(
        "gold_action_distribution_scoring_only" in row
        and "utility_optimal_action_scoring_only" in row
        for row in rollouts
    )
    assert all(row["step_regret"] >= 0.0 for row in rollouts)
    assert any(
        row["gold_action_scoring_only"]
        != row["utility_optimal_action_scoring_only"]
        for row in rollouts
    )
    assert all(
        row["correct_online"]
        == int(row["action"] == row["gold_action_scoring_only"])
        for row in rollouts
    )
    assert all(row["notification_title"] for row in rollouts)
    assert all(row["notification_body"] for row in rollouts)
    for row in rollouts:
        evidence = row["teacher_evidence"] or ""
        assert "gold_action" not in evidence
        assert "oracle" not in evidence.lower()
        assert "reward=" not in evidence


def test_icl_and_rag_prompts_contain_only_matured_past_records():
    reliability_by_status = {
        "memory_prompt_available": "reliable_singleton",
        "memory_retained_ambiguous": "ambiguous_digest_open",
        "memory_unlabeled": "censored_unknown",
    }
    for method in ("ICL", "RAG"):
        _, policy, _, rollouts = run_fast_method(method)
        assert policy.example_counts[0] == 0
        assert all(count <= step - 1 for step, count in enumerate(
            policy.example_counts, start=1
        ))
        released = [
            row
            for row in rollouts
            if row["lesson_status"] in reliability_by_status
        ]
        assert {
            row["lesson_status"] for row in released
        } == set(reliability_by_status)
        assert all(
            row["memory_evidence_reliability"]
            == reliability_by_status[row["lesson_status"]]
            for row in released
        )
        assert all(
            row["feedback_released_at_minute"]
            >= row["feedback_available_at_minute"]
            > row["decision_time_minute"]
            for row in released
        )
        assert all(
            row.get("memory_evidence_reliability") is None
            for row in rollouts
            if row["lesson_status"] == "pending_after_horizon"
        )


def test_shared_draws_stay_paired_while_only_lora_arms_send_update_batches():
    action_sequences = {}
    update_batches = {}
    for method in ("Base", "ICL", "RAG", "RFT", "Online-SDFT"):
        # Isolate the RNG-pairing invariant from the tuned Online-SDFT probe
        # schedule, which intentionally changes its behavior distribution.
        paired_sdft_settings = (
            OnlineSDFTSettings()
            if method == "Online-SDFT"
            else DEFAULT_SDFT_SETTINGS
        )
        _, policy, _, rollouts = run_fast_method(
            method,
            sdft_settings=paired_sdft_settings,
        )
        action_sequences[method] = [row["action"] for row in rollouts]
        update_batches[method] = policy.updates
    # The fake policy deliberately records updates without changing its fixed
    # probabilities, so every arm must retain the common action RNG draws.
    assert all(
        sequence == action_sequences["Base"]
        for sequence in action_sequences.values()
    )
    assert all(not update_batches[method] for method in ("Base", "ICL", "RAG"))
    assert update_batches["RFT"]
    assert update_batches["Online-SDFT"]


def test_method_registry_contains_the_six_reported_methods():
    assert METHODS == (
        "Base",
        "ICL",
        "RAG",
        "REINFORCE",
        "RFT",
        "Online-SDFT",
    )
