"""Method-level tests independent of environment reward implementation."""

from dataclasses import FrozenInstanceError
from inspect import getsource, signature
import re

import numpy as np
import pytest

import online_sdft.methods as methods_module
from online_sdft.config import (
    ACTION_CODES,
    ACTIONS,
    CATEGORIES,
    EXPLORATION_EPSILON,
    FEATURE_DIM,
    ICL_K,
    METHODS,
    MODEL_ID,
    PROMPT_TOKEN_BUDGET,
    REINFORCE_BATCH_SIZE,
    REINFORCE_BASELINE_STEP,
    REINFORCE_ENTROPY_COEF,
    REINFORCE_LR,
    REINFORCE_MAX_GRAD_NORM,
    REINFORCE_TRAINING_OUTCOME_REWARDS,
    RFT_CANDIDATE_COUNT,
    RFT_LR,
    RFT_SAMPLING_MODE,
    RFT_SAMPLING_TEMPERATURE,
)
from online_sdft.environment import StudentObservation
from online_sdft.methods import (
    AGENT_CLASSES,
    ASSESSMENT_FALLBACK,
    DEFAULT_REINFORCE_SETTINGS,
    DEFAULT_RFT_SETTINGS,
    DEFAULT_RFT_STUDENT_SETTINGS,
    DEFAULT_SDFT_SETTINGS,
    ICLAgent,
    LiquidLLMPolicy,
    OnlineSDFTSettings,
    OnlineSDFTAgent,
    RAGAgent,
    REINFORCEAgent,
    REINFORCESettings,
    RFTAgent,
    RFTSettings,
    causal_evidence_reliability,
    causal_route_support,
    create_agent,
    feedback_surface_propensity,
    mixed_context_similarity,
    notification_text_similarity,
)


class RecordingPolicy:
    def __init__(self):
        self.learning_rate = None
        self.updates = []
        self.update_weights = []
        self.support_updates = []
        self.support_update_weights = []
        self.base_calls = []
        self.reinforce_updates = []
        self.configured_settings = []
        self.teacher_calls = []

    def start_run(self, learning_rate):
        self.learning_rate = learning_rate
        self.updates.clear()
        self.update_weights.clear()
        self.support_updates.clear()
        self.support_update_weights.clear()
        self.base_calls.clear()
        self.reinforce_updates.clear()

    def probs(self, context, examples=None):
        del context, examples
        return np.array([0.2, 0.5, 0.3])

    def teacher_probs(self, observation, examples=None):
        self.teacher_calls.append((observation, examples))
        return np.array([0.2, 0.5, 0.3])

    def base_probs(self, context):
        self.base_calls.append(context)
        return np.array([0.4, 0.4, 0.2])

    def update(self, batch, sample_weights=None):
        self.updates.append(batch)
        self.update_weights.append(
            None if sample_weights is None else np.asarray(sample_weights).copy()
        )
        return 0.0

    def update_support(self, batch, sample_weights=None):
        self.support_updates.append(batch)
        self.support_update_weights.append(
            None if sample_weights is None else np.asarray(sample_weights).copy()
        )
        return 0.0

    def configure_online_sdft(self, settings):
        self.configured_settings.append(settings)

    def reinforce_update(self, batch, entropy_coef, max_grad_norm):
        self.reinforce_updates.append((batch, entropy_coef, max_grad_norm))
        return 0.0


class CapturingTokenizer:
    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = [dict(message) for message in messages]
        self.kwargs = dict(kwargs)
        return "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
        )


def prompt_policy(
    *,
    prompt_style="compact",
):
    policy = object.__new__(LiquidLLMPolicy)
    policy.tokenizer = CapturingTokenizer()
    policy.prompt_style = prompt_style
    return policy


def history_example(
    context,
    selection,
    *,
    executed_action="LATER",
    outcome="OPENED_DIGEST",
    delay_minutes=120,
):
    return {
        "context": context,
        "executed_action": executed_action,
        "eventual_user_action": outcome,
        "observed_user_selection": selection,
        "delay_minutes": delay_minutes,
    }


def visible_observation(
    name,
    category=0,
    hour=12.0,
    regime=0,
    importance=0.5,
):
    angle = 2 * np.pi * hour / 24
    category_features = np.eye(len(CATEGORIES))[category]
    features = np.concatenate(
        [
            category_features,
            np.array(
                [
                    importance,
                    np.sin(angle),
                    np.cos(angle),
                    regime / 2,
                    1.0,
                ]
            ),
        ]
    )
    return StudentObservation(name, features)


def test_method_registry_matches_reported_benchmark():
    assert tuple(AGENT_CLASSES) == METHODS
    assert len(AGENT_CLASSES) == 6


def test_factory_builds_capacity_matched_rft_agent():
    agent = create_agent("RFT", RecordingPolicy())
    assert isinstance(agent, RFTAgent)
    assert agent.rft_settings is DEFAULT_RFT_SETTINGS
    assert agent.settings is DEFAULT_RFT_SETTINGS.student_settings
    assert agent.settings is DEFAULT_RFT_STUDENT_SETTINGS
    assert agent.settings.learning_rate == RFT_LR == 7e-4
    assert agent.settings.lora_target_modules == (
        DEFAULT_SDFT_SETTINGS.lora_target_modules
    )
    assert agent.uses_teacher is True


def test_default_sdft_settings_are_the_verified_sharp_sampling_winner():
    settings = DEFAULT_SDFT_SETTINGS

    assert settings.learning_rate == pytest.approx(1e-3)
    assert settings.lora_rank == 4
    assert settings.lora_alpha == 8
    assert settings.lora_target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "self_attn.out_proj",
    )
    assert settings.lora_layers_to_transform == (2, 4, 6, 8, 10, 12)
    assert settings.target_mode == "causal_fusion"
    assert (
        settings.reliable_teacher_weight,
        settings.reliable_decision_weight,
        settings.reliable_behavior_weight,
    ) == pytest.approx((0.05, 0.05, 0.90))
    assert (
        settings.ambiguous_teacher_weight,
        settings.ambiguous_decision_weight,
        settings.ambiguous_behavior_weight,
    ) == pytest.approx((0.0, 1.0, 0.0))
    assert settings.replay_size == 64
    assert settings.batch_size == 8
    assert settings.update_steps == 2
    assert settings.warmup_examples == 4
    assert settings.replay_strategy == "selection_balanced"
    assert settings.replay_recency_half_life == pytest.approx(32.0)
    assert settings.force_newest_every_step is True
    assert settings.behavior_mode == "uncertainty_interrupt_probe"
    assert settings.behavior_epsilon == pytest.approx(0.02)
    assert settings.behavior_epsilon_half_life is None
    assert settings.interrupt_probe_mix == pytest.approx(0.15)
    assert settings.interrupt_probe_half_life == pytest.approx(80.0)
    assert settings.interrupt_probe_max_confidence == pytest.approx(0.60)
    assert settings.exploration_taper_start_step == 160
    assert settings.exploration_taper_half_life == pytest.approx(5.0)


def test_default_rft_settings_are_frozen_resolved_and_capacity_matched():
    assert RFT_CANDIDATE_COUNT == 1
    assert RFT_SAMPLING_TEMPERATURE == 8.0
    assert RFT_SAMPLING_MODE == "categorical"
    assert DEFAULT_RFT_SETTINGS.to_dict() == {
        "student_settings": DEFAULT_RFT_STUDENT_SETTINGS.to_dict(),
        "candidate_count": RFT_CANDIDATE_COUNT,
        "sampling_temperature": RFT_SAMPLING_TEMPERATURE,
        "sampling_mode": RFT_SAMPLING_MODE,
    }
    assert DEFAULT_RFT_STUDENT_SETTINGS is not DEFAULT_SDFT_SETTINGS
    assert DEFAULT_RFT_STUDENT_SETTINGS == OnlineSDFTSettings(
        learning_rate=RFT_LR,
    )
    assert DEFAULT_RFT_STUDENT_SETTINGS.replay_size == 32
    assert DEFAULT_RFT_STUDENT_SETTINGS.replay_recency_half_life is None
    assert DEFAULT_RFT_STUDENT_SETTINGS.behavior_mode == "epsilon_greedy"
    assert DEFAULT_RFT_STUDENT_SETTINGS.behavior_epsilon == pytest.approx(0.06)
    assert DEFAULT_RFT_STUDENT_SETTINGS.behavior_epsilon_half_life is None
    assert DEFAULT_RFT_STUDENT_SETTINGS.exploration_taper_start_step is None
    assert DEFAULT_RFT_STUDENT_SETTINGS.exploration_taper_half_life is None
    assert DEFAULT_RFT_STUDENT_SETTINGS.interrupt_probe_mix == 0.0
    assert DEFAULT_RFT_STUDENT_SETTINGS.interrupt_probe_half_life is None
    assert DEFAULT_RFT_STUDENT_SETTINGS.interrupt_probe_max_confidence == 1.0
    assert (
        DEFAULT_RFT_STUDENT_SETTINGS.lora_target_modules
        == DEFAULT_SDFT_SETTINGS.lora_target_modules
    )
    assert (
        DEFAULT_RFT_STUDENT_SETTINGS.lora_layers_to_transform
        == DEFAULT_SDFT_SETTINGS.lora_layers_to_transform
    )
    with pytest.raises(FrozenInstanceError):
        DEFAULT_RFT_SETTINGS.sampling_temperature = 2.0


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"student_settings": {}}, TypeError, "student settings"),
        ({"candidate_count": 0}, ValueError, "candidate_count=1"),
        ({"candidate_count": 2}, ValueError, "candidate_count=1"),
        ({"candidate_count": True}, ValueError, "candidate_count=1"),
        ({"sampling_temperature": 0.0}, ValueError, "positive and finite"),
        ({"sampling_temperature": -1.0}, ValueError, "positive and finite"),
        ({"sampling_temperature": np.inf}, ValueError, "positive and finite"),
        ({"sampling_temperature": np.nan}, ValueError, "positive and finite"),
        ({"sampling_temperature": True}, ValueError, "positive and finite"),
        ({"sampling_mode": "greedy"}, ValueError, "categorical sampling"),
    ],
)
def test_rft_settings_reject_invalid_or_non_categorical_proposals(
    overrides,
    error,
    match,
):
    with pytest.raises(error, match=match):
        RFTSettings(**overrides)


def test_factory_keeps_rft_and_online_sdft_settings_isolated():
    rft_student = OnlineSDFTSettings(learning_rate=7.5e-4)
    rft_settings = RFTSettings(
        student_settings=rft_student,
        sampling_temperature=4.0,
    )
    rft = create_agent(
        "RFT",
        RecordingPolicy(),
        sdft_settings=DEFAULT_SDFT_SETTINGS,
        rft_settings=rft_settings,
    )
    sdft = create_agent(
        "Online-SDFT",
        RecordingPolicy(),
        sdft_settings=DEFAULT_SDFT_SETTINGS,
        rft_settings=rft_settings,
    )

    assert rft.rft_settings is rft_settings
    assert rft.settings is rft_student
    assert sdft.settings is DEFAULT_SDFT_SETTINGS
    assert not hasattr(sdft, "rft_settings")


@pytest.mark.parametrize(
    ("action", "feedback", "expected"),
    [
        (0, {"outcome": "OPENED_IMMEDIATELY"}, [1, 0, 0]),
        (0, {"outcome": "OPENED_AFTER_DELAY"}, [0, 1, 0]),
        (0, {"outcome": "DELETED_NOTIFICATION"}, [0, 0, 1]),
        (1, {"outcome": "OPENED_DIGEST"}, [1, 1, 0]),
        (1, {"outcome": "DELETED_FROM_DIGEST"}, [0, 0, 1]),
        (2, {"outcome": "NO_OBSERVABLE_SELECTION"}, [1, 1, 1]),
    ],
)
def test_causal_route_support_encodes_only_observable_trajectory(
    action,
    feedback,
    expected,
):
    assert causal_route_support(action, feedback).tolist() == [
        bool(value) for value in expected
    ]


@pytest.mark.parametrize(
    ("action", "feedback", "candidate", "expected_reason"),
    [
        (
            0,
            {
                "outcome": "OPENED_IMMEDIATELY",
                "observed_user_selection": "INTERRUPT",
            },
            0,
            "accepted",
        ),
        (
            0,
            {
                "outcome": "OPENED_IMMEDIATELY",
                "observed_user_selection": "INTERRUPT",
            },
            1,
            "teacher_mismatch",
        ),
        (
            0,
            {
                "outcome": "OPENED_AFTER_DELAY",
                "observed_user_selection": "LATER",
            },
            1,
            "accepted",
        ),
        (
            0,
            {
                "outcome": "DELETED_NOTIFICATION",
                "observed_user_selection": "ARCHIVE",
            },
            2,
            "accepted",
        ),
        (
            1,
            {
                "outcome": "OPENED_DIGEST",
                "observed_user_selection": "LATER",
            },
            1,
            "ambiguous_unverified",
        ),
        (
            1,
            {
                "outcome": "DELETED_FROM_DIGEST",
                "observed_user_selection": "ARCHIVE",
            },
            2,
            "accepted",
        ),
        (
            2,
            {
                "outcome": "NO_OBSERVABLE_SELECTION",
                "observed_user_selection": "UNKNOWN",
            },
            None,
            "censored_unknown",
        ),
    ],
)
def test_rft_accepts_only_teacher_samples_verified_by_singleton_support(
    action,
    feedback,
    candidate,
    expected_reason,
):
    assert RFTAgent.rft_reason(
        action,
        feedback,
        candidate,
    ) == expected_reason


def test_rft_filter_ignores_reward_match_status_and_hidden_evaluator_fields():
    feedback = {
        "action_taken": "INTERRUPT",
        "outcome": "OPENED_AFTER_DELAY",
        "observed_user_selection": "LATER",
        "reward": -999.0,
        "match_status": "MISS",
        "gold_action": "ARCHIVE",
        "oracle_utility": 1_000.0,
    }
    expected = RFTAgent.rft_reason(0, feedback, 1)
    changed = {
        **feedback,
        "reward": 999.0,
        "match_status": "MATCH",
        "gold_action": "INTERRUPT",
        "oracle_utility": -1_000.0,
    }
    assert expected == "accepted"
    assert RFTAgent.rft_reason(0, changed, 1) == expected


def test_rft_reuses_online_sdft_lora_architecture_with_tuned_learning_rate():
    rft_policy = RecordingPolicy()
    sdft_policy = RecordingPolicy()
    rft = RFTAgent(rft_policy)
    sdft = OnlineSDFTAgent(sdft_policy)
    observation = visible_observation("same visible request")

    assert rft.settings is DEFAULT_RFT_STUDENT_SETTINGS
    assert sdft.settings is DEFAULT_SDFT_SETTINGS
    assert rft.settings.lora_target_modules == sdft.settings.lora_target_modules
    assert rft.settings.lora_layers_to_transform == (
        sdft.settings.lora_layers_to_transform
    )
    assert rft.settings.lora_rank == sdft.settings.lora_rank
    assert rft_policy.learning_rate == RFT_LR
    assert sdft_policy.learning_rate == DEFAULT_SDFT_SETTINGS.learning_rate
    assert rft.action_probs(observation) == pytest.approx(
        sdft.action_probs(observation)
    )


def test_rft_uses_explicit_sample_not_teacher_argmax_for_one_hot_target():
    settings = OnlineSDFTSettings(
        learning_rate=1.0,
        replay_size=4,
        batch_size=1,
        update_steps=1,
        warmup_examples=1,
    )
    policy = RecordingPolicy()
    agent = RFTAgent(
        policy,
        rft_settings=RFTSettings(student_settings=settings),
    )
    observation = visible_observation("verified teacher sample")
    before = agent.action_probs(observation)

    agent.observe(
        observation,
        action=0,
        teacher_distribution=np.array([0.05, 0.05, 0.90]),
        teacher_action=2,
        feedback={
            "action_taken": "INTERRUPT",
            "outcome": "OPENED_IMMEDIATELY",
            "observed_user_selection": "INTERRUPT",
        },
        rng=np.random.default_rng(0),
        decision_distribution=before,
        candidate_action=0,
    )

    assert agent.last_rft_candidate_action == 0
    assert agent.last_rft_accepted is True
    assert agent.last_rft_reason == "accepted"
    assert agent.replay[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert agent.online_update_count == 1
    assert len(policy.updates) == 1
    assert policy.updates[0][0][0] == observation.text
    assert policy.updates[0][0][1] == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("action", "feedback", "candidate", "accepted", "reason"),
    [
        (
            1,
            {
                "action_taken": "LATER",
                "outcome": "OPENED_DIGEST",
                "observed_user_selection": "LATER",
            },
            1,
            False,
            "ambiguous_unverified",
        ),
        (
            0,
            {
                "action_taken": "INTERRUPT",
                "outcome": "OPENED_AFTER_DELAY",
                "observed_user_selection": "LATER",
            },
            0,
            False,
            "teacher_mismatch",
        ),
        (
            2,
            {
                "action_taken": "ARCHIVE",
                "outcome": "NO_OBSERVABLE_SELECTION",
                "observed_user_selection": "UNKNOWN",
            },
            None,
            None,
            "censored_unknown",
        ),
    ],
)
def test_rft_rejected_and_censored_rows_never_enter_replay_or_update_lora(
    action,
    feedback,
    candidate,
    accepted,
    reason,
):
    agent = RFTAgent(RecordingPolicy())
    observation = visible_observation(f"rejected-{reason}")
    decision = agent.action_probs(observation)

    teacher_distribution = (
        None if candidate is None else np.array([0.6, 0.3, 0.1])
    )
    teacher_action = None if candidate is None else 0
    agent.observe(
        observation,
        action=action,
        teacher_distribution=teacher_distribution,
        teacher_action=teacher_action,
        feedback=feedback,
        rng=np.random.default_rng(0),
        decision_distribution=decision,
        candidate_action=candidate,
    )

    assert agent.last_rft_accepted is accepted
    assert agent.last_rft_reason == reason
    assert not agent.replay
    assert not agent.policy.updates
    assert agent.online_update_count == 0


def test_online_sdft_keeps_full_soft_teacher_distribution():
    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(
        policy,
        settings=OnlineSDFTSettings(
            target_mode="teacher_only",
            reliable_teacher_weight=1.0,
            reliable_decision_weight=0.0,
            reliable_behavior_weight=0.0,
            ambiguous_teacher_weight=1.0,
            ambiguous_decision_weight=0.0,
            ambiguous_behavior_weight=0.0,
        ),
    )
    observation = StudentObservation("visible", np.ones(3))
    distribution = np.array([0.1, 0.7, 0.2])
    from online_sdft.environment import TeacherObservation
    agent.observe(
        observation,
        action=0,
        teacher_distribution=distribution,
        teacher_action=1,
        feedback={
            "outcome": "factual",
            "observed_user_selection": "LATER",
        },
        rng=np.random.default_rng(0),
        teacher_observation=TeacherObservation(
            context="visible",
            evidence="executed_route=LATER; observed_event=OPENED_DIGEST",
        ),
    )
    target = agent.replay[0][1]
    assert np.isclose(target.sum(), 1.0)
    assert int(np.argmax(target)) == 1
    assert np.allclose(target, distribution)


def test_online_sdft_snips_replay_uses_frozen_feedback_propensity_for_all_reliability():
    settings = OnlineSDFTSettings(
        replay_size=4,
        batch_size=1,
        warmup_examples=1,
        update_steps=1,
        propensity_weight_mode="feedback_surface_snips",
    )
    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(policy, settings=settings)
    behavior = np.array([0.10, 0.20, 0.70])
    rows = (
        (
            "reliable",
            0,
            {
                "action_taken": "INTERRUPT",
                "outcome": "OPENED_IMMEDIATELY",
                "observed_user_selection": "INTERRUPT",
            },
        ),
        (
            "ambiguous",
            1,
            {
                "action_taken": "LATER",
                "outcome": "OPENED_DIGEST",
                "observed_user_selection": "LATER",
            },
        ),
    )
    for index, (text, action, feedback) in enumerate(rows):
        observation = visible_observation(text)
        agent.observe(
            observation,
            action=action,
            teacher_distribution=np.array([0.5, 0.3, 0.2]),
            teacher_action=0,
            feedback=feedback,
            rng=np.random.default_rng(index),
            decision_distribution=np.array([0.2, 0.5, 0.3]),
            behavior_distribution=behavior.copy(),
        )

    assert [row[2] for row in agent.replay] == ["INTERRUPT", "AMBIGUOUS"]
    assert [row[5] for row in agent.replay] == pytest.approx([0.30, 0.30])
    assert len(policy.update_weights) == 2
    assert all(
        weights == pytest.approx([10.0 / 3.0])
        for weights in policy.update_weights
    )

    clipped_row = (*agent.replay[0][:5], 0.05)
    agent.update_from_replay([clipped_row])
    assert policy.update_weights[-1] == pytest.approx([4.0])

    replay_size = len(agent.replay)
    agent.observe(
        visible_observation("censored"),
        action=2,
        teacher_distribution=None,
        teacher_action=None,
        feedback={
            "action_taken": "ARCHIVE",
            "outcome": "NO_OBSERVABLE_SELECTION",
            "observed_user_selection": "UNKNOWN",
        },
        rng=np.random.default_rng(2),
        decision_distribution=np.array([0.2, 0.5, 0.3]),
        behavior_distribution=np.array([0.0, 0.0, 1.0]),
    )
    assert len(agent.replay) == replay_size
    assert len(policy.update_weights) == 3


def test_online_sdft_snips_requires_decision_time_behavior_for_informative_replay():
    agent = OnlineSDFTAgent(
        RecordingPolicy(),
        settings=OnlineSDFTSettings(
            replay_size=1,
            batch_size=1,
            warmup_examples=1,
            update_steps=1,
            propensity_weight_mode="feedback_surface_snips",
        ),
    )

    with pytest.raises(ValueError, match="decision-time behavior"):
        agent.observe(
            visible_observation("missing propensity"),
            action=0,
            teacher_distribution=np.array([0.7, 0.2, 0.1]),
            teacher_action=0,
            feedback={
                "action_taken": "INTERRUPT",
                "outcome": "OPENED_IMMEDIATELY",
                "observed_user_selection": "INTERRUPT",
            },
            rng=np.random.default_rng(0),
            decision_distribution=np.array([0.2, 0.5, 0.3]),
        )


def test_online_sdft_settings_propagate_to_agent_and_policy():
    class ConfigurablePolicy(RecordingPolicy):
        def configure_online_sdft(self, settings):
            self.configured_settings = settings

    settings = OnlineSDFTSettings(
        learning_rate=2e-4,
        replay_size=9,
        replay_prompt_examples=2,
        batch_size=3,
        update_steps=2,
        warmup_examples=4,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target_modules=("q_proj", "v_proj"),
        lora_layers_to_transform=(8, 10, 12),
        optimizer_weight_decay=0.01,
        max_grad_norm=0.75,
        ambiguous_replay_group_weight=0.10,
        teacher_temperature=0.7,
        reasoning_tokens=24,
        target_mode="causal_fusion",
        reliable_teacher_weight=0.15,
        reliable_decision_weight=0.10,
        reliable_behavior_weight=0.75,
        ambiguous_teacher_weight=0.30,
        ambiguous_decision_weight=0.20,
        ambiguous_behavior_weight=0.50,
        ambiguous_projection="causal_support",
        replay_strategy="selection_balanced",
    )
    policy = ConfigurablePolicy()
    agent = OnlineSDFTAgent(policy, settings=settings)

    assert policy.learning_rate == settings.learning_rate
    assert policy.configured_settings is settings
    assert agent.settings is settings
    assert agent.replay_size == settings.replay_size
    assert agent.replay_prompt_examples == settings.replay_prompt_examples
    assert agent.online_batch_size == settings.batch_size
    assert agent.update_steps == settings.update_steps
    assert agent.warmup_examples == settings.warmup_examples
    assert agent.replay_strategy == settings.replay_strategy
    assert agent.behavior_epsilon == settings.behavior_epsilon
    assert (
        agent.behavior_epsilon_half_life
        == settings.behavior_epsilon_half_life
    )
    assert (
        agent.exploration_taper_start_step
        == settings.exploration_taper_start_step
    )
    assert (
        agent.exploration_taper_half_life
        == settings.exploration_taper_half_life
    )
    assert (
        agent.archive_policy_min_feedback
        == settings.archive_policy_min_feedback
    )
    assert agent.interrupt_probe_mix == settings.interrupt_probe_mix
    assert (
        agent.interrupt_probe_half_life
        == settings.interrupt_probe_half_life
    )
    assert (
        agent.interrupt_probe_max_confidence
        == settings.interrupt_probe_max_confidence
    )
    assert settings.to_dict() == {
        "learning_rate": 2e-4,
        "replay_size": 9,
        "replay_prompt_examples": 2,
        "batch_size": 3,
        "update_steps": 2,
        "warmup_examples": 4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_target_modules": ("q_proj", "v_proj"),
        "lora_layers_to_transform": (8, 10, 12),
        "lora_a_learning_rate_scale": 1.0,
        "lm_head_lora_a_learning_rate_scale": None,
        "optimizer_weight_decay": 0.01,
        "optimizer_beta1": 0.9,
        "max_grad_norm": 0.75,
        "lm_head_learning_rate": None,
        "ambiguous_replay_group_weight": 0.10,
        "teacher_temperature": 0.7,
        "reasoning_tokens": 24,
        "target_mode": "causal_fusion",
        "reliable_teacher_weight": 0.15,
        "reliable_decision_weight": 0.10,
        "reliable_behavior_weight": 0.75,
        "ambiguous_teacher_weight": 0.30,
        "ambiguous_decision_weight": 0.20,
        "ambiguous_behavior_weight": 0.50,
        "ambiguous_projection": "causal_support",
        "replay_strategy": "selection_balanced",
        "replay_recency_half_life": None,
        "ambiguous_update_mode": "immediate",
        "force_newest_every_step": True,
        "base_kl_weight": 0.0,
        "behavior_mode": "epsilon_greedy",
        "behavior_epsilon": EXPLORATION_EPSILON,
        "behavior_epsilon_half_life": None,
        "exploration_taper_start_step": None,
        "exploration_taper_half_life": None,
        "archive_probe_mix": 0.0,
        "archive_policy_min_feedback": 0.0,
        "interrupt_probe_mix": 0.0,
        "interrupt_probe_half_life": None,
        "interrupt_probe_max_confidence": 1.0,
        "propensity_weight_mode": "none",
        "propensity_weight_cap": 4.0,
    }


def test_feedback_surface_snips_settings_and_propensity_are_explicit():
    settings = OnlineSDFTSettings(
        propensity_weight_mode="feedback_surface_snips",
        propensity_weight_cap=2.5,
    )

    assert settings.propensity_weight_mode == "feedback_surface_snips"
    assert settings.propensity_weight_cap == pytest.approx(2.5)
    assert feedback_surface_propensity(np.array([0.10, 0.20, 0.70])) == (
        pytest.approx(0.30)
    )


@pytest.mark.parametrize(
    "behavior",
    [
        np.array([0.2, 0.8]),
        np.array([0.2, -0.1, 0.9]),
        np.array([0.2, np.nan, 0.8]),
        np.array([0.2, 0.2, 0.2]),
        np.array([0.0, 0.0, 1.0]),
    ],
)
def test_feedback_surface_propensity_rejects_invalid_behavior(behavior):
    with pytest.raises(ValueError, match="behavior|propensity"):
        feedback_surface_propensity(behavior)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"learning_rate": 0.0}, "positive"),
        ({"batch_size": 5, "replay_size": 4}, "cannot exceed"),
        (
            {
                "replay_prompt_examples": 5,
                "replay_size": 4,
                "batch_size": 4,
            },
            "prompt examples",
        ),
        ({"replay_prompt_examples": -1}, "prompt examples"),
        (
            {"warmup_examples": 5, "replay_size": 4, "batch_size": 4},
            "warmup cannot exceed",
        ),
        ({"reasoning_tokens": -1}, "cannot be negative"),
        ({"lora_rank": 0}, "rank and alpha"),
        ({"lora_alpha": True}, "rank and alpha"),
        ({"lora_dropout": 1.0}, "dropout"),
        ({"lora_target_modules": ()}, "target modules"),
        ({"lora_target_modules": ("q_proj", "q_proj")}, "unique"),
        ({"lora_layers_to_transform": ()}, "cannot be empty"),
        ({"lora_layers_to_transform": (-1,)}, "layer indices"),
        ({"lora_layers_to_transform": (8, 8)}, "layer indices"),
        ({"lora_a_learning_rate_scale": True}, "A learning-rate scale"),
        ({"lora_a_learning_rate_scale": -0.1}, "A learning-rate scale"),
        ({"lora_a_learning_rate_scale": 1.1}, "A learning-rate scale"),
        ({"lora_a_learning_rate_scale": float("nan")}, "A learning-rate scale"),
        (
            {"lm_head_lora_a_learning_rate_scale": True},
            "LM-head LoRA A learning-rate scale",
        ),
        (
            {"lm_head_lora_a_learning_rate_scale": -0.1},
            "LM-head LoRA A learning-rate scale",
        ),
        (
            {"lm_head_lora_a_learning_rate_scale": 1.1},
            "LM-head LoRA A learning-rate scale",
        ),
        (
            {"lm_head_lora_a_learning_rate_scale": float("nan")},
            "LM-head LoRA A learning-rate scale",
        ),
        (
            {"lm_head_lora_a_learning_rate_scale": 1.0},
            "requires an LM-head",
        ),
        ({"optimizer_weight_decay": -0.1}, "weight decay"),
        ({"optimizer_beta1": True}, "beta1"),
        ({"optimizer_beta1": -0.1}, "beta1"),
        ({"optimizer_beta1": 1.0}, "beta1"),
        ({"optimizer_beta1": float("nan")}, "beta1"),
        ({"max_grad_norm": 0.0}, "gradient norm"),
        ({"lm_head_learning_rate": True}, "LM-head learning rate"),
        ({"lm_head_learning_rate": 0.0}, "LM-head learning rate"),
        ({"lm_head_learning_rate": float("nan")}, "LM-head learning rate"),
        ({"lm_head_learning_rate": float("inf")}, "LM-head learning rate"),
        ({"lm_head_learning_rate": 1e-3}, "requires an LM-head"),
        ({"ambiguous_replay_group_weight": 0.0}, "replay group weight"),
        ({"target_mode": "hard_label"}, "target mode"),
        (
            {"target_mode": "support_likelihood"},
            "zero unused fusion weights",
        ),
        ({"ambiguous_projection": "gold_route"}, "projection mode"),
        ({"replay_strategy": "latest_only"}, "replay strategy"),
        ({"replay_recency_half_life": True}, "finite and positive"),
        ({"replay_recency_half_life": 0.0}, "finite and positive"),
        ({"replay_recency_half_life": -1.0}, "finite and positive"),
        ({"replay_recency_half_life": float("nan")}, "finite and positive"),
        ({"replay_recency_half_life": float("inf")}, "finite and positive"),
        ({"replay_recency_half_life": "8"}, "finite and positive"),
        (
            {
                "replay_strategy": "uniform",
                "replay_recency_half_life": 8.0,
            },
            "requires selection-balanced",
        ),
        ({"ambiguous_update_mode": "shock"}, "ambiguous update mode"),
        ({"force_newest_every_step": 1}, "must be boolean"),
        ({"base_kl_weight": -0.1}, "KL weight"),
        ({"behavior_mode": "oracle"}, "behavior mode"),
        ({"behavior_epsilon": True}, "behavior epsilon"),
        ({"behavior_epsilon": "0.01"}, "behavior epsilon"),
        ({"behavior_epsilon": -0.01}, "behavior epsilon"),
        ({"behavior_epsilon": 1.01}, "behavior epsilon"),
        ({"behavior_epsilon": float("nan")}, "behavior epsilon"),
        ({"behavior_epsilon": float("inf")}, "behavior epsilon"),
        ({"behavior_epsilon_half_life": True}, "epsilon half-life"),
        ({"behavior_epsilon_half_life": "10"}, "epsilon half-life"),
        ({"behavior_epsilon_half_life": 0.0}, "epsilon half-life"),
        ({"behavior_epsilon_half_life": -1.0}, "epsilon half-life"),
        (
            {"behavior_epsilon_half_life": float("nan")},
            "epsilon half-life",
        ),
        (
            {"behavior_epsilon_half_life": float("inf")},
            "epsilon half-life",
        ),
        ({"exploration_taper_start_step": True}, "taper start step"),
        ({"exploration_taper_start_step": "160"}, "taper start step"),
        ({"exploration_taper_start_step": 1.5}, "taper start step"),
        ({"exploration_taper_start_step": 0}, "taper start step"),
        ({"exploration_taper_start_step": -1}, "taper start step"),
        ({"exploration_taper_half_life": True}, "taper half-life"),
        ({"exploration_taper_half_life": 0.0}, "taper half-life"),
        ({"exploration_taper_half_life": -1.0}, "taper half-life"),
        (
            {"exploration_taper_half_life": float("nan")},
            "taper half-life",
        ),
        (
            {"exploration_taper_half_life": float("inf")},
            "taper half-life",
        ),
        (
            {"exploration_taper_half_life": 20.0},
            "configured together",
        ),
        (
            {"exploration_taper_start_step": 160},
            "configured together",
        ),
        (
            {
                "exploration_taper_start_step": 160,
                "exploration_taper_half_life": 20.0,
                "behavior_mode": "policy_sampling",
            },
            "epsilon/probe behavior mode",
        ),
        ({"archive_probe_mix": True}, "archive probe mix"),
        ({"archive_probe_mix": -0.1}, "archive probe mix"),
        ({"archive_probe_mix": 1.1}, "archive probe mix"),
        ({"archive_probe_mix": float("nan")}, "archive probe mix"),
        ({"archive_policy_min_feedback": True}, "minimum feedback"),
        ({"archive_policy_min_feedback": "0.2"}, "minimum feedback"),
        ({"archive_policy_min_feedback": -0.1}, "minimum feedback"),
        ({"archive_policy_min_feedback": 1.1}, "minimum feedback"),
        ({"archive_policy_min_feedback": float("nan")}, "minimum feedback"),
        ({"archive_policy_min_feedback": float("inf")}, "minimum feedback"),
        ({"interrupt_probe_mix": True}, "interrupt probe mix"),
        ({"interrupt_probe_mix": -0.1}, "interrupt probe mix"),
        ({"interrupt_probe_mix": 1.0}, "interrupt probe mix"),
        ({"interrupt_probe_mix": float("nan")}, "interrupt probe mix"),
        ({"interrupt_probe_half_life": True}, "probe half-life"),
        ({"interrupt_probe_half_life": 0.0}, "probe half-life"),
        ({"interrupt_probe_half_life": float("inf")}, "probe half-life"),
        (
            {"interrupt_probe_max_confidence": True},
            "maximum confidence",
        ),
        (
            {"interrupt_probe_max_confidence": 0.3},
            "maximum confidence",
        ),
        (
            {"interrupt_probe_max_confidence": 1.1},
            "maximum confidence",
        ),
        ({"propensity_weight_mode": "ips"}, "propensity-weight mode"),
        ({"propensity_weight_cap": True}, "propensity-weight cap"),
        ({"propensity_weight_cap": 0.9}, "propensity-weight cap"),
        ({"propensity_weight_cap": float("nan")}, "propensity-weight cap"),
        ({"propensity_weight_cap": float("inf")}, "propensity-weight cap"),
        ({"propensity_weight_cap": 2.0}, "custom propensity-weight cap"),
        (
            {"behavior_mode": "archive_uniform_probe"},
            "positive mix",
        ),
        (
            {"behavior_mode": "epsilon_greedy", "archive_probe_mix": 0.1},
            "requires archive-uniform",
        ),
        (
            {"behavior_mode": "archive_policy_feedback_floor"},
            "requires a positive",
        ),
        (
            {
                "behavior_mode": "epsilon_greedy",
                "archive_policy_min_feedback": 0.2,
            },
            "requires feedback-floor",
        ),
        (
            {"behavior_mode": "uncertainty_interrupt_probe"},
            "requires a positive mix",
        ),
        (
            {
                "behavior_mode": "uncertainty_interrupt_probe",
                "interrupt_probe_mix": 0.2,
            },
            "requires a half-life",
        ),
        (
            {
                "behavior_mode": "epsilon_greedy",
                "interrupt_probe_mix": 0.2,
                "interrupt_probe_half_life": 20.0,
            },
            "require uncertainty interrupt probing",
        ),
        (
            {
                "target_mode": "support_likelihood",
                "reliable_teacher_weight": 0.0,
                "reliable_decision_weight": 0.0,
                "reliable_behavior_weight": 0.0,
                "ambiguous_teacher_weight": 0.0,
                "ambiguous_decision_weight": 0.0,
                "ambiguous_behavior_weight": 0.0,
                "base_kl_weight": 0.1,
            },
            "incompatible",
        ),
        (
            {"base_kl_weight": 0.1, "replay_prompt_examples": 1},
            "pure parametric",
        ),
        (
            {
                "target_mode": "causal_fusion",
                "reliable_teacher_weight": 0.5,
                "reliable_decision_weight": 0.5,
                "reliable_behavior_weight": 0.5,
            },
            "sum to one",
        ),
        (
            {
                "target_mode": "causal_fusion",
                "ambiguous_teacher_weight": 0.5,
                "ambiguous_decision_weight": 0.5,
                "ambiguous_behavior_weight": 0.5,
            },
            "sum to one",
        ),
        (
            {
                "target_mode": "causal_fusion",
                "reliable_teacher_weight": 0.3,
                "reliable_decision_weight": 0.3,
                "reliable_behavior_weight": 0.4,
                "ambiguous_teacher_weight": 0.2,
                "ambiguous_decision_weight": 0.2,
                "ambiguous_behavior_weight": 0.6,
            },
            "cannot receive less causal weight",
        ),
        (
            {
                "target_mode": "teacher_only",
                "reliable_teacher_weight": 1.0,
                "reliable_decision_weight": 0.0,
                "reliable_behavior_weight": 0.0,
                "ambiguous_teacher_weight": 0.8,
                "ambiguous_decision_weight": 0.2,
                "ambiguous_behavior_weight": 0.0,
            },
            "canonical weights",
        ),
    ],
)
def test_online_sdft_settings_reject_incoherent_arms(overrides, message):
    with pytest.raises(ValueError, match=message):
        OnlineSDFTSettings(**overrides)


def test_online_sdft_settings_normalize_single_lora_target_and_layer():
    settings = OnlineSDFTSettings(
        lora_target_modules="lm_head",
        lora_layers_to_transform=12,
    )

    assert settings.lora_target_modules == ("lm_head",)
    assert settings.lora_layers_to_transform == (12,)


@pytest.mark.parametrize(
    (
        "action",
        "feedback",
        "expected_reliability",
        "expected_weights",
        "expected_behavior",
        "expected_target",
    ),
    [
        (
            0,
            {
                "action_taken": "INTERRUPT",
                "outcome": "OPENED_IMMEDIATELY",
                "observed_user_selection": "INTERRUPT",
            },
            "reliable_singleton",
            (0.10, 0.10, 0.80),
            [1.0, 0.0, 0.0],
            [0.88, 0.06, 0.06],
        ),
        (
            0,
            {
                "action_taken": "INTERRUPT",
                "outcome": "OPENED_AFTER_DELAY",
                "observed_user_selection": "LATER",
            },
            "reliable_singleton",
            (0.10, 0.10, 0.80),
            [0.0, 1.0, 0.0],
            [0.08, 0.86, 0.06],
        ),
        (
            0,
            {
                "action_taken": "INTERRUPT",
                "outcome": "DELETED_NOTIFICATION",
                "observed_user_selection": "ARCHIVE",
            },
            "reliable_singleton",
            (0.10, 0.10, 0.80),
            [0.0, 0.0, 1.0],
            [0.08, 0.06, 0.86],
        ),
        (
            1,
            {
                "action_taken": "LATER",
                "outcome": "OPENED_DIGEST",
                "observed_user_selection": "LATER",
            },
            "ambiguous_digest_open",
            (0.20, 0.20, 0.60),
            [0.5, 0.5, 0.0],
            [0.46, 0.42, 0.12],
        ),
        (
            1,
            {
                "action_taken": "LATER",
                "outcome": "DELETED_FROM_DIGEST",
                "observed_user_selection": "ARCHIVE",
            },
            "reliable_singleton",
            (0.10, 0.10, 0.80),
            [0.0, 0.0, 1.0],
            [0.08, 0.06, 0.86],
        ),
    ],
)
def test_causal_fusion_conditions_exact_target_on_evidence_reliability(
    action,
    feedback,
    expected_reliability,
    expected_weights,
    expected_behavior,
    expected_target,
):
    agent = OnlineSDFTAgent(
        RecordingPolicy(),
        settings=OnlineSDFTSettings(
            reliable_teacher_weight=0.10,
            reliable_decision_weight=0.10,
            reliable_behavior_weight=0.80,
            ambiguous_teacher_weight=0.20,
            ambiguous_decision_weight=0.20,
            ambiguous_behavior_weight=0.60,
            ambiguous_projection="none",
        ),
    )
    teacher = np.array([2.0, 3.0, 5.0])
    decision = np.array([6.0, 3.0, 1.0])

    target = agent.training_target(
        teacher,
        teacher_action=2,
        action=action,
        feedback=feedback,
        decision_distribution=decision,
    )

    assert causal_evidence_reliability(action, feedback) == expected_reliability
    assert agent.fusion_weights(action, feedback) == expected_weights
    assert np.allclose(
        agent.causal_behavior_support(action, feedback),
        expected_behavior,
    )
    assert np.allclose(target, expected_target)
    assert np.isclose(target.sum(), 1.0)


def test_causal_fusion_uses_no_reward_or_evaluator_label():
    agent = OnlineSDFTAgent(RecordingPolicy())
    teacher = np.array([2.0, 3.0, 5.0])
    decision = np.array([6.0, 3.0, 1.0])
    feedback = {
        "action_taken": "LATER",
        "outcome": "OPENED_DIGEST",
        "observed_user_selection": "LATER",
        "reward": -1.0,
        "match_status": "MISS",
        "gold_action": "INTERRUPT",
    }
    target = agent.training_target(
        teacher,
        teacher_action=2,
        action=1,
        feedback=feedback,
        decision_distribution=decision,
    )

    # Evaluator-only or reward fields cannot affect a same-selection target.
    changed_noncausal_fields = {
        **feedback,
        "reward": 1_000.0,
        "match_status": "MATCH",
        "gold_action": "ARCHIVE",
    }
    assert np.allclose(
        agent.training_target(
            teacher,
            teacher_action=0,
            action=1,
            feedback=changed_noncausal_fields,
            decision_distribution=decision,
        ),
        target,
    )


@pytest.mark.parametrize(
    ("action", "feedback", "expected_support"),
    [
        (
            0,
            {
                "action_taken": "INTERRUPT",
                "outcome": "OPENED_IMMEDIATELY",
                "observed_user_selection": "INTERRUPT",
            },
            [1.0, 0.0, 0.0],
        ),
        (
            1,
            {
                "action_taken": "LATER",
                "outcome": "OPENED_DIGEST",
                "observed_user_selection": "LATER",
            },
            [1.0, 1.0, 0.0],
        ),
    ],
)
def test_support_likelihood_target_is_exact_causal_set(
    action,
    feedback,
    expected_support,
):
    agent = OnlineSDFTAgent(
        RecordingPolicy(),
        settings=OnlineSDFTSettings(
            target_mode="support_likelihood",
            reliable_teacher_weight=0.0,
            reliable_decision_weight=0.0,
            reliable_behavior_weight=0.0,
            ambiguous_teacher_weight=0.0,
            ambiguous_decision_weight=0.0,
            ambiguous_behavior_weight=0.0,
        ),
    )

    target = agent.training_target(
        np.array([np.nan, np.nan, np.nan]),
        teacher_action=2,
        action=action,
        feedback={
            **feedback,
            "reward": 1_000.0,
            "gold_action": "ARCHIVE",
        },
        decision_distribution=None,
    )

    assert target.tolist() == expected_support


def test_support_likelihood_replay_uses_dedicated_policy_objective():
    settings = OnlineSDFTSettings(
        target_mode="support_likelihood",
        reliable_teacher_weight=0.0,
        reliable_decision_weight=0.0,
        reliable_behavior_weight=0.0,
        ambiguous_teacher_weight=0.0,
        ambiguous_decision_weight=0.0,
        ambiguous_behavior_weight=0.0,
        replay_size=4,
        batch_size=1,
        warmup_examples=1,
        update_steps=1,
    )
    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(policy, settings=settings)
    observation = visible_observation("digest callback")

    agent.observe(
        observation,
        action=1,
        teacher_distribution=np.array([0.2, 0.5, 0.3]),
        teacher_action=1,
        feedback={
            "action_taken": "LATER",
            "outcome": "OPENED_DIGEST",
            "observed_user_selection": "LATER",
        },
        rng=np.random.default_rng(0),
        decision_distribution=None,
    )

    assert not policy.updates
    assert len(policy.support_updates) == 1
    context, support = policy.support_updates[0][0]
    assert context == observation.text
    assert support.tolist() == [1.0, 1.0, 0.0]


def test_online_sdft_can_skip_ambiguous_callbacks_entirely():
    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(
        policy,
        settings=OnlineSDFTSettings(
            replay_size=4,
            batch_size=1,
            warmup_examples=1,
            update_steps=2,
            ambiguous_update_mode="skip",
        ),
    )
    observation = visible_observation("ambiguous skip")

    agent.observe(
        observation,
        action=1,
        teacher_distribution=np.array([0.2, 0.5, 0.3]),
        teacher_action=1,
        feedback={
            "action_taken": "LATER",
            "outcome": "OPENED_DIGEST",
            "observed_user_selection": "LATER",
        },
        rng=np.random.default_rng(0),
        decision_distribution=np.array([0.2, 0.5, 0.3]),
    )

    assert not agent.replay
    assert not policy.updates
    assert agent.online_update_count == 0


def test_online_sdft_can_defer_ambiguous_callback_until_later_replay():
    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(
        policy,
        settings=OnlineSDFTSettings(
            replay_size=4,
            batch_size=2,
            warmup_examples=1,
            update_steps=1,
            ambiguous_update_mode="defer",
        ),
    )
    ambiguous = visible_observation("ambiguous deferred")
    agent.observe(
        ambiguous,
        action=1,
        teacher_distribution=np.array([0.2, 0.5, 0.3]),
        teacher_action=1,
        feedback={
            "action_taken": "LATER",
            "outcome": "OPENED_DIGEST",
            "observed_user_selection": "LATER",
        },
        rng=np.random.default_rng(0),
        decision_distribution=np.array([0.2, 0.5, 0.3]),
    )
    assert len(agent.replay) == 1
    assert not policy.updates

    reliable = visible_observation("reliable current")
    agent.observe(
        reliable,
        action=0,
        teacher_distribution=np.array([0.8, 0.1, 0.1]),
        teacher_action=0,
        feedback={
            "action_taken": "INTERRUPT",
            "outcome": "OPENED_IMMEDIATELY",
            "observed_user_selection": "INTERRUPT",
        },
        rng=np.random.default_rng(1),
        decision_distribution=np.array([0.5, 0.3, 0.2]),
    )

    assert len(policy.updates) == 1
    assert {context for context, _ in policy.updates[0]} == {
        ambiguous.text,
        reliable.text,
    }


def test_online_sdft_can_force_newest_only_on_first_optimizer_step():
    class FirstChoiceRng:
        def choice(self, candidates, *, size, replace, p):
            del replace, p
            return np.asarray(candidates[:size])

    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(
        policy,
        settings=OnlineSDFTSettings(
            replay_size=4,
            batch_size=1,
            warmup_examples=2,
            update_steps=2,
            force_newest_every_step=False,
        ),
    )
    for index in range(2):
        observation = visible_observation(f"reliable-{index}")
        agent.observe(
            observation,
            action=0,
            teacher_distribution=np.array([0.8, 0.1, 0.1]),
            teacher_action=0,
            feedback={
                "action_taken": "INTERRUPT",
                "outcome": "OPENED_IMMEDIATELY",
                "observed_user_selection": "INTERRUPT",
            },
            rng=FirstChoiceRng(),
            decision_distribution=np.array([0.5, 0.3, 0.2]),
        )

    assert len(policy.updates) == 2
    assert policy.updates[0][0][0] == "reliable-1"
    assert policy.updates[1][0][0] == "reliable-0"


def test_online_sdft_fixed_base_kl_uses_same_policy_anchor_once_per_row():
    beta = 0.25
    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(
        policy,
        settings=OnlineSDFTSettings(
            replay_size=4,
            batch_size=1,
            warmup_examples=1,
            update_steps=1,
            target_mode="teacher_only",
            reliable_teacher_weight=1.0,
            reliable_decision_weight=0.0,
            reliable_behavior_weight=0.0,
            ambiguous_teacher_weight=1.0,
            ambiguous_decision_weight=0.0,
            ambiguous_behavior_weight=0.0,
            base_kl_weight=beta,
        ),
    )
    observation = visible_observation("anchored context")
    teacher = np.array([0.1, 0.7, 0.2])
    base = np.array([0.4, 0.4, 0.2])

    agent.observe(
        observation,
        action=0,
        teacher_distribution=teacher,
        teacher_action=1,
        feedback={
            "action_taken": "INTERRUPT",
            "outcome": "OPENED_IMMEDIATELY",
            "observed_user_selection": "INTERRUPT",
        },
        rng=np.random.default_rng(0),
    )

    assert policy.base_calls == [observation.text]
    assert agent.replay[0][3] == pytest.approx(base)
    assert len(policy.updates) == 1
    _, effective_target = policy.updates[0][0]
    assert effective_target == pytest.approx(
        (teacher + beta * base) / (1.0 + beta)
    )


def test_ambiguous_projection_preserves_only_causally_supported_routes():
    settings = OnlineSDFTSettings(
        ambiguous_teacher_weight=0.0,
        ambiguous_decision_weight=1.0,
        ambiguous_behavior_weight=0.0,
        ambiguous_projection="causal_support",
    )
    agent = OnlineSDFTAgent(RecordingPolicy(), settings=settings)
    target = agent.training_target(
        np.array([0.8, 0.1, 0.1]),
        teacher_action=0,
        action=1,
        feedback={
            "action_taken": "LATER",
            "outcome": "OPENED_DIGEST",
            "observed_user_selection": "LATER",
        },
        decision_distribution=np.array([0.2, 0.6, 0.2]),
    )

    assert np.allclose(target, [0.25, 0.75, 0.0])


def test_unknown_callback_is_censored_before_sdft_target_construction():
    agent = OnlineSDFTAgent(RecordingPolicy())
    feedback = {
        "action_taken": "ARCHIVE",
        "outcome": "NO_OBSERVABLE_SELECTION",
        "observed_user_selection": "UNKNOWN",
    }

    assert causal_evidence_reliability(2, feedback) == "censored_unknown"
    with pytest.raises(ValueError, match="censored feedback"):
        agent.fusion_weights(2, feedback)
    with pytest.raises(ValueError, match="censored feedback"):
        agent.training_target(
            np.array([0.2, 0.3, 0.5]),
            teacher_action=2,
            action=2,
            feedback=feedback,
            decision_distribution=np.array([0.6, 0.3, 0.1]),
        )

    agent.observe(
        visible_observation("censored"),
        action=2,
        teacher_distribution=None,
        teacher_action=None,
        feedback=feedback,
        rng=np.random.default_rng(0),
    )
    assert not agent.policy.updates


def test_selection_balanced_replay_weights_old_less_common_lessons():
    class CapturingAgent(OnlineSDFTAgent):
        def update_from_replay(self, rows):
            self.last_update_rows = rows

    class CapturingRng:
        def choice(self, candidates, *, size, replace, p):
            self.candidates = candidates.copy()
            self.size = size
            self.replace = replace
            self.probabilities = p.copy()
            return np.array([3, 0])

    settings = OnlineSDFTSettings(
        replay_size=5,
        batch_size=3,
        replay_strategy="selection_balanced",
        target_mode="teacher_only",
        reliable_teacher_weight=1.0,
        reliable_decision_weight=0.0,
        reliable_behavior_weight=0.0,
        ambiguous_teacher_weight=1.0,
        ambiguous_decision_weight=0.0,
        ambiguous_behavior_weight=0.0,
    )
    policy = RecordingPolicy()
    agent = CapturingAgent(policy, settings=settings)
    target = np.array([0.2, 0.5, 0.3])
    agent.replay = [
        ("later-1", target, "LATER"),
        ("later-2", target, "LATER"),
        ("later-3", target, "LATER"),
        ("archive-1", target, "ARCHIVE"),
    ]
    rng = CapturingRng()

    agent.observe(
        StudentObservation("new-interrupt", np.ones(3)),
        action=0,
        teacher_distribution=target,
        teacher_action=1,
        feedback={"observed_user_selection": "INTERRUPT"},
        rng=rng,
    )

    assert rng.candidates.tolist() == [0, 1, 2, 3]
    assert rng.size == 2
    assert rng.replace is False
    assert settings.replay_recency_half_life is None
    np.testing.assert_array_equal(
        rng.probabilities,
        np.asarray([1 / 6, 1 / 6, 1 / 6, 1 / 2]),
    )
    assert [row[0] for row in agent.last_update_rows] == [
        "new-interrupt",
        "archive-1",
        "later-1",
    ]
    assert agent.replay[-1][2] == "INTERRUPT"


def test_selection_balanced_recency_preserves_group_mass_and_includes_newest():
    class CapturingAgent(OnlineSDFTAgent):
        def update_from_replay(self, rows):
            self.update_rows.append(rows)

    class CapturingRng:
        def __init__(self):
            self.calls = []

        def choice(self, candidates, *, size, replace, p):
            self.calls.append(
                (candidates.copy(), size, replace, p.copy())
            )
            if len(self.calls) == 1:
                return np.array([2, 1])
            return np.array([4, 2, 0])

    settings = OnlineSDFTSettings(
        replay_size=5,
        batch_size=3,
        update_steps=2,
        replay_strategy="selection_balanced",
        replay_recency_half_life=1.0,
        ambiguous_replay_group_weight=0.5,
        force_newest_every_step=False,
        target_mode="teacher_only",
        reliable_teacher_weight=1.0,
        reliable_decision_weight=0.0,
        reliable_behavior_weight=0.0,
        ambiguous_teacher_weight=1.0,
        ambiguous_decision_weight=0.0,
        ambiguous_behavior_weight=0.0,
    )
    target = np.array([0.2, 0.5, 0.3])
    agent = CapturingAgent(RecordingPolicy(), settings=settings)
    agent.update_rows = []
    agent.replay = [
        ("later-old", target, "LATER"),
        ("interrupt-old", target, "INTERRUPT"),
        ("later-new", target, "LATER"),
        ("ambiguous", target, "AMBIGUOUS"),
    ]
    rng = CapturingRng()

    agent.observe(
        StudentObservation("interrupt-new", np.ones(3)),
        action=0,
        teacher_distribution=target,
        teacher_action=0,
        feedback={"observed_user_selection": "INTERRUPT"},
        rng=rng,
    )

    assert len(rng.calls) == 2
    first_candidates, first_size, first_replace, first_probabilities = rng.calls[0]
    assert first_candidates.tolist() == [0, 1, 2, 3]
    assert first_size == 2
    assert first_replace is False
    np.testing.assert_allclose(
        first_probabilities,
        np.asarray([2 / 25, 2 / 5, 8 / 25, 1 / 5]),
        rtol=0.0,
        atol=1e-15,
    )

    second_candidates, second_size, second_replace, second_probabilities = (
        rng.calls[1]
    )
    assert second_candidates.tolist() == [0, 1, 2, 3, 4]
    assert second_size == 3
    assert second_replace is False
    np.testing.assert_allclose(
        second_probabilities,
        np.asarray([2 / 25, 2 / 45, 8 / 25, 1 / 5, 16 / 45]),
        rtol=0.0,
        atol=1e-15,
    )

    # Label groups retain masses 1, 1, and 0.5 before global normalization.
    assert second_probabilities[[0, 2]].sum() == pytest.approx(2 / 5)
    assert second_probabilities[[1, 4]].sum() == pytest.approx(2 / 5)
    assert second_probabilities[3] == pytest.approx(1 / 5)
    # The second update samples over actual replay indices, so the just-added
    # INTERRUPT has age zero and 8x the weight of the age-three INTERRUPT.
    assert second_probabilities[4] / second_probabilities[1] == pytest.approx(8.0)
    assert agent.update_rows[0][0][0] == "interrupt-new"
    assert agent.update_rows[1][0][0] == "interrupt-new"


def test_icl_uses_latest_direct_interactions_in_chronological_order():
    agent = ICLAgent(RecordingPolicy())
    for index in range(ICL_K + 2):
        agent.observe(
            visible_observation(f"past-{index}"),
            action=0,
            teacher_distribution=None,
            teacher_action=None,
            feedback={
                "action_taken": "INTERRUPT",
                "outcome": "OPENED_AFTER_DELAY",
                "observed_user_selection": "LATER",
            },
            rng=np.random.default_rng(index),
        )
    examples = agent.prompt_examples(visible_observation("current"))
    assert [row["context"] for row in examples] == [
        f"past-{index}" for index in range(2, ICL_K + 2)
    ]
    assert all(row["context"] != "current" for row in examples)
    assert all(
        row["eventual_user_action"] == "OPENED_AFTER_DELAY"
        for row in examples
    )
    assert all(row["observed_user_selection"] == "LATER" for row in examples)
    assert all("teacher_action" not in row for row in examples)


def test_icl_retains_unknown_interaction_without_prompting_a_label():
    agent = ICLAgent(RecordingPolicy(), icl_examples=1)
    agent.observe(
        visible_observation("archived"),
        action=2,
        teacher_distribution=None,
        teacher_action=None,
        feedback={
            "action_taken": "ARCHIVE",
            "outcome": "NO_OBSERVABLE_SELECTION",
            "observed_user_selection": "UNKNOWN",
        },
        rng=np.random.default_rng(0),
    )
    assert len(agent.memory) == 1
    assert agent.memory[0].prompt_example()[
        "observed_user_selection"
    ] == "UNKNOWN"
    assert agent.prompt_examples(visible_observation("current")) == []


def test_icl_budget_counts_only_reliable_singleton_callbacks():
    policy = RecordingPolicy()
    agent = ICLAgent(policy, icl_examples=2)
    records = (
        ("known-old", "LATER", "OPENED_DIGEST"),
        ("known-new", "ARCHIVE", "DELETED_FROM_DIGEST"),
        ("unknown-newest", "UNKNOWN", "NO_OBSERVABLE_SELECTION"),
    )
    for index, (name, selection, outcome) in enumerate(records):
        agent.observe(
            visible_observation(name),
            action=1,
            teacher_distribution=None,
            teacher_action=None,
            feedback={
                "action_taken": "LATER",
                "outcome": outcome,
                "observed_user_selection": selection,
            },
            rng=np.random.default_rng(index),
        )

    examples = agent.prompt_examples(visible_observation("current"))
    assert len(agent.memory) == 3
    assert [row["context"] for row in examples] == ["known-new"]
    assert examples[0]["observed_user_selection"] == "ARCHIVE"


def test_zero_icl_budget_returns_no_history():
    agent = ICLAgent(RecordingPolicy(), icl_examples=0)
    agent.observe(
        visible_observation("past"),
        action=1,
        teacher_distribution=None,
        teacher_action=None,
        feedback={
            "action_taken": "LATER",
            "outcome": "OPENED_DIGEST",
            "observed_user_selection": "LATER",
        },
        rng=np.random.default_rng(0),
    )
    assert agent.prompt_examples(visible_observation("current")) == []


def test_rag_retrieves_visible_nearest_neighbors_best_match_last():
    agent = RAGAgent(RecordingPolicy(), rag_examples=2)
    exact_old = visible_observation("exact-old", hour=23.5, regime=2)
    distractor = visible_observation(
        "distractor",
        category=1,
        hour=12.0,
        regime=0,
    )
    exact_new = visible_observation("exact-new", hour=23.5, regime=2)
    for index, past in enumerate((exact_old, distractor, exact_new)):
        agent.observe(
            past,
            action=0,
            teacher_distribution=None,
            teacher_action=None,
            feedback={
                "action_taken": "INTERRUPT",
                "outcome": "OPENED_IMMEDIATELY",
                "observed_user_selection": "INTERRUPT",
            },
            rng=np.random.default_rng(index),
        )
    current = visible_observation("current", hour=23.5, regime=2)
    examples = agent.prompt_examples(current)
    assert [row["context"] for row in examples] == [
        "exact-old",
        "exact-new",
    ]


def test_rag_retrieves_only_reliable_singleton_neighbors():
    policy = RecordingPolicy()
    agent = RAGAgent(policy, rag_examples=2)
    known_farther = visible_observation(
        "known-farther",
        hour=9.0,
        importance=0.65,
    )
    known_nearer = visible_observation(
        "known-nearer",
        hour=11.0,
        importance=0.55,
    )
    unknown_exact = visible_observation(
        "unknown-exact",
        hour=12.0,
        importance=0.5,
    )
    records = (
        (known_farther, "LATER", "OPENED_DIGEST"),
        (known_nearer, "ARCHIVE", "DELETED_FROM_DIGEST"),
        (unknown_exact, "UNKNOWN", "NO_OBSERVABLE_SELECTION"),
    )
    for index, (past, selection, outcome) in enumerate(records):
        agent.observe(
            past,
            action=1,
            teacher_distribution=None,
            teacher_action=None,
            feedback={
                "action_taken": "LATER",
                "outcome": outcome,
                "observed_user_selection": selection,
            },
            rng=np.random.default_rng(index),
        )

    current = visible_observation("current", hour=12.0, importance=0.5)
    assert mixed_context_similarity(current, unknown_exact, text_weight=0.0) == 1.0
    examples = agent.prompt_examples(current)
    assert len(agent.memory) == 3
    assert [row["context"] for row in examples] == ["known-nearer"]
    assert examples[0]["observed_user_selection"] == "ARCHIVE"


def test_ambiguous_digest_open_is_retained_but_not_hard_labeled_in_memory_prompt():
    for agent in (ICLAgent(RecordingPolicy()), RAGAgent(RecordingPolicy())):
        agent.observe(
            visible_observation("digest-open"),
            action=1,
            teacher_distribution=None,
            teacher_action=None,
            feedback={
                "action_taken": "LATER",
                "outcome": "OPENED_DIGEST",
                "observed_user_selection": "LATER",
            },
            rng=np.random.default_rng(0),
        )

        assert len(agent.memory) == 1
        assert agent.prompt_examples(visible_observation("current")) == []


def test_mixed_similarity_handles_midnight_as_circular_time():
    before_midnight = visible_observation("before", hour=23.5)
    after_midnight = visible_observation("after", hour=0.5)
    noon = visible_observation("noon", hour=12.0)
    assert mixed_context_similarity(before_midnight, after_midnight) > (
        mixed_context_similarity(before_midnight, noon)
    )


def test_visible_notification_text_similarity_distinguishes_scenarios():
    critical = (
        "The notification title is Critical errors above threshold. "
        "The message says Production failures reached eight percent."
    )
    related = (
        "The notification title is Critical errors above threshold. "
        "The message says Production failures reached ten percent."
    )
    routine = (
        "The notification title is Weekly health report. "
        "The message says All checks are within the normal range."
    )
    assert notification_text_similarity(critical, related) > (
        notification_text_similarity(critical, routine)
    )


def test_rag_text_weight_is_visible_only_and_bounded():
    query = visible_observation("critical production failure")
    related = visible_observation("critical production error")
    unrelated = visible_observation("weekly social photos")
    assert mixed_context_similarity(query, related, text_weight=1.0) > (
        mixed_context_similarity(query, unrelated, text_weight=1.0)
    )
    with pytest.raises(ValueError, match="text_weight"):
        mixed_context_similarity(query, related, text_weight=1.1)
    with pytest.raises(ValueError, match="rag_text_weight"):
        RAGAgent(RecordingPolicy(), rag_text_weight=-0.1)


def test_reinforce_defaults_train_the_common_lora_adapter():
    settings = DEFAULT_REINFORCE_SETTINGS
    policy = RecordingPolicy()
    agent = REINFORCEAgent(policy)

    assert agent.settings is settings
    assert settings.learning_rate == REINFORCE_LR == 1e-4
    assert settings.batch_size == REINFORCE_BATCH_SIZE == 8
    assert settings.baseline_step == REINFORCE_BASELINE_STEP == 0.0
    assert settings.entropy_coef == REINFORCE_ENTROPY_COEF == 1.0
    assert settings.max_grad_norm == REINFORCE_MAX_GRAD_NORM == 1.0
    assert dict(settings.reward_outcome_map) == (
        REINFORCE_TRAINING_OUTCOME_REWARDS
    )
    assert settings.to_dict()["reward_outcome_map"] == (
        REINFORCE_TRAINING_OUTCOME_REWARDS
    )
    assert agent.reward_outcome_map == REINFORCE_TRAINING_OUTCOME_REWARDS
    assert agent.online_batch_size == REINFORCE_BATCH_SIZE
    assert policy.learning_rate == REINFORCE_LR
    assert policy.configured_settings == [DEFAULT_SDFT_SETTINGS]
    assert not hasattr(agent, "behavior_mode")
    assert not hasattr(agent, "behavior_epsilon")
    assert not hasattr(agent, "interrupt_probe_mix")
    assert agent.action_probs(visible_observation("visible")) == pytest.approx(
        np.array([0.2, 0.5, 0.3])
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"learning_rate": 0.0}, "learning rate"),
        ({"batch_size": 0}, "batch size"),
        ({"batch_size": 1.5}, "batch size"),
        ({"baseline_step": -0.1}, "baseline step"),
        ({"baseline_step": 1.1}, "baseline step"),
        ({"entropy_coef": -0.1}, "entropy coefficient"),
        ({"max_grad_norm": 0.0}, "gradient norm"),
        ({"reward_outcome_map": {"OPENED": np.inf}}, "rewards must be finite"),
        ({"reward_outcome_map": (("OPENED", 1.0), ("OPENED", 2.0))}, "unique"),
        ({"reward_outcome_map": {"": 1.0}}, "non-empty strings"),
    ],
)
def test_reinforce_settings_reject_invalid_arms(overrides, message):
    with pytest.raises(ValueError, match=message):
        REINFORCESettings(**overrides)


def test_reinforce_factual_reward_map_and_lora_update_are_settings_driven():
    settings = REINFORCESettings(
        learning_rate=5e-4,
        batch_size=1,
        baseline_step=0.0,
        entropy_coef=0.03,
        max_grad_norm=0.7,
        reward_outcome_map={"OPENED_IMMEDIATELY": 2.0},
    )
    observation = visible_observation("critical release approval in ten minutes")
    policy = RecordingPolicy()
    agent = REINFORCEAgent(policy, settings=settings)
    agent.observe(
        observation,
        action=0,
        teacher_distribution=None,
        teacher_action=None,
        feedback={
            "outcome": "OPENED_IMMEDIATELY",
            "observed_user_selection": "INTERRUPT",
            # A configured arm maps only the factual outcome; this scalar is
            # deliberately contradictory to prove it is not consulted.
            "reward": -999.0,
        },
        rng=np.random.default_rng(0),
    )
    assert policy.learning_rate == pytest.approx(5e-4)
    assert policy.reinforce_updates == [
        ([(observation.text, 0, 2.0)], 0.03, 0.7)
    ]
    assert agent.reward_baseline == 0.0
    assert settings.to_dict()["reward_outcome_map"] == {
        "OPENED_IMMEDIATELY": 2.0
    }


def test_reinforce_default_shaping_ignores_reported_and_evaluator_only_values():
    observation = visible_observation("visible deletion callback")
    agent = REINFORCEAgent(RecordingPolicy())
    agent.observe(
        observation,
        action=0,
        teacher_distribution=None,
        teacher_action=None,
        feedback={
            "outcome": "DELETED_NOTIFICATION",
            "observed_user_selection": "ARCHIVE",
            # Reported reward and evaluator-only diagnostics are deliberately
            # contradictory; learner shaping uses only the factual outcome.
            "reward": 999.0,
            "gold_action": "INTERRUPT",
            "utility": 999.0,
            "deadline": 0.0,
            "urgency": 1.0,
            "affinity": 1.0,
        },
        rng=np.random.default_rng(0),
    )

    assert agent.last_training_reward == -5.0
    assert agent.pending_updates == [(observation.text, 0, -5.0)]
    assert agent.reward_baseline == 0.0
    assert not agent.policy.reinforce_updates


def test_reinforce_configured_reward_map_requires_a_mapped_factual_outcome():
    agent = REINFORCEAgent(
        RecordingPolicy(),
        settings=REINFORCESettings(
            batch_size=1,
            reward_outcome_map={"OPENED_IMMEDIATELY": 1.0},
        ),
    )
    agent.observe(
        visible_observation("censored"),
        action=2,
        teacher_distribution=None,
        teacher_action=None,
        feedback={
            "outcome": "NO_OBSERVABLE_SELECTION",
            "observed_user_selection": "UNKNOWN",
            "reward": 0.0,
        },
        rng=np.random.default_rng(0),
    )
    assert agent.online_update_count == 0
    assert not agent.pending_updates
    assert agent.last_training_reward is None

    with pytest.raises(ValueError, match="configured reward map"):
        agent.observe(
            visible_observation("visible"),
            action=0,
            teacher_distribution=None,
            teacher_action=None,
            feedback={
                "outcome": "OPENED_AFTER_DELAY",
                "observed_user_selection": "LATER",
                "reward": -1.0,
            },
            rng=np.random.default_rng(0),
    )
    assert agent.online_update_count == 0
    assert not agent.pending_updates


def test_create_agent_accepts_optional_reinforce_settings():
    settings = REINFORCESettings(
        learning_rate=4e-4,
        batch_size=2,
    )
    adapter_settings = OnlineSDFTSettings(learning_rate=9e-4)
    policy = RecordingPolicy()
    agent = create_agent(
        "REINFORCE",
        policy,
        sdft_settings=adapter_settings,
        reinforce_settings=settings,
    )

    assert isinstance(agent, REINFORCEAgent)
    assert agent.settings is settings
    assert agent.online_batch_size == 2
    assert policy.learning_rate == pytest.approx(4e-4)
    assert policy.configured_settings == [adapter_settings]


def test_reinforce_uses_only_action_reward_and_causal_baseline():
    policy = RecordingPolicy()
    agent = REINFORCEAgent(
        policy,
        settings=REINFORCESettings(
            learning_rate=0.03,
            batch_size=2,
            baseline_step=0.05,
            entropy_coef=0.01,
            max_grad_norm=1.0,
        ),
    )
    observation = visible_observation("visible")
    agent.observe(
        observation,
        action=2,
        teacher_distribution=None,
        teacher_action=None,
        feedback={
            "observed_user_selection": "ARCHIVE",
            "reward": 0.4,
        },
        rng=np.random.default_rng(0),
    )
    assert policy.learning_rate == pytest.approx(0.03)
    assert not policy.reinforce_updates
    assert agent.online_update_count == 0
    assert agent.last_observation_update_count == 0
    assert agent.pending_updates == [(observation.text, 2, 0.4)]
    assert np.isclose(agent.reward_baseline, 0.02)
    assert not agent.memory


def test_reinforce_applies_one_lora_update_per_complete_fresh_batch():
    policy = RecordingPolicy()
    settings = REINFORCESettings(batch_size=2, baseline_step=0.0)
    agent = REINFORCEAgent(policy, settings=settings)
    observation = visible_observation("visible")
    for count in range(1, 5):
        agent.observe(
            observation,
            action=1,
            teacher_distribution=None,
            teacher_action=None,
            feedback={
                "observed_user_selection": "LATER",
                "reward": 0.25,
            },
            rng=np.random.default_rng(count),
            decision_distribution=np.array([0.2, 0.6, 0.2]),
        )
        assert agent.online_update_count == count // settings.batch_size
        assert len(agent.pending_updates) == count % settings.batch_size
        if count % settings.batch_size:
            assert agent.last_observation_update_count == 0
        else:
            assert agent.last_observation_update_count == 1
    assert len(policy.reinforce_updates) == 2
    assert all(len(batch) == 2 for batch, _, _ in policy.reinforce_updates)


def test_unknown_selection_is_unlabeled_memory_but_never_a_gradient_target():
    observation = visible_observation("censored")
    feedback = {
        "observed_user_selection": "UNKNOWN",
        "match_status": "UNKNOWN",
        "reward": 0.0,
    }
    for agent_class in (ICLAgent, RAGAgent):
        policy = RecordingPolicy()
        agent = agent_class(policy)
        agent.observe(
            observation,
            action=2,
            teacher_distribution=np.array([0.3, 0.6, 0.1]),
            teacher_action=1,
            feedback=feedback,
            rng=np.random.default_rng(0),
        )
        assert len(agent.memory) == 1
        assert agent.memory[0].prompt_example()[
            "observed_user_selection"
        ] == "UNKNOWN"
        assert not policy.updates

    policy = RecordingPolicy()
    sdft = OnlineSDFTAgent(policy)
    sdft.observe(
        observation,
        action=2,
        teacher_distribution=np.array([0.3, 0.6, 0.1]),
        teacher_action=1,
        feedback=feedback,
        rng=np.random.default_rng(0),
    )
    assert not sdft.memory
    assert not policy.updates

    policy = RecordingPolicy()
    reinforce = REINFORCEAgent(policy)
    reinforce.observe(
        observation,
        action=2,
        teacher_distribution=None,
        teacher_action=None,
        feedback=feedback,
        rng=np.random.default_rng(0),
    )
    assert not policy.reinforce_updates
    assert reinforce.reward_baseline == 0.0
    assert reinforce.online_update_count == 0
    assert reinforce.last_observation_update_count == 0
    assert reinforce.last_training_reward is None
    assert not reinforce.pending_updates

@pytest.mark.parametrize(
    "feedback",
    [
        {"reward": 1.0},
        {"observed_user_selection": "MALFORMED", "reward": 1.0},
    ],
)
def test_reinforce_rejects_missing_or_malformed_callback_selection(feedback):
    agent = REINFORCEAgent(RecordingPolicy())
    with pytest.raises(ValueError, match="explicit observed user selection"):
        agent.observe(
            visible_observation("visible"),
            action=0,
            teacher_distribution=None,
            teacher_action=None,
            feedback=feedback,
            rng=np.random.default_rng(0),
        )
    assert agent.online_update_count == 0


def test_reinforce_delegates_action_token_advantages_entropy_and_clipping():
    policy = RecordingPolicy()
    agent = REINFORCEAgent(
        policy,
        settings=REINFORCESettings(
            learning_rate=2e-4,
            batch_size=2,
            baseline_step=0.5,
            entropy_coef=0.02,
            max_grad_norm=0.4,
        ),
    )
    observation = visible_observation("visible")
    rewards = (2.0, 4.0)
    for index, reward in enumerate(rewards):
        agent.observe(
            observation,
            action=index,
            teacher_distribution=None,
            teacher_action=None,
            feedback={
                "observed_user_selection": "INTERRUPT",
                "reward": reward,
            },
            rng=np.random.default_rng(index),
        )

    assert policy.reinforce_updates == [
        (
            [(observation.text, 0, 2.0), (observation.text, 1, 3.0)],
            0.02,
            0.4,
        )
    ]
    assert agent.reward_baseline == pytest.approx(2.5)
    assert agent.online_update_count == 1
    assert agent.last_observation_update_count == 1
    assert not agent.pending_updates
    source = getsource(REINFORCEAgent.observe)
    assert "policy.reinforce_update" in source
    assert "online_update_count += 1" in source


def test_liquid_policy_reinforce_update_uses_autograd_and_gradient_clipping():
    import torch

    class TinyPolicyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.route_logits = torch.nn.Parameter(
                torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.0, -0.2]])
            )

    policy = object.__new__(LiquidLLMPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.model = TinyPolicyModel()
    policy.optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
    policy.render_prompt = lambda context: context
    policy._action_logits = lambda prompts: policy.model.route_logits[: len(prompts)]

    loss = policy.reinforce_update(
        [("first", 0, 3.0), ("second", 2, -2.0)],
        entropy_coef=0.01,
        max_grad_norm=0.1,
    )

    assert np.isfinite(loss)
    assert policy.model.training is True
    assert policy.model.route_logits.grad.norm().item() == pytest.approx(
        0.1,
        rel=1e-4,
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        policy.reinforce_update([], entropy_coef=0.0, max_grad_norm=1.0)


def test_compact_icl_prompt_uses_one_structured_history_message():
    policy = prompt_policy()
    past_context = (
        "The notification title is Design review starts soon. The message "
        "says Maya is waiting. This is a calendar notification that arrived "
        "at 10:15 local time during the weekday period. Its on-device "
        "importance score is 0.88 out of 1."
    )
    current_context = (
        "The notification title is Approval needed. The message says Review "
        "the release. This is a manager notification that arrived at 10:30 "
        "local time during the weekday period. Its on-device importance score "
        "is 0.91 out of 1."
    )
    rendered = policy.render_prompt(
        current_context,
        [
            {
                "context": past_context,
                "executed_action": "INTERRUPT",
                "eventual_user_action": "OPENED_AFTER_DELAY",
                "observed_user_selection": "LATER",
                "delay_minutes": 120,
            }
        ],
    )
    assert "Past completed interactions:" in rendered
    assert "1. The notification title is Design review" in rendered
    assert "This is a calendar notification" in rendered
    assert "delivered the notification as an immediate interruption" in rendered
    assert "opened it 120 minutes later" in rendered
    assert "revealed LATER as the observed user selection" in rendered
    assert "Its observed route was B for LATER." in rendered
    assert "UNKNOWN is unlabeled." in rendered
    assert "assistant: B" not in rendered
    assert "Notification: The notification title is Approval needed" in rendered
    assert rendered.endswith("Route:")
    history_message = policy.tokenizer.messages[-1]["content"]
    assert (
        len(history_message) - len(past_context) - len(current_context)
        <= 330
    )
    assert (
        len(history_message.split())
        - len(past_context.split())
        - len(current_context.split())
        <= 55
    )


def test_compact_prompt_preserves_realistic_title_and_body():
    policy = prompt_policy()
    context = (
        "The notification title is Checkout review starts in 10 minutes. "
        "The message says Maya asked you to join on time. Tap to open the "
        "video call. "
        "This is a calendar notification that arrived at 10:30 local time "
        "during the weekday period. Its on-device importance score is 0.92 "
        "out of 1."
    )
    rendered = policy.render_prompt(context)
    assert "notification title is Checkout review starts" in rendered
    assert "message says Maya asked you to join on time" in rendered
    assert "This is a calendar notification" in rendered
    assert "10:30 local time during the weekday period" in rendered
    assert "importance score is 0.92 out of 1" in rendered
    assert "category=" not in rendered
    assert "busy" not in rendered
    assert "interruption_filter" not in rendered


def test_compact_prompt_keeps_unknown_history_unlabeled():
    policy = prompt_policy()
    rendered = policy.render_prompt(
        "category=manager",
        [{
            "context": "category=promo",
            "executed_action": "ARCHIVE",
            "eventual_user_action": "NO_OBSERVABLE_SELECTION",
            "observed_user_selection": "UNKNOWN",
        }],
    )
    assert "This interaction is unlabeled." in rendered
    assert "UNKNOWN is unlabeled." in rendered
    assert "Its observed route was C" not in rendered
    assert "assistant: C" not in rendered


def test_structured_history_is_one_user_message_with_valid_roles():
    policy = prompt_policy()
    policy.render_prompt(
        "current-notification",
        [
            history_example("known-later", "LATER"),
            history_example(
                "censored-exact",
                "UNKNOWN",
                executed_action="ARCHIVE",
                outcome="NO_OBSERVABLE_SELECTION",
            ),
            history_example(
                "known-archive",
                "ARCHIVE",
                outcome="DELETED_FROM_DIGEST",
            ),
        ],
    )

    messages = policy.tokenizer.messages
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "known-later" in messages[-1]["content"]
    assert "censored-exact" in messages[-1]["content"]
    assert "This interaction is unlabeled." in messages[-1]["content"]
    assert "UNKNOWN is unlabeled." in messages[-1]["content"]
    assert "known-archive" in messages[-1]["content"]
    assert policy.tokenizer.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
    }


def test_causal_demo_prompt_uses_alternating_label_only_examples():
    policy = prompt_policy(prompt_style="causal_demos")
    rendered = policy.render_prompt(
        "current-notification",
        [
            history_example(
                "known-later",
                "LATER",
                executed_action="INTERRUPT",
                outcome="OPENED_AFTER_DELAY",
            ),
            history_example(
                "censored-exact",
                "UNKNOWN",
                executed_action="ARCHIVE",
                outcome="NO_OBSERVABLE_SELECTION",
            ),
            history_example(
                "known-archive",
                "ARCHIVE",
                outcome="DELETED_FROM_DIGEST",
            ),
        ],
    )

    messages = policy.tokenizer.messages
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1]["content"] == "Past notification: known-later\nRoute:"
    assert messages[2]["content"] == "B"
    assert messages[3]["content"] == "Past notification: known-archive\nRoute:"
    assert messages[4]["content"] == "C"
    assert messages[5]["content"] == (
        "Current notification: current-notification\nRoute:"
    )
    assert "censored-exact" not in rendered
    assert "OPENED_AFTER_DELAY" not in rendered
    assert "immediate interruption" not in rendered


def test_causal_demo_instruction_requires_a_reliable_label():
    class EchoTokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            del kwargs
            return "\n".join(message["content"] for message in messages)

    compact = prompt_policy(prompt_style="compact")
    compact.tokenizer = EchoTokenizer()
    policy = prompt_policy(prompt_style="causal_demos")
    policy.tokenizer = EchoTokenizer()
    without_history = policy.render_prompt("current")
    unknown_only = policy.render_prompt(
        "current",
        [history_example("censored", "UNKNOWN")],
    )
    with_label = policy.render_prompt(
        "current",
        [history_example("known", "INTERRUPT")],
    )

    marker = "reliable observed user selection"
    assert without_history == compact.render_prompt("current")
    assert unknown_only == without_history
    assert marker not in without_history
    assert marker not in unknown_only
    assert marker in with_label


def test_prompt_role_guard_rejects_nonalternating_or_incomplete_chats():
    with pytest.raises(ValueError, match="must alternate"):
        LiquidLLMPolicy._assert_role_alternation(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ]
        )
    with pytest.raises(ValueError, match="must end with the current user query"):
        LiquidLLMPolicy._assert_role_alternation(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "example"},
                {"role": "assistant", "content": "B"},
            ]
        )


def test_prompt_token_budget_accepts_768_and_rejects_769():
    assert PROMPT_TOKEN_BUDGET == 768
    LiquidLLMPolicy.assert_prompt_token_budget([1, PROMPT_TOKEN_BUDGET])
    with pytest.raises(ValueError, match=r"max=769, budget=768"):
        LiquidLLMPolicy.assert_prompt_token_budget(
            [PROMPT_TOKEN_BUDGET + 1]
        )


def test_liquid_policy_resets_adapter_and_creates_optimizer_only_for_learning_arm():
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([7.0, 8.0]))

    policy = object.__new__(LiquidLLMPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.sdft_settings = OnlineSDFTSettings(
        optimizer_weight_decay=0.02,
    )
    policy.model = TinyModel()
    policy._initial_adapter = {
        "weight": policy.model.weight.detach().cpu().clone(),
    }
    policy.optimizer = object()
    policy.model.train()
    with torch.no_grad():
        policy.model.weight.add_(10.0)

    policy.start_run(None)

    assert policy.model.training is False
    assert policy.model.weight.detach().tolist() == [7.0, 8.0]
    assert policy.optimizer is None

    with torch.no_grad():
        policy.model.weight.mul_(0.0)
    policy.start_run(1e-4)

    assert policy.model.weight.detach().tolist() == [7.0, 8.0]
    assert isinstance(policy.optimizer, torch.optim.AdamW)
    assert policy.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert policy.optimizer.param_groups[0]["betas"] == pytest.approx((0.9, 0.999))
    assert policy.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.02)


def test_liquid_policy_uses_distinct_lm_head_learning_rate_by_module_identity():
    import torch

    class TinyOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.tensor([2.0]))

    class TinySplitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body_adapter = torch.nn.Parameter(torch.tensor([1.0]))
            self.output_adapter = TinyOutput()

        def get_output_embeddings(self):
            return self.output_adapter

    policy = object.__new__(LiquidLLMPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.sdft_settings = OnlineSDFTSettings(
        lora_target_modules=("q_proj", "lm_head"),
        lm_head_learning_rate=2e-3,
        optimizer_beta1=0.5,
        optimizer_weight_decay=0.03,
    )
    policy.model = TinySplitModel()
    policy._initial_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.model.named_parameters()
    }
    policy.optimizer = None

    policy.start_run(1e-4)

    assert isinstance(policy.optimizer, torch.optim.AdamW)
    parameter_lrs = {
        id(parameter): group["lr"]
        for group in policy.optimizer.param_groups
        for parameter in group["params"]
    }
    assert parameter_lrs == {
        id(policy.model.body_adapter): pytest.approx(1e-4),
        id(policy.model.output_adapter.adapter): pytest.approx(2e-3),
    }
    assert all(
        group["weight_decay"] == pytest.approx(0.03)
        for group in policy.optimizer.param_groups
    )
    assert all(
        group["betas"] == pytest.approx((0.5, 0.999))
        for group in policy.optimizer.param_groups
    )


@pytest.mark.parametrize(
    ("a_scale", "expected_a_lr", "expected_trainable"),
    [
        (1.0 / 16.0, 4.375e-5, True),
        (0.0, None, False),
    ],
)
def test_liquid_policy_scales_or_freezes_only_lora_a_factors(
    a_scale,
    expected_a_lr,
    expected_trainable,
):
    import torch

    class TinyFactors(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 1, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(1, 2, bias=False)}
            )

        def get_output_embeddings(self):
            return torch.nn.Identity()

    policy = object.__new__(LiquidLLMPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.sdft_settings = OnlineSDFTSettings(
        lora_a_learning_rate_scale=a_scale,
    )
    policy.model = TinyFactors()
    policy._initial_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.model.named_parameters()
    }
    initial_a = policy.model.lora_A["default"].weight.detach().clone()
    policy.optimizer = None

    expected_parameter_count = (
        sum(parameter.numel() for parameter in policy.model.parameters())
        if expected_trainable
        else policy.model.lora_B["default"].weight.numel()
    )
    assert policy.trainable_parameters == expected_parameter_count

    policy.start_run(7e-4)

    a_parameter = policy.model.lora_A["default"].weight
    b_parameter = policy.model.lora_B["default"].weight
    parameter_lrs = {
        id(parameter): group["lr"]
        for group in policy.optimizer.param_groups
        for parameter in group["params"]
    }
    assert a_parameter.requires_grad is expected_trainable
    assert b_parameter.requires_grad is True
    assert parameter_lrs[id(b_parameter)] == pytest.approx(7e-4)
    if expected_a_lr is None:
        assert id(a_parameter) not in parameter_lrs
    else:
        assert parameter_lrs[id(a_parameter)] == pytest.approx(expected_a_lr)
    assert policy.trainable_parameters == expected_parameter_count

    loss = b_parameter.square().sum()
    if expected_trainable:
        loss = loss + a_parameter.square().sum()
    loss.backward()
    policy.optimizer.step()
    if not expected_trainable:
        assert torch.equal(a_parameter.detach(), initial_a)

    with torch.no_grad():
        b_parameter.add_(3.0)
    policy.start_run(None)
    assert torch.equal(a_parameter.detach(), initial_a)
    assert torch.equal(
        b_parameter.detach(),
        policy._initial_adapter["lora_B.default.weight"],
    )
    assert policy.optimizer is None
    assert policy.trainable_parameters == expected_parameter_count


def test_liquid_policy_reapplies_frozen_a_after_adapter_disabled_scoring():
    from contextlib import contextmanager

    import torch

    class ToggleFactors(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 1, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(1, 2, bias=False)}
            )

        def get_output_embeddings(self):
            return torch.nn.Identity()

        @contextmanager
        def disable_adapter(self):
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            try:
                yield
            finally:
                # PEFT restores the adapter's declared trainability on exit.
                for parameter in self.parameters():
                    parameter.requires_grad_(True)

    policy = object.__new__(LiquidLLMPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.sdft_settings = OnlineSDFTSettings(
        lora_a_learning_rate_scale=0.0,
    )
    policy.model = ToggleFactors()
    policy._initial_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.model.named_parameters()
    }
    policy.optimizer = None
    policy.teacher_temperature = 1.0
    policy.last_teacher_assessment = None
    policy.render_prompt = lambda context: context
    policy.render_teacher_prompt = lambda *args: "teacher"
    policy._generate_teacher_assessment = lambda observation: None
    policy._action_logits = lambda prompts: torch.tensor(
        [[0.4, -0.2, 0.7]] * len(prompts),
        dtype=torch.float32,
    )
    policy.start_run(None)
    a_parameter = policy.model.lora_A["default"].weight
    b_parameter = policy.model.lora_B["default"].weight

    policy.base_probs("student")
    assert a_parameter.requires_grad is False
    assert b_parameter.requires_grad is True
    assert policy.trainable_parameters == b_parameter.numel()

    policy.teacher_probs(object())
    assert a_parameter.requires_grad is False
    assert b_parameter.requires_grad is True
    assert policy.trainable_parameters == b_parameter.numel()


@pytest.mark.parametrize(
    ("head_a_scale", "expected_head_a_lr"),
    [
        (None, 5e-4),
        (1.0, 2e-3),
    ],
)
def test_liquid_policy_scales_lm_head_and_body_a_groups_independently(
    head_a_scale,
    expected_head_a_lr,
):
    import torch

    class TinyAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 1, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(1, 2, bias=False)}
            )

    class TinySplitFactors(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = TinyAdapter()
            self.output = TinyAdapter()

        def get_output_embeddings(self):
            return self.output

    policy = object.__new__(LiquidLLMPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.sdft_settings = OnlineSDFTSettings(
        lora_target_modules=("q_proj", "lm_head"),
        lora_a_learning_rate_scale=0.25,
        lm_head_lora_a_learning_rate_scale=head_a_scale,
        lm_head_learning_rate=2e-3,
    )
    policy.model = TinySplitFactors()
    policy._initial_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.model.named_parameters()
    }
    policy.optimizer = None

    policy.start_run(1e-4)

    parameter_lrs = {
        id(parameter): group["lr"]
        for group in policy.optimizer.param_groups
        for parameter in group["params"]
    }
    assert parameter_lrs == {
        id(policy.model.body.lora_A["default"].weight): pytest.approx(2.5e-5),
        id(policy.model.body.lora_B["default"].weight): pytest.approx(1e-4),
        id(policy.model.output.lora_A["default"].weight): pytest.approx(
            expected_head_a_lr
        ),
        id(policy.model.output.lora_B["default"].weight): pytest.approx(2e-3),
    }


@pytest.mark.parametrize(
    ("body_a_scale", "head_a_scale", "body_a_trainable", "head_a_trainable"),
    [
        (0.0, 1.0, False, True),
        (1.0, 0.0, True, False),
    ],
)
def test_liquid_policy_freezes_body_and_lm_head_a_factors_independently(
    body_a_scale,
    head_a_scale,
    body_a_trainable,
    head_a_trainable,
):
    from contextlib import contextmanager

    import torch

    class TinyAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 1, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(1, 2, bias=False)}
            )

    class ToggleSplitFactors(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = TinyAdapter()
            self.output = TinyAdapter()

        def get_output_embeddings(self):
            return self.output

        @contextmanager
        def disable_adapter(self):
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            try:
                yield
            finally:
                for parameter in self.parameters():
                    parameter.requires_grad_(True)

    policy = object.__new__(LiquidLLMPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.sdft_settings = OnlineSDFTSettings(
        lora_target_modules=("q_proj", "lm_head"),
        lora_a_learning_rate_scale=body_a_scale,
        lm_head_lora_a_learning_rate_scale=head_a_scale,
        lm_head_learning_rate=2e-3,
    )
    policy.model = ToggleSplitFactors()
    policy._initial_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.model.named_parameters()
    }
    policy.optimizer = None
    policy.render_prompt = lambda context: context
    policy._action_logits = lambda prompts: torch.tensor(
        [[0.4, -0.2, 0.7]] * len(prompts),
        dtype=torch.float32,
    )

    body_a = policy.model.body.lora_A["default"].weight
    body_b = policy.model.body.lora_B["default"].weight
    head_a = policy.model.output.lora_A["default"].weight
    head_b = policy.model.output.lora_B["default"].weight
    expected_count = body_b.numel() + head_b.numel()
    if body_a_trainable:
        expected_count += body_a.numel()
    if head_a_trainable:
        expected_count += head_a.numel()

    policy.start_run(1e-4)

    optimizer_parameter_ids = {
        id(parameter)
        for group in policy.optimizer.param_groups
        for parameter in group["params"]
    }
    assert body_a.requires_grad is body_a_trainable
    assert head_a.requires_grad is head_a_trainable
    assert (id(body_a) in optimizer_parameter_ids) is body_a_trainable
    assert (id(head_a) in optimizer_parameter_ids) is head_a_trainable
    assert body_b.requires_grad is True
    assert head_b.requires_grad is True
    assert policy.trainable_parameters == expected_count

    policy.base_probs("student")

    assert body_a.requires_grad is body_a_trainable
    assert head_a.requires_grad is head_a_trainable
    assert body_b.requires_grad is True
    assert head_b.requires_grad is True
    assert policy.trainable_parameters == expected_count


@pytest.mark.parametrize(
    ("support", "expected_kind"),
    [
        ([1.0, 0.0, 0.0], "singleton"),
        ([1.0, 1.0, 0.0], "ambiguous"),
    ],
)
def test_liquid_policy_support_likelihood_matches_probability_mass(
    support,
    expected_kind,
):
    import torch

    class TinyLogitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.route_logits = torch.nn.Parameter(
                torch.tensor([0.4, -0.2, 0.7], dtype=torch.float32)
            )

    policy = object.__new__(LiquidLLMPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.sdft_settings = OnlineSDFTSettings(max_grad_norm=100.0)
    policy.model = TinyLogitModel()
    policy.optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
    policy.render_prompt = lambda context: context
    policy._action_logits = lambda prompts: (
        policy.model.route_logits.unsqueeze(0).expand(len(prompts), -1)
    )

    comparison_logits = policy.model.route_logits.detach().clone().requires_grad_()
    if expected_kind == "singleton":
        expected_loss = torch.nn.functional.cross_entropy(
            comparison_logits.unsqueeze(0),
            torch.tensor([0]),
        )
    else:
        probabilities = torch.softmax(comparison_logits, dim=-1)
        expected_loss = -torch.log(probabilities[:2].sum())
    expected_loss.backward()

    actual_loss = policy.update_support([("visible", np.asarray(support))])

    assert actual_loss == pytest.approx(float(expected_loss.detach()))
    assert policy.model.route_logits.grad.detach().numpy() == pytest.approx(
        comparison_logits.grad.detach().numpy()
    )
    assert policy.model.route_logits.grad[2] > 0


def test_liquid_policy_snips_soft_cross_entropy_self_normalizes_weights():
    import torch

    class TwoRowLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.route_logits = torch.nn.Parameter(
                torch.tensor(
                    [[0.4, -0.2, 0.7], [-0.3, 0.8, 0.1]],
                    dtype=torch.float32,
                )
            )

    policy = object.__new__(LiquidLLMPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.sdft_settings = OnlineSDFTSettings(max_grad_norm=100.0)
    policy.model = TwoRowLogits()
    policy.optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
    policy.render_prompt = lambda context: context
    policy._action_logits = lambda prompts: policy.model.route_logits[: len(prompts)]
    targets = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    weights = np.asarray([2.0, 4.0])

    comparison_logits = policy.model.route_logits.detach().clone().requires_grad_()
    per_row = -(
        torch.tensor(targets, dtype=torch.float32)
        * torch.log_softmax(comparison_logits, dim=-1)
    ).sum(-1)
    expected_loss = (2.0 * per_row[0] + 4.0 * per_row[1]) / 6.0
    expected_loss.backward()

    actual_loss = policy.update(
        [("first", targets[0]), ("second", targets[1])],
        sample_weights=weights,
    )

    assert actual_loss == pytest.approx(float(expected_loss.detach()))
    assert policy.model.route_logits.grad.detach().numpy() == pytest.approx(
        comparison_logits.grad.detach().numpy()
    )


def test_liquid_policy_snips_support_loss_self_normalizes_weights():
    import torch

    class TwoRowLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.route_logits = torch.nn.Parameter(
                torch.tensor(
                    [[0.4, -0.2, 0.7], [-0.3, 0.8, 0.1]],
                    dtype=torch.float32,
                )
            )

    policy = object.__new__(LiquidLLMPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.sdft_settings = OnlineSDFTSettings(max_grad_norm=100.0)
    policy.model = TwoRowLogits()
    policy.optimizer = torch.optim.SGD(policy.model.parameters(), lr=0.0)
    policy.render_prompt = lambda context: context
    policy._action_logits = lambda prompts: policy.model.route_logits[: len(prompts)]
    supports = np.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    weights = np.asarray([2.0, 4.0])

    comparison_logits = policy.model.route_logits.detach().clone().requires_grad_()
    log_probs = torch.log_softmax(comparison_logits, dim=-1)
    per_row = torch.stack(
        (-log_probs[0, 0], -torch.logsumexp(log_probs[1, :2], dim=-1))
    )
    expected_loss = (2.0 * per_row[0] + 4.0 * per_row[1]) / 6.0
    expected_loss.backward()

    actual_loss = policy.update_support(
        [("first", supports[0]), ("second", supports[1])],
        sample_weights=weights,
    )

    assert actual_loss == pytest.approx(float(expected_loss.detach()))
    assert policy.model.route_logits.grad.detach().numpy() == pytest.approx(
        comparison_logits.grad.detach().numpy()
    )


@pytest.mark.parametrize(
    ("batch", "message"),
    [
        ([], "cannot be empty"),
        ([("visible", np.array([1.0, 0.0]))], "one value per route"),
        ([("visible", np.array([1.0, 0.5, 0.0]))], "binary"),
        ([("visible", np.array([0.0, 0.0, 0.0]))], "cannot be empty"),
        ([("visible", np.array([1.0, 1.0, 1.0]))], "full support"),
    ],
)
def test_liquid_policy_support_likelihood_rejects_invalid_masks(batch, message):
    import torch

    policy = object.__new__(LiquidLLMPolicy)
    policy.torch = torch
    policy.device = torch.device("cpu")
    policy.optimizer = object()
    policy.model = torch.nn.Linear(1, 1)
    policy.render_prompt = lambda context: context

    with pytest.raises(ValueError, match=message):
        policy.update_support(batch)


def test_liquid_policy_constructor_accepts_adapter_architecture_settings():
    parameters = signature(LiquidLLMPolicy).parameters

    assert parameters["sdft_settings"].default is DEFAULT_SDFT_SETTINGS
    assert "sdft_settings: OnlineSDFTSettings" in getsource(
        LiquidLLMPolicy.__init__
    )


class AssessmentTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, input_tokens, decoded, *, decode_error=None):
        self.input_tokens = input_tokens
        self.decoded = decoded
        self.decode_error = decode_error
        self.messages = None

    def __len__(self):
        return 128

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        self.messages = [dict(message) for message in messages]
        return "\n".join(message["content"] for message in messages)

    def __call__(self, prompt, **kwargs):
        import torch

        del prompt, kwargs
        input_ids = torch.ones((1, self.input_tokens), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(self, token_ids, **kwargs):
        del token_ids, kwargs
        if self.decode_error is not None:
            raise self.decode_error
        return self.decoded


class AssessmentModel:
    def __init__(self, *, generated_token_id=2, generation_error=None):
        self.generated_token_id = generated_token_id
        self.generation_error = generation_error

    def generate(self, **kwargs):
        import torch

        self.generate_kwargs = kwargs
        if self.generation_error is not None:
            raise self.generation_error
        input_ids = kwargs["input_ids"]
        suffix = torch.full(
            (input_ids.shape[0], kwargs["max_new_tokens"]),
            self.generated_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, suffix), dim=1)


def assessment_policy(
    input_tokens,
    decoded,
    reasoning_tokens=40,
    *,
    generated_token_id=2,
    generation_error=None,
    decode_error=None,
):
    import torch

    policy = object.__new__(LiquidLLMPolicy)
    policy.prompt_style = "compact"
    policy.tokenizer = AssessmentTokenizer(
        input_tokens,
        decoded,
        decode_error=decode_error,
    )
    policy.model = AssessmentModel(
        generated_token_id=generated_token_id,
        generation_error=generation_error,
    )
    policy.device = torch.device("cpu")
    policy.teacher_reasoning_tokens = reasoning_tokens
    return policy


def teacher_observation_for_assessment():
    from online_sdft.environment import TeacherObservation

    return TeacherObservation(
        context=(
            "The notification title is Deployment approval needed. The "
            "message says Review the release. This is a manager notification "
            "that arrived at 10:30 local time during the weekday period."
        ),
        evidence="The user opened it from the digest 120 minutes later.",
        observed_user_selection="LATER",
    )


def test_teacher_generation_budget_counts_input_plus_requested_output():
    policy = assessment_policy(
        input_tokens=PROMPT_TOKEN_BUDGET - 40,
        decoded="The digest open leaves immediate need uncertain.",
        reasoning_tokens=40,
    )
    observation = teacher_observation_for_assessment()

    assessment = policy._generate_teacher_assessment(observation)
    assert assessment == "The digest open leaves immediate need uncertain."
    assert policy.model.generate_kwargs["max_new_tokens"] == 40
    assert policy.model.generate_kwargs["use_cache"] is True

    policy.tokenizer.input_tokens += 1
    with pytest.raises(
        ValueError,
        match=r"input=729, requested=40, budget=768",
    ):
        policy._generate_teacher_assessment(observation)


def test_teacher_assessment_falls_back_if_generation_forces_an_answer():
    policy = assessment_policy(
        input_tokens=32,
        decoded="The final route is B because the digest was opened.",
    )
    assessment = policy._generate_teacher_assessment(
        teacher_observation_for_assessment()
    )
    assert assessment == ASSESSMENT_FALLBACK
    assert len(assessment.split()) == 10
    assert "final route" not in assessment.lower()


@pytest.mark.parametrize(
    "generated_assessment",
    [
        "COPY EXACTLY: B.",
        "The Correct Output is LATER.",
        "Do Not Add Explanation.",
        "Output only C.",
        "Respond with ARCHIVE.",
        "Select option B.",
        "INTERRUPT seems plausible because the request is time-sensitive.",
        "LATER may fit, although immediate delivery remains possible.",
        "The evidence favors ARCHIVE while alternatives remain possible.",
        "Option B is plausible.",
        "A seems most likely.",
        "The user seems busy.",
        "The user's busyness is high.",
        "The deadline appears close.",
        "Urgency is high.",
        "Affinity is probably high.",
        "The interruption filter may block it.",
        "The interruption_filter may block it.",
    ],
)
def test_teacher_assessment_rejects_directives_choices_and_hidden_fields(
    generated_assessment,
):
    policy = assessment_policy(input_tokens=32, decoded=generated_assessment)

    assessment = policy._generate_teacher_assessment(
        teacher_observation_for_assessment()
    )

    assert assessment == ASSESSMENT_FALLBACK


@pytest.mark.parametrize(
    "generated_assessment",
    [
        "The digest open leaves immediate timing uncertain.",
        "Both INTERRUPT and LATER remain possible.",
        "B is plausible, but A is equally plausible.",
        "The callback supports LATER but cannot rule out INTERRUPT.",
        "The output was recorded correctly and remains partial evidence.",
    ],
)
def test_teacher_assessment_keeps_neutral_uncertainty_and_comparisons(
    generated_assessment,
):
    policy = assessment_policy(input_tokens=32, decoded=generated_assessment)

    assessment = policy._generate_teacher_assessment(
        teacher_observation_for_assessment()
    )

    assert assessment == generated_assessment


def test_teacher_assessment_fallback_is_neutral_and_ten_words():
    assert ASSESSMENT_FALLBACK == (
        "The callback is partial evidence, so unobserved alternatives remain "
        "possible."
    )
    assert len(ASSESSMENT_FALLBACK.split()) == 10
    assert not re.search(r"\b(?:A|B|C)\b", ASSESSMENT_FALLBACK)
    assert not any(
        route.lower() in ASSESSMENT_FALLBACK.lower()
        for route in ("INTERRUPT", "LATER", "ARCHIVE")
    )


def test_empty_teacher_assessment_uses_neutral_assessment_block():
    policy = assessment_policy(input_tokens=32, decoded="  \n")
    observation = teacher_observation_for_assessment()
    assessment = policy._generate_teacher_assessment(observation)
    assert assessment == ASSESSMENT_FALLBACK
    rendered = policy.render_teacher_prompt(observation, assessment=assessment)
    assert "Teacher evidence assessment:" in rendered
    assert ASSESSMENT_FALLBACK in rendered


@pytest.mark.parametrize(
    ("generated_token_id", "generation_error", "decode_error"),
    [
        (128, None, None),
        (-1, None, None),
        (2, RuntimeError("generation failed"), None),
        (2, None, RuntimeError("decode failed")),
    ],
)
def test_teacher_assessment_failures_use_neutral_fallback(
    generated_token_id,
    generation_error,
    decode_error,
):
    policy = assessment_policy(
        input_tokens=32,
        decoded="The callback gives partial evidence.",
        generated_token_id=generated_token_id,
        generation_error=generation_error,
        decode_error=decode_error,
    )

    assessment = policy._generate_teacher_assessment(
        teacher_observation_for_assessment()
    )

    assert assessment == ASSESSMENT_FALLBACK
    assert len(assessment.split()) == 10


def test_history_matching_instruction_requires_actual_history():
    class EchoTokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            del kwargs
            return "\n".join(message["content"] for message in messages)

    policy = prompt_policy(prompt_style="interaction_match")
    policy.tokenizer = EchoTokenizer()
    without_history = policy.render_prompt("category=manager")
    with_history = policy.render_prompt(
        "category=manager",
        [{
            "context": "category=manager",
            "executed_action": "LATER",
            "eventual_user_action": "OPENED_DIGEST",
            "observed_user_selection": "LATER",
        }],
    )
    assert "known observed selection is useful" not in without_history
    assert "known observed selection is useful" in with_history


def test_relevance_template_tags_history_without_changing_base_prompt():
    class EchoTokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            del kwargs
            return "\n".join(
                f"{message['role']}: {message['content']}"
                for message in messages
            )

    policy = prompt_policy(prompt_style="history_relevance")
    policy.tokenizer = EchoTokenizer()
    current = (
        "The notification title is Approval needed. The message says Review "
        "the release. This is a manager notification that arrived at 10:30 "
        "local time during the weekday period."
    )
    without_history = policy.render_prompt(current)
    with_history = policy.render_prompt(
        current,
        [
            {
                "context": (
                    "The notification title is Earlier approval. The message "
                    "says Review the plan. This is a manager notification that "
                    "arrived at 09:45 local time during the weekday period."
                ),
                "executed_action": "LATER",
                "eventual_user_action": "OPENED_DIGEST",
                "observed_user_selection": "LATER",
                "delay_minutes": 120,
            },
            {
                "context": (
                    "The notification title is Member offer. The message says "
                    "Save today. This is a promo notification that arrived at "
                    "09:50 local time during the weekday period."
                ),
                "executed_action": "ARCHIVE",
                "eventual_user_action": "DELETED_FROM_DIGEST",
                "observed_user_selection": "ARCHIVE",
                "delay_minutes": 120,
            },
        ],
    )
    assert "EXACT_MATCH" not in without_history
    assert "DIFFERENT_CONTEXT" not in without_history
    assert "EXACT_MATCH" in with_history
    assert "DIFFERENT_CONTEXT" in with_history
    assert "NEW DECISION" in with_history
    assert "do not repeat the last route" in with_history
    assert policy._context_field(current, "category") == "manager"
    assert policy._context_field(current, "regime") == "weekday"


def test_llm_api_accepts_visible_text_not_full_event():
    assert MODEL_ID == "LiquidAI/LFM2.5-230M"
    parameters = signature(LiquidLLMPolicy.render_prompt).parameters
    assert "event" not in parameters
    assert "context" in parameters


def test_methods_module_has_no_reward_or_oracle_implementation():
    source = getsource(methods_module)
    assert "oracle_utilities" not in source
    assert "factual_feedback" not in source
    assert "teacher_distribution(" not in source


def test_teacher_prompt_uses_partial_evidence_without_answer_forcing():
    from online_sdft.environment import TeacherObservation

    policy = prompt_policy()
    observation = TeacherObservation(
        context=(
            "The notification title is Approval needed before deployment. "
            "The message says Maya needs a decision within 20 minutes. This "
            "is a manager notification that arrived at 10:30 local time "
            "during the weekday period. Its on-device importance score is "
            "0.91 out of 1."
        ),
        evidence=(
            "The router placed the notification in a later digest. "
            "The user opened it from the digest 120 minutes later. This "
            "behavior revealed LATER as the observed user selection on the "
            "executed surface."
        ),
        observed_user_selection="LATER",
    )
    rendered = policy.render_teacher_prompt(observation)
    assert "Choose a route for a similar future notification" in rendered
    assert "No hidden label or unchosen outcome is available" in rendered
    assert (
        "A digest open after LATER leaves INTERRUPT versus LATER unresolved"
        in rendered
    )
    assert "UNKNOWN supports no route" in rendered
    assert "Notification:\nThe notification title is Approval needed" in rendered
    assert "This is a manager notification" in rendered
    assert "Observed callback:\nThe router placed" in rendered
    assert "revealed LATER as the observed user selection" in rendered
    assert rendered.strip().endswith("Route:")
    lowered = rendered.lower()
    assert "copy exactly" not in lowered
    assert "correct output" not in lowered
    assert "do not add explanation" not in lowered
    assert "busy" not in lowered
    assert "interruption_filter" not in lowered
    for code in ACTION_CODES:
        lowered_code = code.lower()
        assert f"output only {lowered_code}" not in lowered
        assert f"answer {lowered_code}" not in lowered
        assert f"same as {lowered_code}" not in lowered
    scoring_messages = policy.tokenizer.messages
    assert [message["role"] for message in scoring_messages] == [
        "system",
        "user",
    ]
    scoring_user = scoring_messages[-1]["content"]
    assert len(scoring_user) - len(observation.context) - len(
        observation.evidence
    ) <= 60

    assessment_rendered = policy.render_teacher_assessment_prompt(observation)
    assert "Observed callback:" in assessment_rendered
    assert assessment_rendered.strip().endswith("Assessment:")
    assert "Write one short evidence assessment" not in assessment_rendered
    assessment_user = policy.tokenizer.messages[-1]["content"]
    assert len(assessment_user) - len(observation.context) - len(
        observation.evidence
    ) <= 60


def test_teacher_prompt_rejects_demonstrations_outside_one_trajectory_contract():
    from online_sdft.environment import TeacherObservation

    policy = prompt_policy()
    observation = TeacherObservation(
        context="category=manager",
        evidence="executed_route=LATER; observed_user_selection=LATER",
        observed_user_selection="LATER",
    )
    with pytest.raises(ValueError, match="exactly one completed trajectory"):
        policy.render_teacher_prompt(
            observation,
            [{"context": "past", "evidence": "past callback"}],
        )


def test_unknown_selection_is_not_converted_to_archive_in_teacher_prompt():
    from online_sdft.environment import TeacherObservation

    policy = prompt_policy()
    rendered = policy.render_teacher_prompt(
        TeacherObservation(
            context="category=promo",
            evidence=(
                "The router archived the item without delivering a "
                "notification. No delivered notification surface revealed "
                "a user choice. The user's preferred route remains unknown."
            ),
        )
    )
    assert "preferred route remains unknown" in rendered
    assert "UNKNOWN supports no route" in rendered
    assert "Observed callback:" in rendered
    assert "UNKNOWN = ARCHIVE" not in rendered


def test_online_sdft_uses_soft_targets_replay_without_prompting_by_default():
    from online_sdft.config import SDFT_UPDATE_STEPS
    from online_sdft.environment import TeacherObservation

    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(policy)
    assert agent.update_steps == SDFT_UPDATE_STEPS
    teacher_view = TeacherObservation(
        context="visible",
        evidence="executed_route=LATER; observed_event=OPENED_DIGEST",
    )
    for index in range(5):
        observation = visible_observation(f"visible-{index}")
        decision = agent.action_probs(observation)
        agent.observe(
            observation,
            action=1,
            teacher_distribution=np.array([0.1, 0.7, 0.2]),
            teacher_action=1,
            feedback={
                "action_taken": "LATER",
                "outcome": "OPENED_DIGEST",
                "observed_user_selection": "LATER",
            },
            rng=np.random.default_rng(index),
            teacher_observation=teacher_view,
            decision_distribution=decision,
        )
    assert agent.prompt_examples(StudentObservation("visible", np.ones(3))) == []
    assert len(policy.updates) == 2 * SDFT_UPDATE_STEPS
    assert agent.online_update_count == 2 * SDFT_UPDATE_STEPS
    assert agent.settings is DEFAULT_SDFT_SETTINGS


def test_online_sdft_fifo_replay_prompt_uses_only_reliable_factual_labels():
    settings = OnlineSDFTSettings(
        replay_size=3,
        replay_prompt_examples=2,
        batch_size=1,
        warmup_examples=1,
        update_steps=1,
    )
    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(policy, settings=settings)
    rows = (
        (
            "reliable interrupt",
            0,
                {
                    "action_taken": "INTERRUPT",
                    "outcome": "OPENED_IMMEDIATELY",
                    "delay_minutes": 0,
                "observed_user_selection": "INTERRUPT",
            },
        ),
        (
            "ambiguous digest",
            1,
            {
                "action_taken": "LATER",
                "outcome": "OPENED_DIGEST",
                "delay_minutes": 240,
                "observed_user_selection": "LATER",
            },
        ),
        (
            "reliable archive",
            1,
            {
                "action_taken": "LATER",
                    "outcome": "DELETED_FROM_DIGEST",
                "delay_minutes": 240,
                "observed_user_selection": "ARCHIVE",
            },
        ),
    )
    for index, (text, action, feedback) in enumerate(rows):
        observation = visible_observation(text)
        decision = agent.action_probs(observation)
        agent.observe(
            observation,
            action=action,
            # Deliberately favor LATER: the prompt label must remain factual.
            teacher_distribution=np.array([0.05, 0.90, 0.05]),
            teacher_action=1,
            feedback=feedback,
            rng=np.random.default_rng(index),
            decision_distribution=decision,
        )

    examples = agent.prompt_examples(StudentObservation("current", np.ones(3)))
    assert [example["context"] for example in examples] == [
        "reliable interrupt",
        "reliable archive",
    ]
    assert [example["observed_user_selection"] for example in examples] == [
        "INTERRUPT",
        "ARCHIVE",
    ]
    assert agent.last_prompt_examples_used == 2
    assert not any(
        forbidden in example
        for example in examples
        for forbidden in (
            "soft_target",
            "teacher_probs",
            "reward",
            "gold_action",
            "oracle_utility",
            "counterfactual",
        )
    )

    # The bounded replay is the prompt memory: eviction removes old lessons.
    agent.replay_size = 2
    agent.replay = agent.replay[-2:]
    examples = agent.prompt_examples(StudentObservation("current", np.ones(3)))
    assert [example["context"] for example in examples] == ["reliable archive"]


def test_online_sdft_waits_for_replay_warmup_before_first_update():
    from online_sdft.environment import TeacherObservation

    settings = OnlineSDFTSettings(
        replay_size=8,
        batch_size=4,
        warmup_examples=3,
        update_steps=1,
    )
    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(policy, settings=settings)
    teacher_view = TeacherObservation(
        context="visible",
        evidence="executed_route=INTERRUPT; observed_event=OPENED_IMMEDIATELY",
    )
    for index in range(2):
        observation = visible_observation(f"warmup-{index}")
        decision = agent.action_probs(observation)
        agent.observe(
            observation,
            action=0,
            teacher_distribution=np.array([0.7, 0.2, 0.1]),
            teacher_action=0,
            feedback={
                "action_taken": "INTERRUPT",
                "outcome": "OPENED_IMMEDIATELY",
                "observed_user_selection": "INTERRUPT",
            },
            rng=np.random.default_rng(index),
            teacher_observation=teacher_view,
            decision_distribution=decision,
        )
        assert not policy.updates

    observation = visible_observation("first-update")
    decision = agent.action_probs(observation)
    agent.observe(
        observation,
        action=0,
        teacher_distribution=np.array([0.7, 0.2, 0.1]),
        teacher_action=0,
        feedback={
            "action_taken": "INTERRUPT",
            "outcome": "OPENED_IMMEDIATELY",
            "observed_user_selection": "INTERRUPT",
        },
        rng=np.random.default_rng(2),
        teacher_observation=teacher_view,
        decision_distribution=decision,
    )
    assert len(policy.updates) == 1
    assert len(agent.replay) == 3
    assert agent.online_update_count == 1
    assert agent.last_observation_update_count == 1


def test_online_sdft_delegates_soft_target_replay_to_lora_policy_update():
    settings = OnlineSDFTSettings(
        learning_rate=1e-3,
        replay_size=8,
        batch_size=1,
        update_steps=1,
        warmup_examples=1,
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules=("q_proj", "v_proj"),
        lora_layers_to_transform=(10, 12),
        reasoning_tokens=0,
    )
    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(policy, settings=settings)
    observation = visible_observation("urgent review", category=1, importance=0.9)

    before = agent.action_probs(observation)
    assert before == pytest.approx([0.2, 0.5, 0.3])
    assert policy.learning_rate == settings.learning_rate
    assert not policy.updates

    agent.observe(
        observation,
        action=0,
        teacher_distribution=np.array([0.7, 0.2, 0.1]),
        teacher_action=0,
        feedback={
            "action_taken": "INTERRUPT",
            "outcome": "OPENED_IMMEDIATELY",
            "observed_user_selection": "INTERRUPT",
        },
        rng=np.random.default_rng(0),
        decision_distribution=before,
    )

    assert len(policy.updates) == 1
    context, target = policy.updates[0][0]
    assert context == observation.text
    assert target == pytest.approx(agent.replay[0][1])
    assert agent.online_update_count == 1

    # Replay is training-only and is not inserted into serving prompts.
    agent.replay.clear()
    assert agent.action_probs(observation) == pytest.approx(before)


def test_liquid_policy_uses_peft_lora_and_one_adapter_disabled_teacher_model():
    source = getsource(LiquidLLMPolicy)
    assert 'self.device.type == "cuda"' in source
    assert "torch.float16" in source
    assert "else torch.float32" in source
    assert ".logits[:, -1, :].float()" in source
    assert "LoraConfig" in source
    assert "get_peft_model" in source
    assert "init_lora_weights=True" in source
    assert "ensure_weight_tying=False" in source
    assert "AdamW" in source
    assert "weight_decay=self.sdft_settings.optimizer_weight_decay" in source
    assert ".backward(" in source
    assert "max_norm=self.sdft_settings.max_grad_norm" in source
    assert "with self.model.disable_adapter():" in source
    assert source.count("Lfm2ForCausalLM.from_pretrained") == 1
    assert "merge_and_unload" not in source
    assert "TEACHER_TEMPERATURE" in source
    assert "def teacher_probs" in source
