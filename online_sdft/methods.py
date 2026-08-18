"""LLM policy and the compared online-learning methods.

Every method consumes :class:`StudentObservation`, which contains only the
natural-language serving view. For SDFT, the same Liquid network later acts as
a hindsight teacher over the original serving view and the factual callback
produced by the executed route. It never receives a counterfactual
outcome, benchmark reward, or evaluator label.
Environment execution and rewards are intentionally not simulated in this
module; REINFORCE receives the executed route's scalar reward, or applies an
explicit arm-specific map to its factual outcome, only after acting.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Real
import re
import string
from typing import Any, Protocol

import numpy as np

from .config import (
    ACTION_CODES,
    ACTIONS,
    CATEGORIES,
    EXPLORATION_EPSILON,
    ICL_K,
    INTERRUPT_PROMPT_SUFFIXES,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_LAYERS_TO_TRANSFORM,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_ID,
    ONLINE_BATCH_SIZE,
    PROMPT_STYLE,
    PROMPT_STYLES,
    PROMPT_TOKEN_BUDGET,
    RAG_K,
    RAG_TEXT_WEIGHT,
    REINFORCE_BATCH_SIZE,
    REINFORCE_BASELINE_STEP,
    REINFORCE_ENTROPY_COEF,
    REINFORCE_LR,
    REINFORCE_MAX_GRAD_NORM,
    REINFORCE_TRAINING_OUTCOME_REWARDS,
    REPLAY_SIZE,
    RFT_CANDIDATE_COUNT,
    RFT_LR,
    RFT_SAMPLING_MODE,
    RFT_SAMPLING_TEMPERATURE,
    SDFT_BATCH_SIZE,
    SDFT_LR,
    SDFT_MAX_GRAD_NORM,
    SDFT_OPTIMIZER_WEIGHT_DECAY,
    SDFT_REPLAY_SIZE,
    SDFT_UPDATE_STEPS,
    SDFT_WARMUP_EXAMPLES,
    STUDENT_TEMPERATURE,
    SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    TEACHER_REASONING_SYSTEM_PROMPT,
    TEACHER_TEMPERATURE,
)
from .environment import StudentObservation, TeacherObservation
from .privilege import FactualCallback, narrative_mobile_teacher_evidence


ASSESSMENT_FALLBACK = (
    "The callback is partial evidence, so unobserved alternatives remain "
    "possible."
)

_ASSESSMENT_ACTION = r"(?:INTERRUPT|LATER|ARCHIVE|(?-i:[ABC]))"
_ASSESSMENT_HIDDEN_FIELD_PATTERN = re.compile(
    r"\b(?:busy|busyness|deadline|urgency|affinity)\b|"
    r"\binterruption(?:[\s_-]+)filter\b",
    re.IGNORECASE,
)
_ASSESSMENT_DIRECTIVE_PATTERN = re.compile(
    r"\b(?:copy\s+exactly|correct\s+(?:output|answer|route|choice)|"
    r"do\s+not\s+add\s+(?:an?\s+)?explanation|output\s+only|"
    r"respond\s+(?:only\s+)?with|return\s+(?:only\s+)?"
    + _ASSESSMENT_ACTION
    + r")\b",
    re.IGNORECASE,
)
_ASSESSMENT_ANSWER_PATTERN = re.compile(
    r"(?:^|\b)(?:answer|choose|select|pick|recommend)\s+"
    r"(?:is\s+|route\s+|option\s+)?"
    + _ASSESSMENT_ACTION
    + r"\b|"
    r"\b(?:final|correct|best|preferred|recommended)\s+"
    r"(?:route|answer|output|choice|option)\s*(?:is|:)?\s*"
    + _ASSESSMENT_ACTION
    + r"\b|^[abc][.!]?$",
    re.IGNORECASE,
)
_ASSESSMENT_PREFERENCE_PATTERN = re.compile(
    r"\b"
    + _ASSESSMENT_ACTION
    + r"\s+(?:(?:is|seems|appears|looks)\s+"
    r"(?:the\s+)?(?:best|correct|preferred|recommended|most\s+likely|"
    r"plausible|appropriate|suitable|right)|"
    r"(?:may|could)\s+(?:be\s+)?(?:the\s+)?(?:most\s+likely|plausible|"
    r"appropriate|suitable|right|fit|work))\b|"
    r"\b(?:evidence|callback|notification|assessment|behavior)\s+"
    r"(?:strongly\s+)?(?:favors|favours|points\s+to|indicates|recommends)\s+"
    r"(?:route\s+|option\s+)?"
    + _ASSESSMENT_ACTION
    + r"\b",
    re.IGNORECASE,
)
_ASSESSMENT_DECISIVE_PATTERN = re.compile(
    r"\b(?:best|correct|preferred|recommended|most\s+likely|right)\b",
    re.IGNORECASE,
)


def _assessment_action_mentions(assessment: str) -> set[str]:
    """Return explicit route names/codes without treating the article 'a' as A."""
    mentions = {
        route.upper()
        for route in re.findall(
            r"\b(?:INTERRUPT|LATER|ARCHIVE)\b",
            assessment,
            flags=re.IGNORECASE,
        )
    }
    mentions.update(re.findall(r"\b[ABC]\b", assessment))
    return mentions


def _assessment_crosses_boundary(assessment: str) -> bool:
    """Reject generated text that would widen or force the scoring prompt."""
    forbidden_phrases = (
        "route code",
        "final route",
        "correct route",
        "best route is",
        "gold action",
        "scalar reward",
        "counterfactual",
        "oracle utility",
    )
    lower = assessment.lower()
    if any(phrase in lower for phrase in forbidden_phrases):
        return True
    if _ASSESSMENT_HIDDEN_FIELD_PATTERN.search(assessment):
        return True
    if _ASSESSMENT_DIRECTIVE_PATTERN.search(assessment):
        return True
    if _ASSESSMENT_ANSWER_PATTERN.search(assessment):
        return True
    preference = _ASSESSMENT_PREFERENCE_PATTERN.search(assessment)
    if preference is None:
        return False
    mentions = _assessment_action_mentions(assessment)
    return len(mentions) < 2 or bool(
        _ASSESSMENT_DECISIVE_PATTERN.search(preference.group(0))
    )


@dataclass(frozen=True)
class OnlineSDFTSettings:
    """Configured LoRA, causal-update, and teacher settings for Online-SDFT."""

    learning_rate: float = SDFT_LR
    replay_size: int = SDFT_REPLAY_SIZE
    replay_prompt_examples: int = 0
    batch_size: int = SDFT_BATCH_SIZE
    update_steps: int = SDFT_UPDATE_STEPS
    warmup_examples: int = SDFT_WARMUP_EXAMPLES
    lora_rank: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    lora_target_modules: tuple[str, ...] = LORA_TARGET_MODULES
    lora_layers_to_transform: tuple[int, ...] | None = LORA_LAYERS_TO_TRANSFORM
    lora_a_learning_rate_scale: float = 1.0
    lm_head_lora_a_learning_rate_scale: float | None = None
    optimizer_weight_decay: float = SDFT_OPTIMIZER_WEIGHT_DECAY
    optimizer_beta1: float = 0.9
    max_grad_norm: float = SDFT_MAX_GRAD_NORM
    lm_head_learning_rate: float | None = None
    ambiguous_replay_group_weight: float = 0.05
    teacher_temperature: float = TEACHER_TEMPERATURE
    reasoning_tokens: int = 0
    target_mode: str = "causal_fusion"
    reliable_teacher_weight: float = 0.05
    reliable_decision_weight: float = 0.05
    reliable_behavior_weight: float = 0.90
    ambiguous_teacher_weight: float = 0.0
    ambiguous_decision_weight: float = 1.0
    ambiguous_behavior_weight: float = 0.0
    ambiguous_projection: str = "causal_support"
    replay_strategy: str = "selection_balanced"
    replay_recency_half_life: float | None = None
    ambiguous_update_mode: str = "immediate"
    force_newest_every_step: bool = True
    base_kl_weight: float = 0.0
    behavior_mode: str = "epsilon_greedy"
    behavior_epsilon: float = EXPLORATION_EPSILON
    behavior_epsilon_half_life: float | None = None
    exploration_taper_start_step: int | None = None
    exploration_taper_half_life: float | None = None
    archive_probe_mix: float = 0.0
    archive_policy_min_feedback: float = 0.0
    interrupt_probe_mix: float = 0.0
    interrupt_probe_half_life: float | None = None
    interrupt_probe_max_confidence: float = 1.0
    propensity_weight_mode: str = "none"
    propensity_weight_cap: float = 4.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.teacher_temperature <= 0:
            raise ValueError("learning rate and teacher temperature must be positive")
        integer_fields = (self.lora_rank, self.lora_alpha)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_fields
        ):
            raise ValueError("LoRA rank and alpha must be positive integers")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        raw_targets = self.lora_target_modules
        targets = (raw_targets,) if isinstance(raw_targets, str) else tuple(raw_targets)
        if not targets or any(not isinstance(name, str) or not name for name in targets):
            raise ValueError("LoRA target modules must be non-empty strings")
        if len(set(targets)) != len(targets):
            raise ValueError("LoRA target modules must be unique")
        object.__setattr__(self, "lora_target_modules", targets)
        raw_layers = self.lora_layers_to_transform
        if raw_layers is not None:
            layers = (raw_layers,) if isinstance(raw_layers, int) else tuple(raw_layers)
            if not layers:
                raise ValueError("LoRA layer indices cannot be empty")
            if any(
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or layer < 0
                for layer in layers
            ):
                raise ValueError("LoRA layer indices must be non-negative integers")
            if len(set(layers)) != len(layers):
                raise ValueError("LoRA layer indices must be unique")
            object.__setattr__(self, "lora_layers_to_transform", layers)
        if self.optimizer_weight_decay < 0:
            raise ValueError("LoRA optimizer weight decay cannot be negative")
        if (
            isinstance(self.lora_a_learning_rate_scale, bool)
            or not np.isfinite(self.lora_a_learning_rate_scale)
            or not 0.0 <= self.lora_a_learning_rate_scale <= 1.0
        ):
            raise ValueError(
                "LoRA A learning-rate scale must be finite and in [0, 1]"
            )
        if self.lm_head_lora_a_learning_rate_scale is not None and (
            isinstance(self.lm_head_lora_a_learning_rate_scale, bool)
            or not np.isfinite(self.lm_head_lora_a_learning_rate_scale)
            or not 0.0 <= self.lm_head_lora_a_learning_rate_scale <= 1.0
        ):
            raise ValueError(
                "LM-head LoRA A learning-rate scale must be finite and in [0, 1]"
            )
        if (
            self.lm_head_lora_a_learning_rate_scale is not None
            and "lm_head" not in targets
        ):
            raise ValueError(
                "LM-head LoRA A learning-rate scale requires an LM-head adapter"
            )
        if (
            isinstance(self.optimizer_beta1, bool)
            or not np.isfinite(self.optimizer_beta1)
            or not 0.0 <= self.optimizer_beta1 < 1.0
        ):
            raise ValueError("LoRA optimizer beta1 must be finite and in [0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("LoRA maximum gradient norm must be positive")
        if self.lm_head_learning_rate is not None and (
            isinstance(self.lm_head_learning_rate, bool)
            or not np.isfinite(self.lm_head_learning_rate)
            or self.lm_head_learning_rate <= 0.0
        ):
            raise ValueError("LM-head learning rate must be finite and positive")
        if (
            self.lm_head_learning_rate is not None
            and "lm_head" not in targets
        ):
            raise ValueError("LM-head learning rate requires an LM-head adapter")
        if not 0.0 < self.ambiguous_replay_group_weight <= 1.0:
            raise ValueError(
                "ambiguous replay group weight must be in (0, 1]"
            )
        if min(
            self.replay_size,
            self.batch_size,
            self.update_steps,
            self.warmup_examples,
        ) <= 0:
            raise ValueError(
                "replay, batch, update, and warmup counts must be positive"
            )
        if self.batch_size > self.replay_size:
            raise ValueError("SDFT batch size cannot exceed replay size")
        if not 0 <= self.replay_prompt_examples <= self.replay_size:
            raise ValueError(
                "SDFT replay prompt examples must be between zero and replay size"
            )
        if self.warmup_examples > self.replay_size:
            raise ValueError("SDFT warmup cannot exceed replay size")
        if self.reasoning_tokens < 0:
            raise ValueError("reasoning token count cannot be negative")
        if self.reasoning_tokens > 64:
            raise ValueError("reasoning token count cannot exceed 64")
        if self.target_mode not in {
            "teacher_only",
            "causal_fusion",
            "support_likelihood",
        }:
            raise ValueError("unknown SDFT target mode")
        if self.ambiguous_projection not in {"none", "causal_support"}:
            raise ValueError("unknown ambiguous projection mode")
        if self.replay_strategy not in {
            "uniform",
            "selection_balanced",
        }:
            raise ValueError("unknown SDFT replay strategy")
        if self.replay_recency_half_life is not None:
            if (
                isinstance(self.replay_recency_half_life, bool)
                or not isinstance(self.replay_recency_half_life, Real)
                or not np.isfinite(self.replay_recency_half_life)
                or self.replay_recency_half_life <= 0.0
            ):
                raise ValueError(
                    "replay recency half-life must be finite and positive"
                )
            if self.replay_strategy != "selection_balanced":
                raise ValueError(
                    "replay recency requires selection-balanced replay"
                )
        if self.ambiguous_update_mode not in {"immediate", "skip", "defer"}:
            raise ValueError("unknown ambiguous update mode")
        if not isinstance(self.force_newest_every_step, bool):
            raise ValueError("force-newest setting must be boolean")
        if self.base_kl_weight < 0:
            raise ValueError("fixed-base KL weight cannot be negative")
        if self.base_kl_weight and self.target_mode == "support_likelihood":
            raise ValueError(
                "fixed-base KL is incompatible with support likelihood"
            )
        if self.base_kl_weight and self.replay_prompt_examples:
            raise ValueError(
                "fixed-base KL requires pure parametric replay prompts"
            )
        if self.behavior_mode not in {
            "epsilon_greedy",
            "policy_sampling",
            "archive_policy_sampling",
            "archive_uniform_probe",
            "archive_policy_feedback_floor",
            "uncertainty_interrupt_probe",
        }:
            raise ValueError("unknown Online-SDFT behavior mode")
        if (
            isinstance(self.behavior_epsilon, bool)
            or not isinstance(self.behavior_epsilon, Real)
            or not np.isfinite(self.behavior_epsilon)
            or not 0.0 <= self.behavior_epsilon <= 1.0
        ):
            raise ValueError(
                "Online-SDFT behavior epsilon must be finite and in [0, 1]"
            )
        if self.behavior_epsilon_half_life is not None and (
            isinstance(self.behavior_epsilon_half_life, bool)
            or not isinstance(self.behavior_epsilon_half_life, Real)
            or not np.isfinite(self.behavior_epsilon_half_life)
            or self.behavior_epsilon_half_life <= 0.0
        ):
            raise ValueError(
                "behavior epsilon half-life must be finite and positive"
            )
        if self.exploration_taper_start_step is not None and (
            isinstance(self.exploration_taper_start_step, bool)
            or not isinstance(self.exploration_taper_start_step, int)
            or self.exploration_taper_start_step <= 0
        ):
            raise ValueError(
                "exploration taper start step must be a positive integer"
            )
        if self.exploration_taper_half_life is not None and (
            isinstance(self.exploration_taper_half_life, bool)
            or not isinstance(self.exploration_taper_half_life, Real)
            or not np.isfinite(self.exploration_taper_half_life)
            or self.exploration_taper_half_life <= 0.0
        ):
            raise ValueError(
                "exploration taper half-life must be finite and positive"
            )
        if (
            self.exploration_taper_start_step is None
            or self.exploration_taper_half_life is None
        ) and (
            self.exploration_taper_start_step is not None
            or self.exploration_taper_half_life is not None
        ):
            raise ValueError(
                "exploration taper start step and half-life must be configured "
                "together"
            )
        if (
            self.exploration_taper_start_step is not None
            and self.behavior_mode
            not in {
                "epsilon_greedy",
                "archive_uniform_probe",
                "uncertainty_interrupt_probe",
            }
        ):
            raise ValueError(
                "exploration taper requires an epsilon/probe behavior mode"
            )
        if (
            isinstance(self.archive_probe_mix, bool)
            or not np.isfinite(self.archive_probe_mix)
            or not 0.0 <= self.archive_probe_mix <= 1.0
        ):
            raise ValueError("archive probe mix must be a finite number in [0, 1]")
        if self.behavior_mode == "archive_uniform_probe":
            if self.archive_probe_mix == 0.0:
                raise ValueError("archive-uniform probing requires a positive mix")
        elif self.archive_probe_mix != 0.0:
            raise ValueError("archive probe mix requires archive-uniform probing")
        if (
            isinstance(self.archive_policy_min_feedback, bool)
            or not isinstance(self.archive_policy_min_feedback, Real)
            or not np.isfinite(self.archive_policy_min_feedback)
            or not 0.0 <= self.archive_policy_min_feedback <= 1.0
        ):
            raise ValueError(
                "archive policy minimum feedback must be a finite number in [0, 1]"
            )
        if self.behavior_mode == "archive_policy_feedback_floor":
            if self.archive_policy_min_feedback == 0.0:
                raise ValueError(
                    "archive policy feedback-floor behavior requires a positive "
                    "minimum"
                )
        elif self.archive_policy_min_feedback != 0.0:
            raise ValueError(
                "archive policy minimum feedback requires feedback-floor behavior"
            )
        if (
            isinstance(self.interrupt_probe_mix, bool)
            or not isinstance(self.interrupt_probe_mix, Real)
            or not np.isfinite(self.interrupt_probe_mix)
            or not 0.0 <= self.interrupt_probe_mix < 1.0
        ):
            raise ValueError(
                "interrupt probe mix must be finite and in [0, 1)"
            )
        if self.interrupt_probe_half_life is not None and (
            isinstance(self.interrupt_probe_half_life, bool)
            or not isinstance(self.interrupt_probe_half_life, Real)
            or not np.isfinite(self.interrupt_probe_half_life)
            or self.interrupt_probe_half_life <= 0.0
        ):
            raise ValueError(
                "interrupt probe half-life must be finite and positive"
            )
        minimum_confidence = 1.0 / len(ACTIONS)
        if (
            isinstance(self.interrupt_probe_max_confidence, bool)
            or not isinstance(self.interrupt_probe_max_confidence, Real)
            or not np.isfinite(self.interrupt_probe_max_confidence)
            or not minimum_confidence
            <= self.interrupt_probe_max_confidence
            <= 1.0
        ):
            raise ValueError(
                "interrupt probe maximum confidence must be finite and in "
                f"[{minimum_confidence}, 1]"
            )
        if self.behavior_mode == "uncertainty_interrupt_probe":
            if self.interrupt_probe_mix == 0.0:
                raise ValueError(
                    "uncertainty interrupt probing requires a positive mix"
                )
            if self.interrupt_probe_half_life is None:
                raise ValueError(
                    "uncertainty interrupt probing requires a half-life"
                )
        elif (
            self.interrupt_probe_mix != 0.0
            or self.interrupt_probe_half_life is not None
            or self.interrupt_probe_max_confidence != 1.0
        ):
            raise ValueError(
                "interrupt probe settings require uncertainty interrupt probing"
            )
        if self.propensity_weight_mode not in {
            "none",
            "feedback_surface_snips",
        }:
            raise ValueError("unknown SDFT propensity-weight mode")
        if (
            isinstance(self.propensity_weight_cap, bool)
            or not isinstance(self.propensity_weight_cap, Real)
            or not np.isfinite(self.propensity_weight_cap)
            or self.propensity_weight_cap < 1.0
        ):
            raise ValueError(
                "SDFT propensity-weight cap must be finite and at least one"
            )
        if (
            self.propensity_weight_mode == "none"
            and self.propensity_weight_cap != 4.0
        ):
            raise ValueError(
                "a custom propensity-weight cap requires propensity weighting"
            )
        reliable_weights = (
            self.reliable_teacher_weight,
            self.reliable_decision_weight,
            self.reliable_behavior_weight,
        )
        ambiguous_weights = (
            self.ambiguous_teacher_weight,
            self.ambiguous_decision_weight,
            self.ambiguous_behavior_weight,
        )
        profiles = (reliable_weights, ambiguous_weights)
        if any(weight < 0 for weights in profiles for weight in weights):
            raise ValueError("SDFT target weights cannot be negative")
        if self.target_mode == "teacher_only":
            if not all(weights == (1.0, 0.0, 0.0) for weights in profiles):
                raise ValueError("teacher-only mode requires its canonical weights")
        elif self.target_mode == "causal_fusion":
            if not all(np.isclose(sum(weights), 1.0) for weights in profiles):
                raise ValueError("causal-fusion target weights must sum to one")
            if self.reliable_behavior_weight < self.ambiguous_behavior_weight:
                raise ValueError(
                    "reliable callbacks cannot receive less causal weight than "
                    "ambiguous callbacks"
                )
        elif any(weight != 0.0 for weights in profiles for weight in weights):
            raise ValueError(
                "support-likelihood mode requires zero unused fusion weights"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class REINFORCESettings:
    """Configured factual-reward LoRA settings for REINFORCE.

    ``reward_outcome_map`` is learner-only shaping keyed solely by the matured
    callback's factual outcome. It never changes the shared observable-reward
    metric. An empty map instead consumes the environment's scalar reward.
    """

    learning_rate: float = REINFORCE_LR
    batch_size: int = REINFORCE_BATCH_SIZE
    baseline_step: float = REINFORCE_BASELINE_STEP
    entropy_coef: float = REINFORCE_ENTROPY_COEF
    max_grad_norm: float = REINFORCE_MAX_GRAD_NORM
    reward_outcome_map: (
        Mapping[str, float] | tuple[tuple[str, float], ...]
    ) = ()

    def __post_init__(self) -> None:
        finite_scalars = (
            self.learning_rate,
            self.baseline_step,
            self.entropy_coef,
            self.max_grad_norm,
        )
        if not np.isfinite(finite_scalars).all():
            raise ValueError("REINFORCE scalar settings must be finite")
        if self.learning_rate <= 0:
            raise ValueError("REINFORCE learning rate must be positive")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("REINFORCE batch size must be positive")
        if not 0.0 <= self.baseline_step <= 1.0:
            raise ValueError("REINFORCE baseline step must be in [0, 1]")
        if self.entropy_coef < 0:
            raise ValueError("REINFORCE entropy coefficient cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("REINFORCE maximum gradient norm must be positive")

        raw_items = (
            self.reward_outcome_map.items()
            if isinstance(self.reward_outcome_map, Mapping)
            else self.reward_outcome_map
        )
        normalized_items: list[tuple[str, float]] = []
        seen_outcomes: set[str] = set()
        for outcome, reward in raw_items:
            if not isinstance(outcome, str) or not outcome:
                raise ValueError("REINFORCE reward outcomes must be non-empty strings")
            if outcome in seen_outcomes:
                raise ValueError("REINFORCE reward outcomes must be unique")
            numeric_reward = float(reward)
            if not np.isfinite(numeric_reward):
                raise ValueError("REINFORCE outcome rewards must be finite")
            seen_outcomes.add(outcome)
            normalized_items.append((outcome, numeric_reward))
        object.__setattr__(self, "reward_outcome_map", tuple(normalized_items))

    def to_dict(self) -> dict:
        values = asdict(self)
        values["reward_outcome_map"] = dict(self.reward_outcome_map)
        return values


DEFAULT_SDFT_SETTINGS = OnlineSDFTSettings(
    replay_size=64,
    replay_recency_half_life=32.0,
    behavior_mode="uncertainty_interrupt_probe",
    behavior_epsilon=0.02,
    exploration_taper_start_step=160,
    exploration_taper_half_life=5.0,
    interrupt_probe_mix=0.15,
    interrupt_probe_half_life=80.0,
    interrupt_probe_max_confidence=0.60,
)
DEFAULT_REINFORCE_SETTINGS = REINFORCESettings(
    reward_outcome_map=REINFORCE_TRAINING_OUTCOME_REWARDS,
)
DEFAULT_RFT_STUDENT_SETTINGS = OnlineSDFTSettings(
    learning_rate=RFT_LR,
)


@dataclass(frozen=True)
class RFTSettings:
    """Isolated proposal and LoRA-student settings for causal online RFT."""

    student_settings: OnlineSDFTSettings = DEFAULT_RFT_STUDENT_SETTINGS
    candidate_count: int = RFT_CANDIDATE_COUNT
    sampling_temperature: float = RFT_SAMPLING_TEMPERATURE
    sampling_mode: str = RFT_SAMPLING_MODE

    def __post_init__(self) -> None:
        if not isinstance(self.student_settings, OnlineSDFTSettings):
            raise TypeError("RFT student settings must be OnlineSDFTSettings")
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count != 1
        ):
            raise ValueError(
                "causal online RFT currently requires candidate_count=1"
            )
        if isinstance(self.sampling_temperature, bool):
            raise ValueError(
                "RFT sampling temperature must be positive and finite"
            )
        temperature = float(self.sampling_temperature)
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                "RFT sampling temperature must be positive and finite"
            )
        object.__setattr__(self, "sampling_temperature", temperature)
        if self.sampling_mode != "categorical":
            raise ValueError("causal online RFT requires categorical sampling")

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_settings": self.student_settings.to_dict(),
            "candidate_count": self.candidate_count,
            "sampling_temperature": self.sampling_temperature,
            "sampling_mode": self.sampling_mode,
        }


DEFAULT_RFT_SETTINGS = RFTSettings()


class StudentPolicy(Protocol):
    """Minimal policy interface shared by the real LFM and test doubles."""

    def start_run(self, learning_rate: float | None) -> None: ...

    def probs(
        self,
        context: str,
        examples: list[dict] | None = None,
    ) -> np.ndarray: ...

    def teacher_probs(
        self,
        observation: TeacherObservation,
        examples: list[dict] | None = None,
    ) -> np.ndarray: ...

    def update(
        self,
        batch: list[tuple[str, np.ndarray]],
        sample_weights: np.ndarray | None = None,
    ) -> float: ...

    def base_probs(self, context: str) -> np.ndarray: ...

    def update_support(
        self,
        batch: list[tuple[str, np.ndarray]],
        sample_weights: np.ndarray | None = None,
    ) -> float: ...

    def reinforce_update(
        self,
        batch: list[tuple[str, int, float]],
        entropy_coef: float,
        max_grad_norm: float,
    ) -> float: ...


class LiquidLLMPolicy:
    """LFM2.5 student whose A/B/C next-token logits define route scores."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "auto",
        local_files_only: bool = False,
        prompt_style: str = PROMPT_STYLE,
        sdft_settings: OnlineSDFTSettings = DEFAULT_SDFT_SETTINGS,
    ):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, Lfm2ForCausalLM

        self.torch = torch
        self.model_id = model_id
        self.sdft_settings = sdft_settings
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(
                f"prompt_style must be one of {PROMPT_STYLES}, got {prompt_style!r}"
            )
        self.prompt_style = prompt_style
        self.teacher_temperature = TEACHER_TEMPERATURE
        self.teacher_reasoning_tokens = 0
        self.last_teacher_assessment: str | None = None
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        # T4-class Colab GPUs are optimized for FP16 and do not accelerate
        # BF16. Keep CPU/MPS in FP32 so the published CPU protocol is
        # unchanged, while CUDA runs use tensor cores for practical reruns.
        self.model_dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )
        torch.manual_seed(0)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = Lfm2ForCausalLM.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            dtype=self.model_dtype,
        )
        adapter_config = LoraConfig(
            r=sdft_settings.lora_rank,
            lora_alpha=sdft_settings.lora_alpha,
            lora_dropout=sdft_settings.lora_dropout,
            target_modules=list(sdft_settings.lora_target_modules),
            layers_to_transform=(
                None
                if sdft_settings.lora_layers_to_transform is None
                else list(sdft_settings.lora_layers_to_transform)
            ),
            layers_pattern=(
                None
                if sdft_settings.lora_layers_to_transform is None
                else "layers"
            ),
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=True,
            ensure_weight_tying=False,
        )
        self.model = get_peft_model(base_model, adapter_config).to(self.device)
        self.model.eval()
        self.model.config.use_cache = False

        self.action_token_ids = []
        for code in ACTION_CODES:
            token_ids = self.tokenizer.encode(code, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"action code {code!r} is not one token: {token_ids}"
                )
            self.action_token_ids.append(token_ids[0])

        self._initial_adapter = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.optimizer: Any | None = None
        self._restore_adapter_trainability()

    @property
    def trainable_parameters(self) -> int:
        """Return stable optimizer-visible adapter capacity for this setting."""
        body_a_scale = self.sdft_settings.lora_a_learning_rate_scale
        head_a_scale = self.sdft_settings.lm_head_lora_a_learning_rate_scale
        lm_head_parameter_ids = (
            {
                id(parameter)
                for parameter in self.model.get_output_embeddings().parameters()
            }
            if head_a_scale is not None
            else set()
        )
        return sum(
            parameter.numel()
            for name, parameter in self.model.named_parameters()
            if name in self._initial_adapter
            and not (
                "lora_A." in name
                and (
                    head_a_scale
                    if id(parameter) in lm_head_parameter_ids
                    else body_a_scale
                )
                == 0.0
            )
        )

    def start_run(self, learning_rate: float | None) -> None:
        """Reset the one shared adapter before every method and seed."""
        for name, parameter in self.model.named_parameters():
            if name not in self._initial_adapter:
                continue
            parameter.data.copy_(self._initial_adapter[name].to(self.device))
        self._restore_adapter_trainability()
        self.optimizer = None
        if learning_rate is not None:
            trainable = [
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            ]
            optimizer_parameters: Any
            body_a_scale = self.sdft_settings.lora_a_learning_rate_scale
            head_a_scale = self.sdft_settings.lm_head_lora_a_learning_rate_scale
            needs_lm_head_identity = (
                self.sdft_settings.lm_head_learning_rate is not None
                or head_a_scale is not None
            )
            lm_head_parameter_ids: set[int] = set()
            if needs_lm_head_identity:
                output_module = self.model.get_output_embeddings()
                lm_head_parameter_ids = {
                    id(parameter) for parameter in output_module.parameters()
                }
                if not lm_head_parameter_ids:
                    raise RuntimeError(
                        "configured LM-head adapter parameters are missing"
                    )
            if (
                self.sdft_settings.lm_head_learning_rate is None
                and body_a_scale == 1.0
                and head_a_scale in {None, 1.0}
            ):
                optimizer_parameters = [parameter for _, parameter in trainable]
            else:
                parameter_groups: dict[float, list[Any]] = {}
                for name, parameter in trainable:
                    parameter_lr = learning_rate
                    is_lm_head = id(parameter) in lm_head_parameter_ids
                    if (
                        self.sdft_settings.lm_head_learning_rate is not None
                        and is_lm_head
                    ):
                        parameter_lr = self.sdft_settings.lm_head_learning_rate
                    if "lora_A." in name:
                        parameter_lr *= (
                            head_a_scale
                            if is_lm_head and head_a_scale is not None
                            else body_a_scale
                        )
                    parameter_groups.setdefault(parameter_lr, []).append(
                        parameter
                    )
                optimizer_parameters = [
                    {"params": parameters, "lr": parameter_lr}
                    for parameter_lr, parameters in parameter_groups.items()
                ]
            self.optimizer = self.torch.optim.AdamW(
                optimizer_parameters,
                lr=learning_rate,
                betas=(self.sdft_settings.optimizer_beta1, 0.999),
                weight_decay=self.sdft_settings.optimizer_weight_decay,
            )
        self.model.eval()

    def _restore_adapter_trainability(self) -> None:
        """Reapply factor freezing after PEFT's adapter-disabled context."""
        body_a_scale = self.sdft_settings.lora_a_learning_rate_scale
        head_a_scale = self.sdft_settings.lm_head_lora_a_learning_rate_scale
        lm_head_parameter_ids = (
            {
                id(parameter)
                for parameter in self.model.get_output_embeddings().parameters()
            }
            if head_a_scale is not None
            else set()
        )
        for name, parameter in self.model.named_parameters():
            if name in self._initial_adapter:
                a_scale = (
                    head_a_scale
                    if id(parameter) in lm_head_parameter_ids
                    else body_a_scale
                )
                parameter.requires_grad_(
                    not (a_scale == 0.0 and "lora_A." in name)
                )

    def configure_online_sdft(self, settings: OnlineSDFTSettings) -> None:
        """Select teacher settings for the adapter-disabled shared model."""
        architecture_fields = (
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "lora_layers_to_transform",
        )
        if any(
            getattr(settings, field) != getattr(self.sdft_settings, field)
            for field in architecture_fields
        ):
            raise ValueError(
                "LiquidLLMPolicy LoRA architecture must match Online-SDFT settings"
            )
        self.sdft_settings = settings
        self.teacher_temperature = settings.teacher_temperature
        self.teacher_reasoning_tokens = settings.reasoning_tokens
        self.last_teacher_assessment = None

    def render_prompt(
        self,
        context: str,
        examples: list[dict] | None = None,
    ) -> str:
        messages = self.student_messages(context, examples)
        self._assert_role_alternation(messages)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def student_messages(
        self,
        context: str,
        examples: list[dict] | None = None,
    ) -> list[dict]:
        """Build a valid chat sequence for one serving-time decision."""
        history = list(examples or [])

        if self.prompt_style == "causal_demos":
            labeled_history = [
                row
                for row in history
                if row.get("observed_user_selection") in ACTIONS
            ]
            if not labeled_history:
                return [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": self._student_query(context, False),
                    },
                ]
            messages = [
                {
                    "role": "system",
                    "content": self.student_system_prompt(
                        has_examples=bool(labeled_history)
                    ),
                }
            ]
            for row in labeled_history:
                selection = row["observed_user_selection"]
                code = ACTION_CODES[ACTIONS.index(selection)]
                messages.extend(
                    [
                        {
                            "role": "user",
                            "content": (
                                f"Past notification: {row['context']}\nRoute:"
                            ),
                        },
                        {"role": "assistant", "content": code},
                    ]
                )
            messages.append(
                {
                    "role": "user",
                    "content": f"Current notification: {context}\nRoute:",
                }
            )
            return messages

        if self.prompt_style == "legacy":
            lines = []
            for index, row in enumerate(history, start=1):
                selection = row.get("observed_user_selection", "UNKNOWN")
                lines.append(f"example {index} notification: {row['context']}")
                lines.append(
                    f"example {index} completed interaction: "
                    f"executed={row.get('executed_action', 'UNKNOWN')}; "
                    f"eventual_user_action="
                    f"{row.get('eventual_user_action', 'UNKNOWN')}; "
                    f"observed_selection={selection}"
                )
                if selection in ACTIONS:
                    code = ACTION_CODES[ACTIONS.index(selection)]
                    lines.append(f"example {index} observed route: {code}")
                else:
                    lines.append(f"example {index} route: UNLABELED")
            if lines:
                lines.append(
                    "Treat these as user-specific evidence, not universal rules."
                )
            lines.extend(
                [f"current notification: {context}", "current route:"]
            )
            return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(lines)},
            ]

        messages = [
            {
                "role": "system",
                "content": self.student_system_prompt(
                    has_examples=bool(history)
                ),
            }
        ]
        query = self._student_query(context, bool(history))
        blocks = [
            self._structured_history_block(context, index, row)
            for index, row in enumerate(history, start=1)
        ]
        content = ""
        if blocks:
            content = (
                "Past completed interactions:\n"
                + "\n".join(blocks)
                + "\nUNKNOWN is unlabeled.\n\n"
            )
        messages.append({"role": "user", "content": content + query})
        return messages

    def _structured_history_block(
        self,
        context: str,
        index: int,
        row: dict,
    ) -> str:
        selection = row.get("observed_user_selection", "UNKNOWN")
        callback = FactualCallback(
            action_taken=row.get("executed_action", "UNKNOWN"),
            outcome=row.get("eventual_user_action", "UNKNOWN"),
            observed_user_selection=selection,
            delay_minutes=int(row.get("delay_minutes", 0)),
        )
        label = "This interaction is unlabeled."
        if selection in ACTIONS:
            code = ACTION_CODES[ACTIONS.index(selection)]
            label = f"Its observed route was {code} for {selection}."
        return (
            f"{index}. "
            + self._history_relevance(context, row)
            + row["context"]
            + " "
            + narrative_mobile_teacher_evidence(callback)
            + " "
            + label
        )

    def _history_relevance(self, context: str, row: dict) -> str:
        if self.prompt_style != "history_relevance":
            return ""
        current_category = self._context_field(context, "category")
        current_regime = self._context_field(context, "regime")
        example_category = self._context_field(row["context"], "category")
        example_regime = self._context_field(row["context"], "regime")
        tag = (
            "EXACT_MATCH"
            if current_category == example_category
            and current_regime == example_regime
            else "DIFFERENT_CONTEXT"
        )
        return f"Relevance: {tag}.\n"

    def _student_query(self, context: str, has_examples: bool) -> str:
        query = f"Notification: {context}\nRoute:"
        if has_examples and self.prompt_style in {
            "history_guarded",
            "history_relevance",
        }:
            query = (
                "NEW DECISION: do not repeat the last route. Ignore past "
                "labels whose category or regime differs. If none matches "
                "both fields, decide as if no history were shown.\n" + query
            )
        return query

    @staticmethod
    def _assert_role_alternation(messages: list[dict]) -> None:
        """Reject malformed chat histories before tokenization."""
        roles = [message["role"] for message in messages]
        if not roles or roles[0] != "system":
            raise ValueError("chat must start with exactly one system message")
        expected = "user"
        for role in roles[1:]:
            if role != expected:
                raise ValueError(
                    f"chat roles must alternate after system, got {roles}"
                )
            expected = "assistant" if role == "user" else "user"
        if roles[-1] != "user":
            raise ValueError("chat must end with the current user query")

    @staticmethod
    def _context_field(context: str, name: str) -> str | None:
        """Read category/regime from prose, with legacy field compatibility."""
        prose_patterns = {
            "category": r"This (?:is a )?([\w-]+) notification",
            "regime": r"during the ([\w-]+) period",
        }
        if name in prose_patterns:
            match = re.search(prose_patterns[name], context)
            if match:
                return match.group(1)
        prefix = f"{name}="
        for part in context.replace("\n", ";").split(";"):
            stripped = part.strip()
            if stripped.startswith("Metadata: "):
                stripped = stripped.removeprefix("Metadata: ").strip()
            if stripped.startswith(prefix):
                return stripped[len(prefix):].strip()
        return None

    def student_system_prompt(self, has_examples: bool = False) -> str:
        """Return the serving instruction for the configured prompt style."""
        history_only_styles = {
            "causal_demos",
            "interaction_match",
            "history_guarded",
            "history_relevance",
        }
        if self.prompt_style in history_only_styles and not has_examples:
            return SYSTEM_PROMPT
        suffix = INTERRUPT_PROMPT_SUFFIXES.get(self.prompt_style)
        if suffix is None:
            return SYSTEM_PROMPT
        return f"{SYSTEM_PROMPT}\n{suffix}"

    def render_teacher_prompt(
        self,
        observation: TeacherObservation,
        examples: list[dict] | None = None,
        assessment: str | None = None,
    ) -> str:
        """Render one completed, phone-observable trajectory for hindsight."""
        messages = self.teacher_messages(observation, examples, assessment)
        self._assert_role_alternation(messages)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def teacher_messages(
        self,
        observation: TeacherObservation,
        examples: list[dict] | None = None,
        assessment: str | None = None,
    ) -> list[dict]:
        """Build a neutral hindsight prompt without turning evidence into a label."""
        if examples:
            raise ValueError(
                "the hindsight teacher accepts exactly one completed trajectory"
            )
        assessment_block = ""
        if assessment:
            assessment_block = (
                "\n\nTeacher evidence assessment:\n"
                + assessment.strip()
            )
        user_content = (
            f"Notification:\n{observation.context}\n\n"
            + f"Observed callback:\n{observation.evidence}"
            + assessment_block
            + "\n\nRoute:"
        )
        return [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def render_teacher_assessment_prompt(
        self,
        observation: TeacherObservation,
    ) -> str:
        """Ask the same model for a bounded, auditable evidence assessment."""
        messages = [
            {"role": "system", "content": TEACHER_REASONING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Notification:\n{observation.context}\n\n"
                    f"Observed callback:\n{observation.evidence}\n\n"
                    "Assessment:"
                ),
            },
        ]
        self._assert_role_alternation(messages)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate_teacher_assessment(
        self,
        observation: TeacherObservation,
    ) -> str | None:
        tokens = int(getattr(self, "teacher_reasoning_tokens", 0))
        if tokens == 0:
            return None
        prompt = self.render_teacher_assessment_prompt(observation)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_lengths = encoded["attention_mask"].sum(-1).tolist()
        self.assert_prompt_token_budget(prompt_lengths)
        input_tokens = max(map(int, prompt_lengths))
        if input_tokens + tokens > PROMPT_TOKEN_BUDGET:
            raise ValueError(
                "teacher assessment exceeds the operational token budget: "
                f"input={input_tokens}, requested={tokens}, "
                f"budget={PROMPT_TOKEN_BUDGET}"
            )
        input_length = encoded["input_ids"].shape[1]
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }
        try:
            generated = self.model.generate(
                **encoded,
                max_new_tokens=tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            if self.device.type == "mps":
                self.torch.mps.synchronize()
            generated_ids = generated[0, input_length:].detach().cpu()
            vocab_size = getattr(
                getattr(self.model, "config", None),
                "vocab_size",
                None,
            )
            if vocab_size is None:
                try:
                    vocab_size = len(self.tokenizer)
                except TypeError:
                    vocab_size = None
            if generated_ids.numel() and (
                int(generated_ids.min()) < 0
                or (
                    vocab_size is not None
                    and int(generated_ids.max()) >= int(vocab_size)
                )
            ):
                raise ValueError("teacher generated an invalid token id")
            assessment = self.tokenizer.decode(
                generated_ids.tolist(),
                skip_special_tokens=True,
            ).strip()
        except (OverflowError, RuntimeError, ValueError):
            if self.device.type == "mps":
                self.torch.mps.empty_cache()
            return ASSESSMENT_FALLBACK
        if not assessment:
            return ASSESSMENT_FALLBACK
        assessment = " ".join(assessment.split())
        if _assessment_crosses_boundary(assessment):
            return ASSESSMENT_FALLBACK
        return assessment

    def _action_logits(self, prompts: list[str]):
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        if "attention_mask" in encoded:
            token_counts = encoded["attention_mask"].sum(-1).tolist()
        else:
            token_counts = [encoded["input_ids"].shape[-1]] * len(prompts)
        self.assert_prompt_token_budget(token_counts)
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }
        # Compute normalization and the distillation loss in FP32 even when
        # the frozen CUDA backbone runs in FP16.
        logits = self.model(**encoded).logits[:, -1, :].float()
        action_ids = self.torch.tensor(
            self.action_token_ids,
            device=self.device,
        )
        return logits.index_select(-1, action_ids)

    @staticmethod
    def assert_prompt_token_budget(token_counts: list[int]) -> None:
        """Enforce the declared operational input budget before every forward."""
        over_budget = [
            int(count)
            for count in token_counts
            if int(count) > PROMPT_TOKEN_BUDGET
        ]
        if over_budget:
            raise ValueError(
                "prompt exceeds the operational token budget: "
                f"max={max(over_budget)}, budget={PROMPT_TOKEN_BUDGET}"
            )

    def probs(
        self,
        context: str,
        examples: list[dict] | None = None,
    ) -> np.ndarray:
        self.model.eval()
        with self.torch.no_grad():
            logits = self._action_logits([self.render_prompt(context, examples)])
            probabilities = self.torch.softmax(
                logits / STUDENT_TEMPERATURE,
                dim=-1,
            )[0]
        values = probabilities.float().cpu().numpy()
        if not np.isfinite(values).all():
            values = np.ones(len(ACTIONS), dtype=float)
        values = np.clip(values, 1e-8, None)
        return values / values.sum()

    def base_probs(self, context: str) -> np.ndarray:
        """Score a student prompt with this same model's adapter disabled."""
        self.model.eval()
        try:
            with self.torch.no_grad(), self.model.disable_adapter():
                logits = self._action_logits([self.render_prompt(context)])
                probabilities = self.torch.softmax(
                    logits / STUDENT_TEMPERATURE,
                    dim=-1,
                )[0]
        finally:
            self._restore_adapter_trainability()
        values = probabilities.float().cpu().numpy()
        if not np.isfinite(values).all():
            values = np.ones(len(ACTIONS), dtype=float)
        values = np.clip(values, 1e-8, None)
        return values / values.sum()

    def teacher_probs(
        self,
        observation: TeacherObservation,
        examples: list[dict] | None = None,
    ) -> np.ndarray:
        """Use the same model with LoRA disabled for fixed-base hindsight."""
        self.model.eval()
        try:
            with self.torch.no_grad():
                # This is deliberately one physical model instance. Disabling
                # its adapter gives the fixed-initial teacher while the
                # privileged hindsight prompt supplies the only distinction.
                with self.model.disable_adapter():
                    assessment = self._generate_teacher_assessment(observation)
                    self.last_teacher_assessment = assessment
                    logits = self._action_logits(
                        [
                            self.render_teacher_prompt(
                                observation,
                                examples,
                                assessment,
                            )
                        ]
                    )[0]
                    probabilities = self.torch.softmax(
                        logits / self.teacher_temperature,
                        dim=-1,
                    )
        finally:
            self._restore_adapter_trainability()
        values = probabilities.float().cpu().numpy()
        if not np.isfinite(values).all():
            values = np.ones(len(ACTIONS), dtype=float)
        values = np.clip(values, 1e-8, None)
        return values / values.sum()

    def _validated_loss_weights(
        self,
        sample_weights: np.ndarray | None,
        batch_size: int,
    ):
        """Return detached positive FP32 weights for one optimizer batch."""
        if sample_weights is None:
            return None
        values = np.asarray(sample_weights, dtype=float)
        if values.shape != (batch_size,):
            raise ValueError("sample weights must contain one value per row")
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("sample weights must be finite and positive")
        return self.torch.tensor(
            values,
            device=self.device,
            dtype=self.torch.float32,
        )

    def update(
        self,
        batch: list[tuple[str, np.ndarray]],
        sample_weights: np.ndarray | None = None,
    ) -> float:
        """Apply one soft-target cross-entropy update to LoRA only."""
        if self.optimizer is None:
            raise RuntimeError(
                "start_run must receive a learning rate before LoRA update"
            )
        self.model.train()
        prompts = [self.render_prompt(context) for context, _ in batch]
        targets = self.torch.tensor(
            np.stack([target for _, target in batch]),
            device=self.device,
            dtype=self.torch.float32,
        )
        logits = self._action_logits(prompts)
        per_example_loss = -(
            targets * self.torch.log_softmax(logits, dim=-1)
        ).sum(-1)
        weights = self._validated_loss_weights(sample_weights, len(batch))
        loss = (
            per_example_loss.mean()
            if weights is None
            else (weights * per_example_loss).sum() / weights.sum()
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            max_norm=self.sdft_settings.max_grad_norm,
        )
        self.optimizer.step()
        if self.device.type == "mps":
            self.torch.mps.synchronize()
            self.torch.mps.empty_cache()
        return float(loss.detach().cpu())

    def reinforce_update(
        self,
        batch: list[tuple[str, int, float]],
        entropy_coef: float,
        max_grad_norm: float,
    ) -> float:
        """Apply one factual-reward action-token update to LoRA only."""
        if self.optimizer is None:
            raise RuntimeError(
                "start_run must receive a learning rate before LoRA update"
            )
        if not batch:
            raise ValueError("REINFORCE update batch cannot be empty")
        if not np.isfinite(entropy_coef) or entropy_coef < 0.0:
            raise ValueError(
                "REINFORCE entropy coefficient must be finite and non-negative"
            )
        if not np.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
            raise ValueError(
                "REINFORCE maximum gradient norm must be finite and positive"
            )
        actions_array = np.asarray(
            [action for _, action, _ in batch],
            dtype=int,
        )
        advantages_array = np.asarray(
            [advantage for _, _, advantage in batch],
            dtype=float,
        )
        if np.any(actions_array < 0) or np.any(actions_array >= len(ACTIONS)):
            raise ValueError("REINFORCE actions must index a configured route")
        if not np.isfinite(advantages_array).all():
            raise ValueError("REINFORCE advantages must be finite")

        self.model.train()
        prompts = [self.render_prompt(context) for context, _, _ in batch]
        actions = self.torch.tensor(
            actions_array,
            device=self.device,
            dtype=self.torch.long,
        )
        advantages = self.torch.tensor(
            advantages_array,
            device=self.device,
            dtype=self.torch.float32,
        )
        logits = self._action_logits(prompts) / STUDENT_TEMPERATURE
        log_probabilities = self.torch.log_softmax(logits, dim=-1)
        selected_log_probabilities = log_probabilities.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum(-1)
        loss = -(
            advantages.detach() * selected_log_probabilities
            + entropy_coef * entropy
        ).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            max_norm=max_grad_norm,
        )
        self.optimizer.step()
        if self.device.type == "mps":
            self.torch.mps.synchronize()
            self.torch.mps.empty_cache()
        return float(loss.detach().cpu())

    def update_support(
        self,
        batch: list[tuple[str, np.ndarray]],
        sample_weights: np.ndarray | None = None,
    ) -> float:
        """Maximize probability assigned to each callback's causal support."""
        if self.optimizer is None:
            raise RuntimeError(
                "start_run must receive a learning rate before LoRA update"
            )
        if not batch:
            raise ValueError("causal support update batch cannot be empty")
        self.model.train()
        prompts = [self.render_prompt(context) for context, _ in batch]
        supports_array = np.stack(
            [np.asarray(support, dtype=float) for _, support in batch]
        )
        if supports_array.shape != (len(batch), len(ACTIONS)):
            raise ValueError("causal support masks must have one value per route")
        if not np.isfinite(supports_array).all():
            raise ValueError("causal support masks must be finite")
        if not np.logical_or(supports_array == 0.0, supports_array == 1.0).all():
            raise ValueError("causal support masks must be binary")
        support_sizes = supports_array.sum(axis=1)
        if not np.all(support_sizes >= 1):
            raise ValueError("causal support masks cannot be empty")
        if not np.all(support_sizes < len(ACTIONS)):
            raise ValueError("uninformative full support must remain censored")
        supports = self.torch.tensor(
            supports_array,
            device=self.device,
            dtype=self.torch.bool,
        )
        logits = self._action_logits(prompts)
        log_probabilities = self.torch.log_softmax(logits, dim=-1)
        supported_log_probabilities = log_probabilities.masked_fill(
            ~supports,
            -self.torch.inf,
        )
        per_example_loss = -self.torch.logsumexp(
            supported_log_probabilities,
            dim=-1,
        )
        weights = self._validated_loss_weights(sample_weights, len(batch))
        loss = (
            per_example_loss.mean()
            if weights is None
            else (weights * per_example_loss).sum() / weights.sum()
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            max_norm=self.sdft_settings.max_grad_norm,
        )
        self.optimizer.step()
        if self.device.type == "mps":
            self.torch.mps.synchronize()
            self.torch.mps.empty_cache()
        return float(loss.detach().cpu())


@dataclass
class InteractionRecord:
    """One completed causal interaction retained by ICL or RAG."""

    observation: StudentObservation
    action: int
    feedback: dict

    def prompt_example(self) -> dict:
        return {
            "context": self.observation.text,
            "executed_action": self.feedback.get(
                "action_taken", ACTIONS[self.action]
            ),
            "eventual_user_action": self.feedback.get(
                "outcome", "UNKNOWN"
            ),
            "delay_minutes": self.feedback.get("delay_minutes", 0),
            "observed_user_selection": self.feedback.get(
                "observed_user_selection", "UNKNOWN"
            ),
        }


def mixed_context_similarity(
    query: StudentObservation,
    candidate: StudentObservation,
    text_weight: float = RAG_TEXT_WEIGHT,
) -> float:
    """Gower-style similarity over the shared decision-time fields only.

    Category and regime use exact match; hour uses circular distance; and
    importance uses normalized absolute distance. Deadline, urgency, affinity,
    the hidden dataset answer, and the post-action user selection are never
    retrieval keys.
    """
    category_count = len(CATEGORIES)
    query_features = query.features
    candidate_features = candidate.features

    category_similarity = float(
        np.argmax(query_features[:category_count])
        == np.argmax(candidate_features[:category_count])
    )

    query_importance = query_features[category_count]
    candidate_importance = candidate_features[category_count]
    importance_similarity = 1.0 - abs(
        float(query_importance - candidate_importance)
    )

    query_clock = query_features[category_count + 1 : category_count + 3]
    candidate_clock = candidate_features[
        category_count + 1 : category_count + 3
    ]
    clock_cosine = float(
        np.clip(np.dot(query_clock, candidate_clock), -1.0, 1.0)
    )
    hour_similarity = 1.0 - float(np.arccos(clock_cosine) / np.pi)

    query_regime = int(round(2 * query_features[category_count + 3]))
    candidate_regime = int(
        round(2 * candidate_features[category_count + 3])
    )
    regime_similarity = float(query_regime == candidate_regime)
    metadata_similarity = float(
        np.mean(
            [
                category_similarity,
                importance_similarity,
                hour_similarity,
                regime_similarity,
            ]
        )
    )
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("text_weight must be between 0 and 1")
    if text_weight == 0.0:
        return metadata_similarity
    text_similarity = notification_text_similarity(
        query.text,
        candidate.text,
    )
    return float(
        (1.0 - text_weight) * metadata_similarity
        + text_weight * text_similarity
    )


_CONTENT_STOPWORDS = {
    "a", "an", "and", "at", "before", "during", "for", "from", "in",
    "is", "it", "local", "message", "notification", "of", "on", "out",
    "says", "score", "the", "this", "time", "to", "was", "with",
}


def _notification_content_tokens(text: str) -> set[str]:
    """Tokenize only the visible title/body portion of a serving prompt."""
    content = text.split(" This is a ", 1)[0].lower()
    translation = str.maketrans(
        {character: " " for character in string.punctuation}
    )
    return {
        token
        for token in content.translate(translation).split()
        if len(token) > 1 and token not in _CONTENT_STOPWORDS
    }


def notification_text_similarity(query_text: str, candidate_text: str) -> float:
    """Return Jaccard similarity over decision-visible title/body tokens."""
    query_tokens = _notification_content_tokens(query_text)
    candidate_tokens = _notification_content_tokens(candidate_text)
    union = query_tokens | candidate_tokens
    if not union:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(union)


def feedback_surface_propensity(behavior_distribution: np.ndarray) -> float:
    """Return the decision-time chance of an observable feedback surface."""
    behavior = np.asarray(behavior_distribution, dtype=float)
    if behavior.shape != (len(ACTIONS),):
        raise ValueError("behavior distribution must contain one value per route")
    if not np.isfinite(behavior).all() or np.any(behavior < 0.0):
        raise ValueError("behavior distribution must be finite and non-negative")
    if not np.isclose(float(behavior.sum()), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("behavior distribution must sum to one")
    propensity = float(
        behavior[ACTIONS.index("INTERRUPT")]
        + behavior[ACTIONS.index("LATER")]
    )
    if propensity <= 0.0:
        raise ValueError("feedback-surface propensity must be positive")
    return propensity


class OnlineAgent:
    """Base class for one method on one chronological stream."""

    name = "Base"
    learning_rate: float | None = None
    samples_from_policy = False
    uses_teacher = False
    stores_interactions = False
    replay_size = REPLAY_SIZE
    online_batch_size = ONLINE_BATCH_SIZE
    update_steps = 1

    def __init__(
        self,
        policy: StudentPolicy,
        icl_examples: int = ICL_K,
        rag_examples: int = RAG_K,
    ):
        self.policy = policy
        if icl_examples < 0 or rag_examples < 0:
            raise ValueError("example counts must be non-negative")
        self.icl_examples = icl_examples
        self.rag_examples = rag_examples
        self.memory: list[InteractionRecord] = []
        self.replay: list[tuple[Any, ...]] = []
        self.policy.start_run(self.learning_rate)

    def prompt_examples(
        self,
        observation: StudentObservation,
    ) -> list[dict]:
        del observation
        return []

    def action_probs(self, observation: StudentObservation) -> np.ndarray:
        return self.policy.probs(
            observation.text,
            self.prompt_examples(observation),
        )

    def reliable_memory(self) -> list[InteractionRecord]:
        """Return callbacks that identify one route on the executed surface.

        Memory retains every matured factual interaction for audit, including
        ambiguous digest opens and censored archive outcomes. Only singleton
        causal evidence becomes a labeled prompt example: otherwise the
        executed route can prime its own future selection without identifying
        the user's preferred route.
        """
        return [
            record
            for record in self.memory
            if record.feedback.get("observed_user_selection") != "UNKNOWN"
            and causal_evidence_reliability(
                record.action,
                record.feedback,
            ) == "reliable_singleton"
        ]

    def training_target(
        self,
        teacher_distribution: np.ndarray,
        teacher_action: int,
        *,
        action: int,
        feedback: dict,
        decision_distribution: np.ndarray | None,
        candidate_action: int | None = None,
    ) -> np.ndarray | None:
        del (
            teacher_distribution,
            teacher_action,
            action,
            feedback,
            decision_distribution,
            candidate_action,
        )
        return None

    def observe(
        self,
        observation: StudentObservation,
        action: int,
        teacher_distribution: np.ndarray | None,
        teacher_action: int | None,
        feedback: dict,
        rng: np.random.Generator,
        teacher_observation: TeacherObservation | None = None,
        decision_distribution: np.ndarray | None = None,
        candidate_action: int | None = None,
        behavior_distribution: np.ndarray | None = None,
    ) -> None:
        """Retain direct history or apply a hindsight distillation update."""
        if hasattr(self, "last_observation_update_count"):
            self.last_observation_update_count = 0
        if self.stores_interactions:
            self.memory.append(
                InteractionRecord(
                    observation=observation,
                    action=action,
                    feedback=feedback,
                )
            )
            return
        if feedback.get("observed_user_selection") == "UNKNOWN":
            # Censoring is not a negative label and not a teacher-generated
            # hard label. The teacher distribution may still be logged by the
            # experiment for audit, but it cannot enter memory or an update.
            getattr(self, "_decision_base_cache", {}).pop(
                id(observation),
                None,
            )
            return
        if teacher_distribution is None or teacher_action is None:
            if self.uses_teacher:
                raise ValueError("teacher-supervised method requires a teacher")
            return
        target = self.training_target(
            teacher_distribution,
            teacher_action,
            action=action,
            feedback=feedback,
            decision_distribution=decision_distribution,
            candidate_action=candidate_action,
        )
        if target is None:
            getattr(self, "_decision_base_cache", {}).pop(
                id(observation),
                None,
            )
            return

        feedback_propensity = (
            None
            if behavior_distribution is None
            else feedback_surface_propensity(behavior_distribution)
        )
        propensity_mode = getattr(
            getattr(self, "settings", None),
            "propensity_weight_mode",
            "none",
        )
        if propensity_mode != "none" and feedback_propensity is None:
            raise ValueError(
                "propensity-weighted replay requires the decision-time "
                "behavior distribution"
            )

        selection = feedback["observed_user_selection"]
        replay_label = (
            "AMBIGUOUS"
            if ACTIONS[action] == "LATER"
            and feedback.get("outcome") == "OPENED_DIGEST"
            else selection
        )
        replay_prompt_example = (
            InteractionRecord(
                observation=observation,
                action=action,
                feedback=feedback,
            ).prompt_example()
            if getattr(self, "replay_prompt_examples", 0) > 0
            and causal_evidence_reliability(action, feedback)
            == "reliable_singleton"
            else None
        )
        self.replay.append(
            (
                observation.text,
                target,
                replay_label,
                getattr(self, "_decision_base_cache", {}).pop(
                    id(observation),
                    None,
                ),
                replay_prompt_example,
                feedback_propensity,
            )
        )
        self.replay = self.replay[-self.replay_size:]
        if len(self.replay) < getattr(self, "warmup_examples", 1):
            return
        settings = getattr(self, "settings", None)
        if (
            replay_label == "AMBIGUOUS"
            and getattr(settings, "ambiguous_update_mode", "immediate")
            == "defer"
        ):
            return
        for update_step in range(self.update_steps):
            force_newest = bool(
                update_step == 0
                or getattr(settings, "force_newest_every_step", True)
            )
            indices = [len(self.replay) - 1] if force_newest else []
            candidate_stop = len(self.replay) - int(force_newest)
            candidates = np.arange(candidate_stop)
            sample_size = min(
                self.online_batch_size - len(indices),
                len(candidates),
            )
            if sample_size:
                probabilities = None
                replay_strategy = getattr(self, "replay_strategy", "uniform")
                if replay_strategy == "selection_balanced":
                    recency_half_life = getattr(
                        settings,
                        "replay_recency_half_life",
                        None,
                    )
                    if recency_half_life is not None:
                        labels = [
                            self.replay[index][2] for index in candidates
                        ]
                        weights = np.empty(len(candidates), dtype=float)
                        for label in dict.fromkeys(labels):
                            positions = np.asarray(
                                [
                                    position
                                    for position, candidate_label in enumerate(
                                        labels
                                    )
                                    if candidate_label == label
                                ],
                                dtype=int,
                            )
                            ages = (
                                len(self.replay)
                                - 1
                                - candidates[positions]
                            )
                            # Subtracting the within-label newest age is a
                            # numerically stable common rescaling of
                            # q = 2 ** (-age / half_life).
                            relative_recency = np.exp2(
                                -(ages - ages.min()) / recency_half_life
                            )
                            group_mass = (
                                getattr(
                                    settings,
                                    "ambiguous_replay_group_weight",
                                    1.0,
                                )
                                if label == "AMBIGUOUS"
                                else 1.0
                            )
                            weights[positions] = (
                                group_mass
                                * relative_recency
                                / relative_recency.sum()
                            )
                    else:
                        counts = Counter(
                            self.replay[index][2]
                            for index in candidates
                        )
                        weights = np.asarray(
                            [
                                (
                                    getattr(
                                        getattr(self, "settings", None),
                                        "ambiguous_replay_group_weight",
                                        1.0,
                                    )
                                    if self.replay[index][2] == "AMBIGUOUS"
                                    else 1.0
                                )
                                / counts[self.replay[index][2]]
                                for index in candidates
                            ],
                            dtype=float,
                        )
                    probabilities = weights / weights.sum()
                indices += rng.choice(
                    candidates,
                    size=sample_size,
                    replace=False,
                    p=probabilities,
                ).tolist()
            self.update_from_replay([self.replay[index] for index in indices])
            if hasattr(self, "last_observation_update_count"):
                self.last_observation_update_count += 1
                self.online_update_count += 1


class BaseAgent(OnlineAgent):
    """Frozen LFM without external memory."""


class ICLAgent(OnlineAgent):
    """Frozen LFM prompted with recent reliable causal interactions."""

    name = "ICL"
    stores_interactions = True

    def prompt_examples(
        self,
        observation: StudentObservation,
    ) -> list[dict]:
        del observation
        if self.icl_examples == 0:
            return []
        return [
            record.prompt_example()
            for record in self.reliable_memory()[-self.icl_examples:]
        ]


class RAGAgent(OnlineAgent):
    """Frozen LFM prompted with similar reliable causal interactions."""

    name = "RAG"
    stores_interactions = True

    def __init__(
        self,
        policy: StudentPolicy,
        icl_examples: int = ICL_K,
        rag_examples: int = RAG_K,
        rag_text_weight: float = RAG_TEXT_WEIGHT,
    ):
        if not 0.0 <= rag_text_weight <= 1.0:
            raise ValueError("rag_text_weight must be between 0 and 1")
        self.rag_text_weight = rag_text_weight
        super().__init__(policy, icl_examples, rag_examples)

    def prompt_examples(
        self,
        observation: StudentObservation,
    ) -> list[dict]:
        records = self.reliable_memory()
        if not records:
            return []
        similarities = [
            (
                mixed_context_similarity(
                    observation,
                    record.observation,
                    self.rag_text_weight,
                ),
                index,
                record,
            )
            for index, record in enumerate(records)
        ]
        closest = sorted(
            similarities,
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )[:self.rag_examples]
        # Put the best match closest to the current query. For equal scores,
        # the newest record is also closest to the query in the prompt.
        closest.sort(key=lambda item: (item[0], item[1]))
        return [record.prompt_example() for _, _, record in closest]


class REINFORCEAgent(OnlineAgent):
    """Batched action-token LoRA policy gradient from factual rewards only."""

    name = "REINFORCE"
    learning_rate = REINFORCE_LR
    online_batch_size = REINFORCE_BATCH_SIZE
    samples_from_policy = True
    uses_teacher = False

    def __init__(
        self,
        policy: StudentPolicy,
        icl_examples: int = ICL_K,
        rag_examples: int = RAG_K,
        settings: REINFORCESettings = DEFAULT_REINFORCE_SETTINGS,
        adapter_settings: OnlineSDFTSettings = DEFAULT_SDFT_SETTINGS,
    ):
        self.settings = settings
        self.learning_rate = settings.learning_rate
        self.online_batch_size = settings.batch_size
        self.reward_outcome_map = dict(settings.reward_outcome_map)
        self.reward_baseline = 0.0
        self.last_training_reward: float | None = None
        self.pending_updates: list[tuple[str, int, float]] = []
        self.online_update_count = 0
        self.last_observation_update_count = 0
        configure = getattr(policy, "configure_online_sdft", None)
        if configure is not None:
            configure(adapter_settings)
        super().__init__(policy, icl_examples, rag_examples)

    def observe(
        self,
        observation: StudentObservation,
        action: int,
        teacher_distribution: np.ndarray | None,
        teacher_action: int | None,
        feedback: dict,
        rng: np.random.Generator,
        teacher_observation: TeacherObservation | None = None,
        decision_distribution: np.ndarray | None = None,
        candidate_action: int | None = None,
        behavior_distribution: np.ndarray | None = None,
    ) -> None:
        """Batch known factual rewards and update the shared LoRA adapter."""
        del (
            teacher_distribution,
            teacher_action,
            rng,
            teacher_observation,
            decision_distribution,
            candidate_action,
            behavior_distribution,
        )
        self.last_observation_update_count = 0
        self.last_training_reward = None
        selection = feedback.get("observed_user_selection")
        if selection not in {*ACTIONS, "UNKNOWN"}:
            raise ValueError(
                "REINFORCE feedback must contain an explicit observed user "
                "selection"
            )
        if selection == "UNKNOWN":
            # A neutral censored event must remain neutral after baseline
            # subtraction, so it produces no policy-gradient update.
            return
        if self.reward_outcome_map:
            outcome = feedback.get("outcome")
            if outcome not in self.reward_outcome_map:
                raise ValueError(
                    "REINFORCE factual outcome is missing from the configured "
                    "reward map"
                )
            reward = self.reward_outcome_map[outcome]
        else:
            reward = float(feedback["reward"])
        if not np.isfinite(reward):
            raise ValueError("REINFORCE factual reward must be finite")
        self.last_training_reward = float(reward)
        advantage = reward - self.reward_baseline
        self.pending_updates.append((observation.text, action, advantage))
        self.reward_baseline += self.settings.baseline_step * (
            reward - self.reward_baseline
        )
        if len(self.pending_updates) < self.online_batch_size:
            return

        self.policy.reinforce_update(
            list(self.pending_updates),
            entropy_coef=self.settings.entropy_coef,
            max_grad_norm=self.settings.max_grad_norm,
        )
        self.pending_updates.clear()
        self.online_update_count += 1
        self.last_observation_update_count = 1


def causal_route_support(action: int, feedback: dict) -> np.ndarray:
    """Return routes identifiable from one factual delivery-surface callback.

    This encodes the public observation contract, not the evaluator's hidden
    preferred route. In particular, opening a digest cannot distinguish a
    missed immediate need from a genuine preference to read later.
    """
    if action < 0 or action >= len(ACTIONS):
        raise ValueError("action index is out of range")
    action_name = ACTIONS[action]
    recorded_action = feedback.get("action_taken", action_name)
    if recorded_action != action_name:
        raise ValueError("feedback must belong to the executed route")

    outcome = feedback.get("outcome")
    support_by_trajectory = {
        ("INTERRUPT", "OPENED_IMMEDIATELY"): ("INTERRUPT",),
        ("INTERRUPT", "OPENED_AFTER_DELAY"): ("LATER",),
        ("INTERRUPT", "DELETED_NOTIFICATION"): ("ARCHIVE",),
        ("LATER", "OPENED_DIGEST"): ("INTERRUPT", "LATER"),
        ("LATER", "DELETED_FROM_DIGEST"): ("ARCHIVE",),
        ("ARCHIVE", "NO_OBSERVABLE_SELECTION"): ACTIONS,
    }
    try:
        supported = support_by_trajectory[(action_name, outcome)]
    except KeyError as error:
        raise ValueError(
            "unknown action/outcome trajectory for causal support: "
            f"{action_name}/{outcome}"
        ) from error
    return np.asarray(
        [route in supported for route in ACTIONS],
        dtype=bool,
    )


def causal_evidence_reliability(action: int, feedback: dict) -> str:
    """Classify callback reliability from public causal support cardinality."""
    support = causal_route_support(action, feedback)
    supported_routes = int(support.sum())
    if supported_routes == 1:
        return "reliable_singleton"
    if supported_routes == 2:
        return "ambiguous_digest_open"
    if support.all():
        return "censored_unknown"
    raise ValueError("causal support must contain at least one route")


class OnlineSDFTAgent(OnlineAgent):
    """Online LoRA student from reliability-conditioned soft evidence."""

    name = "Online-SDFT"
    learning_rate = SDFT_LR
    replay_size = SDFT_REPLAY_SIZE
    online_batch_size = SDFT_BATCH_SIZE
    update_steps = SDFT_UPDATE_STEPS
    uses_teacher = True

    def __init__(
        self,
        policy: StudentPolicy,
        icl_examples: int = ICL_K,
        rag_examples: int = RAG_K,
        settings: OnlineSDFTSettings = DEFAULT_SDFT_SETTINGS,
    ):
        self.settings = settings
        self.learning_rate = settings.learning_rate
        self.online_update_count = 0
        self.last_observation_update_count = 0
        self.replay_size = settings.replay_size
        self.replay_prompt_examples = settings.replay_prompt_examples
        self.online_batch_size = settings.batch_size
        self.update_steps = settings.update_steps
        self.warmup_examples = settings.warmup_examples
        self.replay_strategy = settings.replay_strategy
        self.behavior_mode = settings.behavior_mode
        self.behavior_epsilon = settings.behavior_epsilon
        self.behavior_epsilon_half_life = settings.behavior_epsilon_half_life
        self.exploration_taper_start_step = (
            settings.exploration_taper_start_step
        )
        self.exploration_taper_half_life = (
            settings.exploration_taper_half_life
        )
        self.archive_probe_mix = settings.archive_probe_mix
        self.archive_policy_min_feedback = settings.archive_policy_min_feedback
        self.interrupt_probe_mix = settings.interrupt_probe_mix
        self.interrupt_probe_half_life = settings.interrupt_probe_half_life
        self.interrupt_probe_max_confidence = (
            settings.interrupt_probe_max_confidence
        )
        self.last_prompt_examples_used = 0
        self._decision_base_cache: dict[int, np.ndarray] = {}
        configure = getattr(policy, "configure_online_sdft", None)
        if configure is not None:
            configure(settings)
        super().__init__(policy, icl_examples, rag_examples)

    def prompt_examples(
        self,
        observation: StudentObservation,
    ) -> list[dict]:
        """Return recent reliable replay lessons without retrieval.

        Only factual singleton callbacks enter this FIFO prompt. Ambiguous
        digest opens, censored outcomes, soft targets, and teacher outputs are
        excluded. A zero count keeps pure parametric Online-SDFT unchanged.
        """
        del observation
        if self.replay_prompt_examples == 0:
            self.last_prompt_examples_used = 0
            return []
        eligible = [
            row[4]
            for row in self.replay
            if len(row) > 4 and row[4] is not None
        ]
        examples = eligible[-self.replay_prompt_examples :]
        self.last_prompt_examples_used = len(examples)
        return [dict(example) for example in examples]

    def action_probs(self, observation: StudentObservation) -> np.ndarray:
        """Serve from the shared Liquid model with its current LoRA state."""
        return self.policy.probs(
            observation.text,
            self.prompt_examples(observation),
        )

    def observe(
        self,
        observation: StudentObservation,
        action: int,
        teacher_distribution: np.ndarray | None,
        teacher_action: int | None,
        feedback: dict,
        rng: np.random.Generator,
        teacher_observation: TeacherObservation | None = None,
        decision_distribution: np.ndarray | None = None,
        candidate_action: int | None = None,
        behavior_distribution: np.ndarray | None = None,
    ) -> None:
        """Cache one adapter-disabled student anchor for informative replay."""
        cache_key = id(observation)
        if (
            self.settings.base_kl_weight
            and feedback.get("observed_user_selection") != "UNKNOWN"
        ):
            self._decision_base_cache[cache_key] = self.policy.base_probs(
                observation.text
            )
        try:
            super().observe(
                observation,
                action,
                teacher_distribution,
                teacher_action,
                feedback,
                rng,
                teacher_observation=teacher_observation,
                decision_distribution=decision_distribution,
                candidate_action=candidate_action,
                behavior_distribution=behavior_distribution,
            )
        except Exception:
            self._decision_base_cache.pop(cache_key, None)
            raise

    def update_from_replay(self, rows: list[tuple[Any, ...]]) -> None:
        """Fit only the PEFT adapter on resolved replay contexts and targets."""
        sample_weights = None
        if self.settings.propensity_weight_mode == "feedback_surface_snips":
            if any(len(row) < 6 or row[5] is None for row in rows):
                raise ValueError(
                    "propensity-weighted replay requires stored feedback "
                    "propensities"
                )
            propensities = np.asarray([row[5] for row in rows], dtype=float)
            if (
                not np.isfinite(propensities).all()
                or np.any(propensities <= 0.0)
                or np.any(propensities > 1.0)
            ):
                raise ValueError(
                    "stored feedback propensities must be finite and in (0, 1]"
                )
            sample_weights = np.minimum(
                self.settings.propensity_weight_cap,
                1.0 / np.maximum(propensities, 1e-8),
            )
        if self.settings.base_kl_weight:
            if any(len(row) < 4 or row[3] is None for row in rows):
                raise ValueError("fixed-base KL replay requires cached anchors")
            beta = self.settings.base_kl_weight
            batch = []
            for row in rows:
                target = np.asarray(row[1], dtype=float)
                target = np.clip(target, 1e-8, None)
                target /= target.sum()
                base = np.asarray(row[3], dtype=float)
                base = np.clip(base, 1e-8, None)
                base /= base.sum()
                effective_target = (target + beta * base) / (1.0 + beta)
                batch.append((row[0], effective_target))
        else:
            batch = [(row[0], row[1]) for row in rows]
        if self.settings.target_mode == "support_likelihood":
            if sample_weights is None:
                self.policy.update_support(batch)
            else:
                self.policy.update_support(
                    batch,
                    sample_weights=sample_weights,
                )
            return
        if sample_weights is None:
            self.policy.update(batch)
        else:
            self.policy.update(batch, sample_weights=sample_weights)

    @staticmethod
    def causal_behavior_support(action: int, feedback: dict) -> np.ndarray:
        """Return maximum-entropy guidance over the causal support set."""
        support = causal_route_support(action, feedback).astype(float)
        return support / support.sum()

    def fusion_weights(self, action: int, feedback: dict) -> tuple[float, ...]:
        """Resolve teacher, decision, and behavior weights for one callback."""
        reliability = causal_evidence_reliability(action, feedback)
        if reliability == "reliable_singleton":
            return (
                self.settings.reliable_teacher_weight,
                self.settings.reliable_decision_weight,
                self.settings.reliable_behavior_weight,
            )
        if reliability == "ambiguous_digest_open":
            return (
                self.settings.ambiguous_teacher_weight,
                self.settings.ambiguous_decision_weight,
                self.settings.ambiguous_behavior_weight,
            )
        raise ValueError("censored feedback cannot produce an SDFT target")

    def training_target(
        self,
        teacher_distribution: np.ndarray,
        teacher_action: int,
        *,
        action: int,
        feedback: dict,
        decision_distribution: np.ndarray | None,
        candidate_action: int | None = None,
    ) -> np.ndarray | None:
        """Build a soft target from causal evidence available at release."""
        del teacher_action, candidate_action
        reliability = None
        if self.settings.ambiguous_update_mode == "skip":
            reliability = causal_evidence_reliability(action, feedback)
            if reliability == "ambiguous_digest_open":
                return None
        if self.settings.target_mode == "support_likelihood":
            support = causal_route_support(action, feedback)
            if support.all():
                raise ValueError("censored feedback cannot produce an SDFT target")
            return support.astype(float)
        teacher = np.clip(teacher_distribution, 1e-8, None)
        teacher = teacher / teacher.sum()
        if self.settings.target_mode == "teacher_only":
            return teacher
        if decision_distribution is None:
            raise ValueError("causal fusion requires the frozen decision prior")
        decision = np.clip(decision_distribution, 1e-8, None)
        decision = decision / decision.sum()
        if reliability is None:
            reliability = causal_evidence_reliability(action, feedback)
        if (
            reliability == "ambiguous_digest_open"
            and self.settings.ambiguous_projection == "causal_support"
        ):
            support = causal_route_support(action, feedback).astype(float)
            teacher = teacher * support
            teacher = teacher / teacher.sum()
            decision = decision * support
            decision = decision / decision.sum()
        teacher_weight, decision_weight, behavior_weight = self.fusion_weights(
            action,
            feedback,
        )
        behavior = self.causal_behavior_support(action, feedback)
        target = (
            teacher_weight * teacher
            + decision_weight * decision
            + behavior_weight * behavior
        )
        target = np.clip(target, 1e-8, None)
        return target / target.sum()


class RFTAgent(OnlineSDFTAgent):
    """Teacher-sampled hard-target distillation with causal rejection.

    RFT shares Online-SDFT's frozen LFM base and LoRA architecture. It retains
    its independently configured replay-32 epsilon-greedy schedule and tuned
    learning rate. Its target is one categorical teacher candidate retained
    only when a delayed factual callback gives singleton causal support for
    that route.
    """

    name = "RFT"

    def __init__(
        self,
        policy: StudentPolicy,
        icl_examples: int = ICL_K,
        rag_examples: int = RAG_K,
        rft_settings: RFTSettings = DEFAULT_RFT_SETTINGS,
    ):
        self.rft_settings = rft_settings
        super().__init__(
            policy,
            icl_examples,
            rag_examples,
            rft_settings.student_settings,
        )
        self.last_rft_candidate_action: int | None = None
        self.last_rft_accepted: bool | None = None
        self.last_rft_reason: str | None = None

    @staticmethod
    def rft_reason(
        action: int,
        feedback: dict,
        candidate_action: int | None,
    ) -> str:
        """Classify one teacher sample using only public causal support."""
        reliability = causal_evidence_reliability(action, feedback)
        if reliability == "censored_unknown":
            return "censored_unknown"
        if candidate_action is None:
            raise ValueError("RFT requires one sampled teacher candidate")
        if candidate_action < 0 or candidate_action >= len(ACTIONS):
            raise ValueError("RFT teacher candidate is out of range")
        if reliability == "ambiguous_digest_open":
            return "ambiguous_unverified"
        support = causal_route_support(action, feedback)
        if support[candidate_action]:
            return "accepted"
        return "teacher_mismatch"

    def observe(
        self,
        observation: StudentObservation,
        action: int,
        teacher_distribution: np.ndarray | None,
        teacher_action: int | None,
        feedback: dict,
        rng: np.random.Generator,
        teacher_observation: TeacherObservation | None = None,
        decision_distribution: np.ndarray | None = None,
        candidate_action: int | None = None,
        behavior_distribution: np.ndarray | None = None,
    ) -> None:
        reason = self.rft_reason(action, feedback, candidate_action)
        self.last_rft_candidate_action = candidate_action
        self.last_rft_reason = reason
        self.last_rft_accepted = (
            None
            if reason == "censored_unknown"
            else reason == "accepted"
        )
        super().observe(
            observation,
            action,
            teacher_distribution,
            teacher_action,
            feedback,
            rng,
            teacher_observation=teacher_observation,
            decision_distribution=decision_distribution,
            candidate_action=candidate_action,
            behavior_distribution=behavior_distribution,
        )

    def training_target(
        self,
        teacher_distribution: np.ndarray,
        teacher_action: int,
        *,
        action: int,
        feedback: dict,
        decision_distribution: np.ndarray | None,
        candidate_action: int | None = None,
    ) -> np.ndarray | None:
        """Return one-hot CE supervision only for a verified teacher sample."""
        del teacher_distribution, teacher_action, decision_distribution
        if self.rft_reason(action, feedback, candidate_action) != "accepted":
            return None
        target = np.zeros(len(ACTIONS), dtype=float)
        target[candidate_action] = 1.0
        return target


AGENT_CLASSES = {
    agent_class.name: agent_class
    for agent_class in (
        BaseAgent,
        ICLAgent,
        RAGAgent,
        REINFORCEAgent,
        RFTAgent,
        OnlineSDFTAgent,
    )
}


def create_agent(
    method: str,
    policy: StudentPolicy,
    icl_examples: int = ICL_K,
    rag_examples: int = RAG_K,
    rag_text_weight: float = RAG_TEXT_WEIGHT,
    sdft_settings: OnlineSDFTSettings = DEFAULT_SDFT_SETTINGS,
    reinforce_settings: REINFORCESettings = DEFAULT_REINFORCE_SETTINGS,
    rft_settings: RFTSettings = DEFAULT_RFT_SETTINGS,
) -> OnlineAgent:
    """Construct one named method or fail loudly on an invalid benchmark arm."""
    try:
        agent_class = AGENT_CLASSES[method]
    except KeyError as error:
        raise ValueError(f"unknown method {method!r}") from error
    if agent_class is RFTAgent:
        return agent_class(
            policy,
            icl_examples,
            rag_examples,
            rft_settings,
        )
    if agent_class is OnlineSDFTAgent:
        return agent_class(
            policy,
            icl_examples,
            rag_examples,
            sdft_settings,
        )
    if agent_class is RAGAgent:
        return agent_class(
            policy,
            icl_examples,
            rag_examples,
            rag_text_weight,
        )
    if agent_class is REINFORCEAgent:
        return agent_class(
            policy,
            icl_examples,
            rag_examples,
            reinforce_settings,
            sdft_settings,
        )
    return agent_class(policy, icl_examples, rag_examples)
