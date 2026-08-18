"""Prequential experiment loop and artifact orchestration.

This module is the only place where a method interacts with the environment.
The ordering is explicit: release any older feedback whose observation window
has closed, observe the new context, act, freeze the score, execute one route,
and queue its callback for a later round.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import (
    ACTIONS,
    DATASET_NUMPY_VERSION,
    DATASET_VERSION,
    DECISION_INTERVAL_MINUTES,
    DIGEST_DELIVERY_DELAY_MINUTES,
    EXPLORATION_EPSILON,
    FEEDBACK_WINDOWS_MINUTES,
    FIG,
    ICL_K,
    METHODS,
    MODEL_ID,
    ONLINE_BATCH_SIZE,
    OBSERVED_OUTCOME_REWARDS,
    PREFERENCE_SAMPLING_TEMPERATURE,
    PROMPT_STYLE,
    PROMPT_TOKEN_BUDGET,
    OUT,
    PHASE_LENGTH,
    RAG_K,
    RAG_TEXT_WEIGHT,
    REGIMES,
    REPLAY_SIZE,
    RFT_PROTOCOL_VERSION,
    RFT_SETTINGS_PROVENANCE,
    SDFT_DISTILL_TEMPERATURE,
    STREAM_LENGTH,
    TEACHER_PROMPT_VERSION,
)
from .environment import (
    DEFAULT_ENVIRONMENT,
    Event,
    NotificationRoutingEnvironment,
    one_hot,
)
from .methods import (
    DEFAULT_REINFORCE_SETTINGS,
    DEFAULT_RFT_SETTINGS,
    DEFAULT_SDFT_SETTINGS,
    LiquidLLMPolicy,
    OnlineSDFTSettings,
    REINFORCESettings,
    RFTSettings,
    StudentPolicy,
    causal_evidence_reliability,
    causal_route_support,
    create_agent,
    feedback_surface_propensity,
)
from .reporting import write_compact_results, write_figures
from .privilege import project_factual_callback


METHOD_RNG_OFFSETS = {
    "Base": 5,
    "ICL": 18,
    "RAG": 31,
    "RFT": 44,
    "Online-SDFT": 57,
    "REINFORCE": 70,
}
RFT_CANDIDATE_RNG_OFFSET = 83
RFT_CANDIDATE_SAMPLER = "blake2b-event-keyed-uniform-inverse-cdf-v1"


def epsilon_greedy(
    probs: np.ndarray,
    epsilon: float = EXPLORATION_EPSILON,
) -> np.ndarray:
    """Return the declared serving distribution over the student's ranking."""
    if (
        isinstance(epsilon, (bool, np.bool_))
        or not isinstance(epsilon, (int, float, np.integer, np.floating))
        or not np.isfinite(epsilon)
        or not 0.0 <= epsilon <= 1.0
    ):
        raise ValueError("serving epsilon must be finite and in [0, 1]")
    probs = np.asarray(probs, dtype=float)
    if not np.isfinite(probs).all():
        probs = np.ones(len(ACTIONS), dtype=float)
    greedy = one_hot(int(np.argmax(probs)), len(ACTIONS))
    behavior = (
        (1 - epsilon) * greedy
        + epsilon / len(ACTIONS)
    )
    return behavior / behavior.sum()


def policy_sampling(probs: np.ndarray) -> np.ndarray:
    """Normalize the differentiable policy used by REINFORCE to act."""
    behavior = np.asarray(probs, dtype=float)
    if not np.isfinite(behavior).all():
        behavior = np.ones(len(ACTIONS), dtype=float)
    behavior = np.clip(behavior, 1e-8, None)
    return behavior / behavior.sum()


def rft_sampling_distribution(
    teacher_probs: np.ndarray,
    sampling_temperature: float,
) -> np.ndarray:
    """Temperature-scale the original teacher distribution for RFT only."""
    values = np.asarray(teacher_probs, dtype=float)
    if values.shape != (len(ACTIONS),):
        raise ValueError("RFT teacher probabilities must match the action count")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(
            "RFT teacher probabilities must be finite and non-negative"
        )
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("RFT teacher probabilities must have positive mass")
    temperature = float(sampling_temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("RFT sampling temperature must be positive and finite")
    values = values / total
    logits = np.full_like(values, -np.inf)
    positive = values > 0.0
    logits[positive] = np.log(values[positive]) / temperature
    logits -= np.max(logits)
    proposals = np.exp(logits)
    return proposals / proposals.sum()


def rft_event_uniform(
    seed: int,
    event_id: str,
    step: int,
    *,
    rng_offset: int | None = None,
) -> float:
    """Return a stable event-keyed uniform independent of rollout control flow."""
    offset = RFT_CANDIDATE_RNG_OFFSET if rng_offset is None else int(rng_offset)
    key = json.dumps(
        [RFT_CANDIDATE_SAMPLER, int(seed), str(event_id), int(step), offset],
        separators=(",", ":"),
    ).encode("utf-8")
    bits = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    mantissa = bits >> 11
    return (mantissa + 0.5) / (1 << 53)


def rft_inverse_cdf_sample(probs: np.ndarray, uniform: float) -> int:
    """Map an event-keyed uniform to one categorical route."""
    values = np.asarray(probs, dtype=float)
    if values.shape != (len(ACTIONS),):
        raise ValueError("RFT proposal probabilities must match the action count")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(
            "RFT proposal probabilities must be finite and non-negative"
        )
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("RFT proposal probabilities must have positive mass")
    draw = float(uniform)
    if not np.isfinite(draw) or not 0.0 <= draw < 1.0:
        raise ValueError("RFT categorical uniform must be in [0, 1)")
    cumulative = np.cumsum(values / total)
    cumulative[-1] = 1.0
    return min(
        int(np.searchsorted(cumulative, draw, side="right")),
        len(ACTIONS) - 1,
    )


def archive_uniform_probe(
    probs: np.ndarray,
    mix: float,
    baseline_epsilon: float = EXPLORATION_EPSILON,
) -> np.ndarray:
    """Probe feedback-bearing routes at a fixed rate when ARCHIVE ranks first."""
    behavior = epsilon_greedy(probs, baseline_epsilon)
    if int(np.argmax(np.asarray(probs, dtype=float))) != ACTIONS.index("ARCHIVE"):
        return behavior
    probe = np.zeros(len(ACTIONS), dtype=float)
    probe[ACTIONS.index("INTERRUPT")] = 0.5
    probe[ACTIONS.index("LATER")] = 0.5
    behavior = (1.0 - mix) * behavior + mix * probe
    return behavior / behavior.sum()


def uncertainty_interrupt_probe(
    probs: np.ndarray,
    step: int,
    mix: float,
    half_life: float,
    max_confidence: float,
    baseline_epsilon: float = EXPLORATION_EPSILON,
) -> np.ndarray:
    """Add a decaying INTERRUPT probe using only pre-action learner state.

    The epsilon-greedy baseline retains positive support for every route.  The
    extra dose is active only while the normalized student distribution has no
    action above ``max_confidence``; it decays exponentially from ``mix`` at
    the first 1-based decision step.
    """
    values = np.asarray(probs, dtype=float)
    if values.shape != (len(ACTIONS),):
        raise ValueError("student probabilities must match the action count")
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, np.integer))
        or step < 1
    ):
        raise ValueError("interrupt probe step must be a positive integer")
    if (
        isinstance(mix, bool)
        or not isinstance(mix, (int, float, np.integer, np.floating))
        or not np.isfinite(mix)
        or not 0.0 < mix < 1.0
    ):
        raise ValueError("interrupt probe mix must be finite and in (0, 1)")
    if (
        isinstance(half_life, bool)
        or not isinstance(half_life, (int, float, np.integer, np.floating))
        or not np.isfinite(half_life)
        or half_life <= 0.0
    ):
        raise ValueError("interrupt probe half-life must be finite and positive")
    minimum_confidence = 1.0 / len(ACTIONS)
    if (
        isinstance(max_confidence, bool)
        or not isinstance(
            max_confidence,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(max_confidence)
        or not minimum_confidence <= max_confidence <= 1.0
    ):
        raise ValueError(
            "interrupt probe maximum confidence must be finite and in "
            f"[{minimum_confidence}, 1]"
        )

    behavior = epsilon_greedy(values, baseline_epsilon)
    confidence = float(np.max(policy_sampling(values)))
    if confidence > max_confidence:
        return behavior
    effective_mix = float(mix) * 2.0 ** (-(int(step) - 1) / float(half_life))
    probe = one_hot(ACTIONS.index("INTERRUPT"), len(ACTIONS))
    behavior = (1.0 - effective_mix) * behavior + effective_mix * probe
    return behavior / behavior.sum()


def archive_policy_feedback_floor(
    probs: np.ndarray,
    minimum_feedback: float,
) -> np.ndarray:
    """Floor feedback-bearing policy mass only when ARCHIVE ranks first."""
    behavior = np.asarray(probs, dtype=float)
    if not np.isfinite(behavior).all():
        behavior = np.ones(len(ACTIONS), dtype=float)
    behavior = np.clip(behavior, 0.0, None)
    total = float(behavior.sum())
    if total <= 0.0:
        behavior = np.ones(len(ACTIONS), dtype=float)
        total = float(behavior.sum())
    behavior = behavior / total

    archive_index = ACTIONS.index("ARCHIVE")
    if int(np.argmax(behavior)) != archive_index:
        return behavior
    feedback_indices = np.asarray(
        [ACTIONS.index("INTERRUPT"), ACTIONS.index("LATER")],
        dtype=int,
    )
    feedback_mass = float(behavior[feedback_indices].sum())
    if feedback_mass >= minimum_feedback:
        return behavior

    if np.isclose(feedback_mass, 0.0, rtol=0.0, atol=1e-12):
        feedback_ratio = np.asarray([0.5, 0.5], dtype=float)
    else:
        feedback_ratio = behavior[feedback_indices] / feedback_mass
    floored = behavior.copy()
    floored[feedback_indices] = minimum_feedback * feedback_ratio
    floored[archive_index] = 1.0 - minimum_feedback
    return floored


def run_method(
    seed: int,
    method: str,
    stream: list[Event],
    policy: StudentPolicy,
    rollout_writer,
    curve_writer,
    environment: NotificationRoutingEnvironment = DEFAULT_ENVIRONMENT,
    icl_examples: int = ICL_K,
    rag_examples: int = RAG_K,
    rag_text_weight: float = RAG_TEXT_WEIGHT,
    sdft_settings: OnlineSDFTSettings = DEFAULT_SDFT_SETTINGS,
    reinforce_settings: REINFORCESettings = DEFAULT_REINFORCE_SETTINGS,
    rft_settings: RFTSettings = DEFAULT_RFT_SETTINGS,
) -> dict:
    """Run one method with delayed, action-dependent feedback."""
    # Common random numbers pair serving exploration and simulator draws across
    # methods. Method-specific randomness is isolated to replay, so an extra
    # update cannot perturb a later action draw.
    action_rng = np.random.default_rng(seed * 1000 + 1)
    feedback_rng = np.random.default_rng(seed * 1000 + 2)
    learning_rng = np.random.default_rng(
        seed * 1000 + METHOD_RNG_OFFSETS[method]
    )
    agent = create_agent(
        method=method,
        policy=policy,
        icl_examples=icl_examples,
        rag_examples=rag_examples,
        rag_text_weight=rag_text_weight,
        sdft_settings=sdft_settings,
        reinforce_settings=reinforce_settings,
        rft_settings=rft_settings,
    )
    cumulative_regret = 0.0
    cumulative_correct = 0
    cumulative_observed_reward = 0.0
    phase_correct = Counter()
    phase_total = Counter()
    phase_regret = Counter()
    pending_feedback: list[dict] = []
    rollout_records: list[dict] = []
    reinforce_batch_records: list[dict] = []

    def release_ready_feedback(now_minute: int) -> None:
        """Apply only lessons whose real observation window has closed."""
        ready = [
            item
            for item in pending_feedback
            if item["available_at_minute"] <= now_minute
        ]
        pending_feedback[:] = [
            item
            for item in pending_feedback
            if item["available_at_minute"] > now_minute
        ]
        for item in ready:
            teacher_probs = None
            teacher_action = None
            rft_candidate_action = None
            rft_candidate_probs = None
            rft_candidate_entropy = None
            rft_candidate_uniform = None
            teacher_assessment = None
            teacher_view = None
            if agent.uses_teacher:
                teacher_view = environment.teacher_observation(
                    item["observation"],
                    item["action"],
                    project_factual_callback(item["feedback"]),
                )
                if teacher_view.observed_user_selection != "UNKNOWN":
                    teacher_probs = policy.teacher_probs(teacher_view)
                    teacher_action = int(np.argmax(teacher_probs))
                    if method == "RFT":
                        rft_candidate_probs = rft_sampling_distribution(
                            teacher_probs,
                            rft_settings.sampling_temperature,
                        )
                        rft_candidate_uniform = rft_event_uniform(
                            seed,
                            item["record"]["event_id"],
                            item["record"]["t"],
                        )
                        rft_candidate_action = rft_inverse_cdf_sample(
                            rft_candidate_probs,
                            rft_candidate_uniform,
                        )
                        positive = rft_candidate_probs[
                            rft_candidate_probs > 0.0
                        ]
                        rft_candidate_entropy = float(
                            -(positive * np.log(positive)).sum()
                        )
                    teacher_assessment = getattr(
                        policy,
                        "last_teacher_assessment",
                        None,
                    )

            agent.observe(
                item["observation"],
                item["action"],
                teacher_probs,
                teacher_action,
                item["feedback"],
                learning_rng,
                teacher_observation=teacher_view,
                decision_distribution=item["decision_distribution"],
                candidate_action=rft_candidate_action,
                behavior_distribution=item["behavior_distribution"],
            )

            record = item["record"]
            record["feedback_released_at_minute"] = now_minute
            if method in {"RFT", "Online-SDFT"}:
                support = causal_route_support(
                    item["action"],
                    item["feedback"],
                )
                record["causal_support"] = [
                    ACTIONS[index]
                    for index, allowed in enumerate(support)
                    if allowed
                ]
            if method == "RFT":
                record["rft_candidate_probs"] = (
                    None
                    if rft_candidate_probs is None
                    else dict(zip(ACTIONS, map(float, rft_candidate_probs)))
                )
                record["rft_candidate_entropy"] = rft_candidate_entropy
                record["rft_candidate_uniform"] = rft_candidate_uniform
                record["rft_candidate_action"] = (
                    None
                    if agent.last_rft_candidate_action is None
                    else ACTIONS[agent.last_rft_candidate_action]
                )
                record["rft_accepted"] = agent.last_rft_accepted
                record["rft_reason"] = agent.last_rft_reason
                record["rft_update_index"] = agent.online_update_count
                record["rft_updates_applied"] = (
                    agent.last_observation_update_count
                )
            if method == "Online-SDFT":
                record["sdft_update_index"] = agent.online_update_count
                record["sdft_updates_applied"] = (
                    agent.last_observation_update_count
                )
                record["sdft_objective"] = agent.settings.target_mode
                reliability = causal_evidence_reliability(
                    item["action"],
                    item["feedback"],
                )
                record["sdft_evidence_reliability"] = reliability
                if (
                    reliability != "censored_unknown"
                    and agent.settings.target_mode != "support_likelihood"
                ):
                    weights = agent.fusion_weights(
                        item["action"],
                        item["feedback"],
                    )
                    record["sdft_fusion_weights"] = dict(
                        zip(("teacher", "decision", "behavior"), weights)
                    )
            if method in {"ICL", "RAG"}:
                memory_reliability = causal_evidence_reliability(
                    item["action"],
                    item["feedback"],
                )
                record["memory_evidence_reliability"] = memory_reliability
                record["lesson_status"] = {
                    "reliable_singleton": "memory_prompt_available",
                    "ambiguous_digest_open": "memory_retained_ambiguous",
                    "censored_unknown": "memory_unlabeled",
                }[memory_reliability]
            elif item["feedback"]["observed_user_selection"] == "UNKNOWN":
                record["lesson_status"] = "censored_no_update"
            elif method == "REINFORCE":
                record["reinforce_training_reward"] = (
                    agent.last_training_reward
                )
                record["reinforce_batch_position"] = (
                    len(reinforce_batch_records) + 1
                )
                reinforce_batch_records.append(record)
                if agent.last_observation_update_count:
                    update_index = agent.online_update_count
                    for batch_record in reinforce_batch_records:
                        batch_record["reinforce_update_index"] = update_index
                        batch_record["lesson_status"] = "feedback_applied"
                    reinforce_batch_records.clear()
                else:
                    record["lesson_status"] = "feedback_buffered"
            elif method == "RFT":
                if record["rft_accepted"]:
                    record["lesson_status"] = (
                        "rft_target_applied"
                        if agent.last_observation_update_count
                        else "rft_target_buffered"
                    )
                else:
                    record["lesson_status"] = {
                        "ambiguous_unverified": (
                            "rft_rejected_ambiguous_support"
                        ),
                        "teacher_mismatch": (
                            "rft_rejected_teacher_candidate"
                        ),
                    }[record["rft_reason"]]
            else:
                if method == "Online-SDFT":
                    if (
                        record["sdft_evidence_reliability"]
                        == "ambiguous_digest_open"
                        and agent.settings.ambiguous_update_mode
                        in {"skip", "defer"}
                    ):
                        record["lesson_status"] = {
                            "skip": "ambiguous_target_skipped",
                            "defer": "ambiguous_target_deferred",
                        }[agent.settings.ambiguous_update_mode]
                    else:
                        target_kind = (
                            "support_target"
                            if agent.settings.target_mode == "support_likelihood"
                            else "soft_target"
                        )
                        record["lesson_status"] = (
                            f"{target_kind}_applied"
                            if agent.last_observation_update_count
                            else f"{target_kind}_buffered"
                        )
                else:
                    record["lesson_status"] = "observed_no_update"
            record["teacher_evidence"] = (
                None if teacher_view is None else teacher_view.evidence
            )
            record["teacher_probs"] = (
                None
                if teacher_probs is None
                else dict(zip(ACTIONS, map(float, teacher_probs)))
            )
            record["teacher_assessment"] = teacher_assessment
            record["teacher_rollout"] = (
                None
                if teacher_action is None
                else ACTIONS[teacher_action]
            )

    for step, event in enumerate(stream, start=1):
        current_minute = (step - 1) * DECISION_INTERVAL_MINUTES
        release_ready_feedback(current_minute)
        observation = environment.student_observation(event)
        student_probs = agent.action_probs(observation)
        behavior_mode = getattr(agent, "behavior_mode", "epsilon_greedy")
        behavior_epsilon = (
            getattr(agent, "behavior_epsilon", EXPLORATION_EPSILON)
            if method == "Online-SDFT"
            else EXPLORATION_EPSILON
        )
        behavior_epsilon_half_life = (
            getattr(agent, "behavior_epsilon_half_life", None)
            if method == "Online-SDFT"
            else None
        )
        if behavior_epsilon_half_life is not None:
            behavior_epsilon = float(behavior_epsilon) * 2.0 ** (
                -(step - 1) / float(behavior_epsilon_half_life)
            )
        samples_on_archive = (
            behavior_mode == "archive_policy_sampling"
            and int(np.argmax(student_probs)) == ACTIONS.index("ARCHIVE")
        )
        if (
            agent.samples_from_policy
            or behavior_mode == "policy_sampling"
            or samples_on_archive
        ):
            behavior_probs = policy_sampling(student_probs)
        elif behavior_mode == "archive_uniform_probe":
            behavior_probs = archive_uniform_probe(
                student_probs,
                getattr(agent, "archive_probe_mix", 0.0),
                behavior_epsilon,
            )
        elif behavior_mode == "archive_policy_feedback_floor":
            behavior_probs = archive_policy_feedback_floor(
                student_probs,
                getattr(agent, "archive_policy_min_feedback", 0.0),
            )
        elif behavior_mode == "uncertainty_interrupt_probe":
            behavior_probs = uncertainty_interrupt_probe(
                student_probs,
                step,
                getattr(agent, "interrupt_probe_mix", 0.0),
                getattr(agent, "interrupt_probe_half_life", None),
                getattr(agent, "interrupt_probe_max_confidence", 1.0),
                behavior_epsilon,
            )
        else:
            behavior_probs = epsilon_greedy(student_probs, behavior_epsilon)
        exploration_taper_weight = 1.0
        exploration_taper_start_step = (
            getattr(agent, "exploration_taper_start_step", None)
            if method == "Online-SDFT"
            else None
        )
        if (
            exploration_taper_start_step is not None
            and step > exploration_taper_start_step
        ):
            exploration_taper_half_life = getattr(
                agent,
                "exploration_taper_half_life",
                None,
            )
            exploration_taper_weight = 2.0 ** (
                -(step - exploration_taper_start_step)
                / float(exploration_taper_half_life)
            )
            greedy_behavior = one_hot(
                int(np.argmax(student_probs)),
                len(ACTIONS),
            )
            behavior_probs = (
                exploration_taper_weight * behavior_probs
                + (1.0 - exploration_taper_weight) * greedy_behavior
            )
            behavior_probs = behavior_probs / behavior_probs.sum()
        feedback_propensity = feedback_surface_propensity(behavior_probs)
        action = int(
            action_rng.choice(len(ACTIONS), p=behavior_probs)
        )

        # Freeze evaluation before any factual outcome or update exists.
        utilities = environment.oracle_utilities(event)
        gold_action = environment.gold_action(event)
        utility_optimal_action = int(np.argmax(utilities))
        step_regret = float(
            utilities[utility_optimal_action] - utilities[action]
        )
        correct = int(action == gold_action)
        cumulative_regret += step_regret
        cumulative_correct += correct
        phase_correct[event.phase] += correct
        phase_total[event.phase] += 1
        phase_regret[event.phase] += step_regret

        # Execute only the chosen route. The simulator samples the future
        # callback now but the learner cannot consume it until its declared
        # action-specific observation window has elapsed.
        feedback = environment.execute(event, action, feedback_rng)
        observed_reward = float(feedback["reward"])
        cumulative_observed_reward += observed_reward
        feedback_available_at = current_minute + int(
            feedback["delay_minutes"]
        )

        record = {
            "seed": seed,
            "method": method,
            "t": step,
            "event_id": event.event_id,
            "phase": event.phase,
            "regime": REGIMES[event.phase],
            "category": event.category,
            "notification_title": event.title,
            "notification_body": event.body,
            "decision_time_minute": current_minute,
            "student_probs": dict(
                zip(ACTIONS, map(float, student_probs))
            ),
            "behavior_mode": behavior_mode,
            "behavior_epsilon": float(behavior_epsilon),
            "exploration_taper_weight": (
                float(exploration_taper_weight)
                if method == "Online-SDFT"
                else None
            ),
            "behavior_probs": dict(
                zip(ACTIONS, map(float, behavior_probs))
            ),
            "action": ACTIONS[action],
            "feedback": feedback,
            "feedback_available_at_minute": feedback_available_at,
            "feedback_released_at_minute": None,
            "lesson_status": "pending",
            "reinforce_training_reward": None,
            "reinforce_update_index": None,
            "reinforce_batch_position": None,
            "rft_candidate_action": None,
            "rft_candidate_probs": None,
            "rft_candidate_entropy": None,
            "rft_candidate_uniform": None,
            "rft_accepted": None,
            "rft_reason": None,
            "rft_update_index": None,
            "rft_updates_applied": None,
            "sdft_update_index": None,
            "sdft_updates_applied": None,
            "sdft_prompt_examples_used": (
                agent.last_prompt_examples_used
                if method == "Online-SDFT"
                else None
            ),
            "causal_support": None,
            "sdft_evidence_reliability": None,
            "sdft_objective": None,
            "sdft_fusion_weights": None,
            "sdft_feedback_propensity": (
                feedback_propensity if method == "Online-SDFT" else None
            ),
            "sdft_propensity_weight": (
                min(
                    sdft_settings.propensity_weight_cap,
                    1.0 / max(feedback_propensity, 1e-8),
                )
                if method == "Online-SDFT"
                and sdft_settings.propensity_weight_mode
                == "feedback_surface_snips"
                else None
            ),
            "teacher_evidence": None,
            "teacher_probs": None,
            "teacher_assessment": None,
            "teacher_rollout": None,
            "gold_action_scoring_only": ACTIONS[gold_action],
            "gold_action_distribution_scoring_only": dict(
                zip(
                    ACTIONS,
                    map(float, environment.gold_action_distribution(event)),
                )
            ),
            "utility_optimal_action_scoring_only": ACTIONS[
                utility_optimal_action
            ],
            "correct_online": correct,
            "observed_feedback_reward": observed_reward,
            "step_regret": step_regret,
            "cum_regret": cumulative_regret,
            "cum_accuracy": cumulative_correct / step,
            "cum_observed_reward": cumulative_observed_reward,
        }
        rollout_records.append(record)
        pending_feedback.append(
            {
                "available_at_minute": feedback_available_at,
                "event": event,
                "observation": observation,
                "action": action,
                "feedback": feedback,
                "decision_distribution": student_probs.copy(),
                "behavior_distribution": behavior_probs.copy(),
                "feedback_propensity": feedback_propensity,
                "record": record,
            }
        )
        if step % 20 == 0:
            print(
                f"  {method} t={step}/{len(stream)} "
                f"acc={cumulative_correct / step:.3f}",
                flush=True,
            )

        curve_writer.writerow(
            {
                "seed": seed,
                "method": method,
                "t": step,
                "phase": event.phase,
                "regime": REGIMES[event.phase],
                "step_correct": correct,
                "step_feedback_reward": observed_reward,
                "step_regret": step_regret,
                "cum_accuracy": cumulative_correct / step,
                "cum_regret": cumulative_regret,
                "cum_observed_reward": cumulative_observed_reward,
            }
        )

    # Do not flush future feedback after the evaluation horizon: a real model
    # could use it later, but it cannot improve any action scored in this run.
    for record in reinforce_batch_records:
        if record["lesson_status"] == "feedback_buffered":
            record["lesson_status"] = "feedback_gradient_unflushed_at_horizon"
    for record in rollout_records:
        if record["lesson_status"] == "pending":
            record["lesson_status"] = "pending_after_horizon"
        rollout_writer.write(json.dumps(record) + "\n")
    rollout_writer.flush()

    return {
        "seed": seed,
        "method": method,
        "online_accuracy": cumulative_correct / len(stream),
        "cum_regret": cumulative_regret,
        "regret_per_decision": cumulative_regret / len(stream),
        "cumulative_observed_reward": cumulative_observed_reward,
        "observed_reward_per_decision": (
            cumulative_observed_reward / len(stream)
        ),
        **{
            f"online_accuracy_{REGIMES[phase]}": (
                phase_correct[phase] / phase_total[phase]
                if phase_total[phase]
                else 0.0
            )
            for phase in range(3)
        },
        **{
            f"regret_{REGIMES[phase]}": phase_regret[phase]
            for phase in range(3)
        },
    }


def experiment_config(
    seeds: int,
    seed_start: int,
    model_id: str,
    policy: LiquidLLMPolicy,
    prompt_style: str = PROMPT_STYLE,
    icl_examples: int = ICL_K,
    rag_examples: int = RAG_K,
    rag_text_weight: float = RAG_TEXT_WEIGHT,
    sdft_settings: OnlineSDFTSettings = DEFAULT_SDFT_SETTINGS,
    environment: NotificationRoutingEnvironment = DEFAULT_ENVIRONMENT,
    reinforce_settings: REINFORCESettings = DEFAULT_REINFORCE_SETTINGS,
    rft_settings: RFTSettings = DEFAULT_RFT_SETTINGS,
) -> dict:
    dataset_fingerprint = environment.stream_fingerprint(
        range(seed_start, seed_start + seeds)
    )
    sdft_trainable_parameters = int(policy.trainable_parameters)
    rft_student_settings = rft_settings.student_settings

    def peft_architecture(settings: OnlineSDFTSettings) -> dict:
        targets = list(settings.lora_target_modules)
        layers = settings.lora_layers_to_transform
        serialized_layers = None if layers is None else list(layers)
        return {
            "implementation": "peft.LoraConfig + peft.get_peft_model",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": settings.lora_rank,
            "lora_alpha": settings.lora_alpha,
            "lora_dropout": settings.lora_dropout,
            "target_modules": targets,
            "layers_to_transform": serialized_layers,
            "layers_pattern": None if layers is None else "layers",
            "bias": "none",
            "init_lora_weights": True,
            "ensure_weight_tying": False,
            "merged_for_serving": False,
        }

    sdft_peft_architecture = peft_architecture(sdft_settings)
    reinforce_peft_architecture = sdft_peft_architecture.copy()
    rft_peft_architecture = peft_architecture(rft_student_settings)
    same_rft_architecture = (
        rft_peft_architecture == sdft_peft_architecture
    )
    trainability_fields = (
        "lora_a_learning_rate_scale",
        "lm_head_lora_a_learning_rate_scale",
    )
    same_rft_adapter_capacity = same_rft_architecture and all(
        getattr(rft_student_settings, field) == getattr(sdft_settings, field)
        for field in trainability_fields
    )
    rft_trainable_parameters = (
        sdft_trainable_parameters if same_rft_adapter_capacity else None
    )
    replay_schedule_fields = (
        "replay_size",
        "replay_prompt_examples",
        "batch_size",
        "update_steps",
        "warmup_examples",
        "ambiguous_replay_group_weight",
        "teacher_temperature",
        "reasoning_tokens",
        "replay_strategy",
        "replay_recency_half_life",
        "ambiguous_update_mode",
        "force_newest_every_step",
        "base_kl_weight",
        "behavior_mode",
        "behavior_epsilon",
        "behavior_epsilon_half_life",
        "exploration_taper_start_step",
        "exploration_taper_half_life",
        "archive_probe_mix",
        "archive_policy_min_feedback",
        "interrupt_probe_mix",
        "interrupt_probe_half_life",
        "interrupt_probe_max_confidence",
        "propensity_weight_mode",
        "propensity_weight_cap",
    )
    same_rft_replay_schedule = all(
        getattr(rft_student_settings, field) == getattr(sdft_settings, field)
        for field in replay_schedule_fields
    )
    optimizer_fields = (
        "learning_rate",
        "optimizer_weight_decay",
        "optimizer_beta1",
        "max_grad_norm",
        "lm_head_learning_rate",
        "lora_a_learning_rate_scale",
        "lm_head_lora_a_learning_rate_scale",
    )
    same_rft_optimizer_hyperparameters = all(
        getattr(rft_student_settings, field) == getattr(sdft_settings, field)
        for field in optimizer_fields
    )
    return {
        "seeds": seeds,
        "seed_start": seed_start,
        "dataset_version": DATASET_VERSION,
        "dataset_numpy_version": DATASET_NUMPY_VERSION,
        "runtime_numpy_version": np.__version__,
        "dataset_fingerprint": dataset_fingerprint,
        "method_dataset_versions": {
            method: DATASET_VERSION for method in METHODS
        },
        "method_dataset_fingerprints": {
            method: dataset_fingerprint for method in METHODS
        },
        "paired_method_stream": (
            "one immutable generated event stream per seed is reused by all "
            "methods before any method-specific action or feedback randomness"
        ),
        "gold_action_sampling": {
            "distribution": (
                "probability-power temperature scaling of normalized "
                "evaluator-utility weights"
            ),
            "negative_utility_handling": (
                "subtract the event minimum only when it is negative"
            ),
            "temperature": PREFERENCE_SAMPLING_TEMPERATURE,
            "exponent": 1.0 / PREFERENCE_SAMPLING_TEMPERATURE,
            "formula": (
                "q_i = p_i**(1 / temperature) / "
                "sum_j p_j**(1 / temperature)"
            ),
            "numerical_implementation": (
                "stable log space with exact zero support preserved"
            ),
            "draw_timing": "once per event from an isolated seeded RNG",
        },
        "notification_context": (
            "decision-relevant synthetic title/body + category + local time "
            "+ regime + local importance"
        ),
        "stream_length": STREAM_LENGTH,
        "phase_length": PHASE_LENGTH,
        "decision_interval_minutes": DECISION_INTERVAL_MINUTES,
        "digest_delivery_delay_minutes": DIGEST_DELIVERY_DELAY_MINUTES,
        "feedback_windows_minutes": FEEDBACK_WINDOWS_MINUTES,
        "feedback_observation_matrix": {
            "INTERRUPT": (
                "immediate open -> INTERRUPT; delayed read -> LATER; "
                "notification deletion -> ARCHIVE"
            ),
            "LATER": (
                "digest open -> LATER, including after a missed immediate "
                "preference; digest deletion -> ARCHIVE"
            ),
            "ARCHIVE": "no delivered surface -> UNKNOWN",
        },
        "observed_reward": dict(OBSERVED_OUTCOME_REWARDS),
        "actions": ACTIONS,
        "methods": METHODS,
        "student_model": model_id,
        "student_policy": (
            "next-token A/B/C probabilities; REINFORCE, RFT, and "
            "Online-SDFT each train the same reset-per-arm PEFT LoRA adapter"
        ),
        "adaptive_model_sharing": (
            "one physical Liquid model instance is reused sequentially; its "
            "LoRA adapter is reset to the identical initialization before "
            "every method and seed. Only RFT and Online-SDFT disable the "
            "adapter for same-model hindsight teaching"
        ),
        "student_backbone": "frozen Liquid LFM base weights",
        "student_backbone_trainable_parameters": 0,
        "device": str(policy.device),
        "exploration_epsilon": EXPLORATION_EPSILON,
        "behavior_policy": (
            "Base, ICL, RAG, and RFT use 6% epsilon-greedy; REINFORCE "
            "samples its current LoRA policy; Online-SDFT uses its configured "
            "uncertainty-triggered INTERRUPT probe and post-step-160 taper"
        ),
        "randomness_pairing": (
            "common per-seed action and feedback RNG streams across methods; "
            "method-specific RNG only for learning internals; RFT teacher "
            "candidate sampling uses an event-keyed deterministic uniform "
            "that cannot "
            "shift action, feedback, or replay draws"
        ),
        "replay_size": REPLAY_SIZE,
        "online_batch_size": ONLINE_BATCH_SIZE,
        "prompt_style": prompt_style,
        "history_rendering": (
            "alternating notification/route demonstrations from reliable "
            "singleton callbacks only; ambiguous and UNKNOWN interactions "
            "remain retained but are not prompted"
        ),
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "prompt_token_budget": PROMPT_TOKEN_BUDGET,
        "icl_examples": icl_examples,
        "rag_examples": rag_examples,
        "rag_text_weight": rag_text_weight,
        "rag_similarity": (
            f"{1.0 - rag_text_weight:.2f} metadata similarity (equal-weight "
            "category/regime match + importance + circular hour) + "
            f"{rag_text_weight:.2f} visible title/body token Jaccard similarity"
        ),
        "reinforce_lr": reinforce_settings.learning_rate,
        "reinforce_batch_size": reinforce_settings.batch_size,
        "reinforce_policy": (
            "action-token REINFORCE trains only the same reset LFM PEFT LoRA "
            "adapter used by RFT and Online-SDFT"
        ),
        "reinforce_trainable_parameters": sdft_trainable_parameters,
        "online_reinforce_peft_architecture": reinforce_peft_architecture,
        "reinforce_initialization": (
            "the common PEFT LoRA initialization restored before this arm"
        ),
        "online_reinforce_settings": reinforce_settings.to_dict(),
        "reinforce_training_reward": {
            "source": "matured executed-surface factual outcome only",
            "outcome_map": dict(reinforce_settings.reward_outcome_map),
            "unknown_selection": (
                "censored before outcome mapping; no gradient target"
            ),
            "reported_metric": (
                "learner-only shaping; rollout observed_feedback_reward and "
                "cumulative_observed_reward use the shared observed_reward map"
            ),
        },
        "reinforce_optimizer": (
            f"AdamW autograd over exactly {reinforce_settings.batch_size} "
            "newly matured known factual-outcome callbacks; learner reward "
            "from reinforce_training_reward outcome map; each row consumed "
            "once; no replay; incomplete horizon batch not flushed"
        ),
        "reinforce_baseline": (
            "fixed zero causal baseline; step=0.0"
            if reinforce_settings.baseline_step == 0.0
            else f"causal reward EMA; step={reinforce_settings.baseline_step}"
        ),
        "reinforce_entropy_coef": reinforce_settings.entropy_coef,
        "reinforce_max_grad_norm": reinforce_settings.max_grad_norm,
        "reinforce_capacity_match": {
            "settings_source": "online_sdft_settings LoRA architecture",
            "same_physical_model": True,
            "same_frozen_base_model": True,
            "same_lora_architecture": True,
            "same_adapter_parameter_count": True,
            "same_adapter_initialization": True,
            "adapter_reset_before_arm": True,
            "uses_hindsight_teacher": False,
        },
        "rft_protocol_version": RFT_PROTOCOL_VERSION,
        "rft_settings_provenance": RFT_SETTINGS_PROVENANCE,
        "online_rft_settings": rft_settings.to_dict(),
        "online_rft_peft_architecture": rft_peft_architecture,
        "online_rft_trainable_parameters": rft_trainable_parameters,
        "rft_candidate_policy": (
            f"K={rft_settings.candidate_count} "
            f"{rft_settings.sampling_mode} sample at temperature "
            f"{rft_settings.sampling_temperature:g} from the same model with "
            "its adapter disabled, which is the fixed-initial hindsight "
            "distribution used by Online-SDFT"
        ),
        "rft_candidate_sampler": {
            "scheme": RFT_CANDIDATE_SAMPLER,
            "key_fields": [
                "seed",
                "event_id",
                "t",
                "rft_candidate_rng_offset",
            ],
            "inverse_cdf": True,
            "stateful_rng": False,
        },
        "rft_acceptance_filter": (
            "accept only when the delayed public causal support is a reliable "
            "singleton containing the sampled teacher route; reject teacher "
            "mismatches and every ambiguous digest open; UNKNOWN is censored"
        ),
        "rft_target": (
            "one-hot cross-entropy target on the accepted sampled teacher "
            "route; rejected and censored rows never enter replay"
        ),
        "rft_capacity_match": {
            "settings_source": "online_rft_settings.student_settings",
            "same_frozen_base_model": True,
            "same_lora_architecture": same_rft_architecture,
            "same_adapter_parameter_count": same_rft_adapter_capacity,
            "same_adapter_initialization": same_rft_architecture,
            "same_replay_schedule": same_rft_replay_schedule,
            "same_optimizer_hyperparameters": (
                same_rft_optimizer_hyperparameters
            ),
            "same_adapter_disabled_teacher": True,
            "same_teacher_forward_budget": (
                "one hindsight distribution for each matured non-UNKNOWN "
                "callback"
            ),
            "configured_differences": (
                "temperature-8 sample-filter hard targets and a tuned 7e-4 "
                "learning rate with replay-32 epsilon-greedy serving versus "
                "Online-SDFT's replay-64 recency/probe/taper schedule and "
                "reliability-conditioned soft targets"
            ),
        },
        "rft_candidate_rng_offset": RFT_CANDIDATE_RNG_OFFSET,
        "sdft_lr": sdft_settings.learning_rate,
        "sdft_replay_size": sdft_settings.replay_size,
        "sdft_replay_prompt_examples": sdft_settings.replay_prompt_examples,
        "sdft_batch_size": sdft_settings.batch_size,
        "sdft_update_steps": sdft_settings.update_steps,
        "sdft_warmup_examples": sdft_settings.warmup_examples,
        "sdft_distill_temperature": SDFT_DISTILL_TEMPERATURE,
        "online_sdft_settings": sdft_settings.to_dict(),
        "adaptive_default_configuration_provenance": {
            "Online-SDFT": {
                "selection_seeds": [0, 1, 2],
                "disjoint_confirmation": False,
                "interpretation": (
                    "user-requested in-sample tuning on the canonical streams"
                ),
            },
            "RFT": {
                "selection_seeds": [1200],
                "disjoint_confirmation": False,
                "interpretation": (
                    "single-stream temperature and learning-rate screen"
                ),
            },
            "REINFORCE": {
                "selection_seeds": [0, 1, 2],
                "disjoint_confirmation": False,
                "selection_rule": (
                    "strict candidate gate: pooled exact-action correct > "
                    "Base and pooled total regret < Base; rank accuracy first, "
                    "then regret"
                ),
                "interpretation": (
                    "user-requested in-sample tuning on the canonical streams; "
                    "no disjoint confirmation set"
                ),
            },
        },
        "online_sdft_peft_architecture": sdft_peft_architecture,
        "online_sdft_student": (
            "PEFT LoRA adapter trained online on the same Liquid LFM used "
            "for hindsight teaching; base-model weights stay frozen and the "
            "adapter is never merged"
        ),
        "online_sdft_trainable_parameters": sdft_trainable_parameters,
        "online_sdft_optimizer": {
            "type": "AdamW",
            "learning_rate": sdft_settings.learning_rate,
            "lm_head_learning_rate": sdft_settings.lm_head_learning_rate,
            "lora_a_learning_rate_scale": (
                sdft_settings.lora_a_learning_rate_scale
            ),
            "lm_head_lora_a_learning_rate_scale": (
                sdft_settings.lm_head_lora_a_learning_rate_scale
            ),
            "weight_decay": sdft_settings.optimizer_weight_decay,
            "max_grad_norm": sdft_settings.max_grad_norm,
            "batch_size": sdft_settings.batch_size,
            "update_steps_per_release": sdft_settings.update_steps,
        },
        "sdft_replay_prompt_policy": (
            "most-recent reliable singleton rows from the bounded SDFT "
            "training replay in FIFO release order; no similarity retrieval; "
            "ambiguous, UNKNOWN, teacher, and soft-target data excluded"
            if sdft_settings.replay_prompt_examples
            else "disabled; serving uses only learned finite parameters"
        ),
        "sdft_evidence_reliability": {
            "reliable_singleton": {
                "callbacks": (
                    "immediate/delayed interrupt open or deletion from a "
                    "delivered surface"
                ),
                "teacher_weight": sdft_settings.reliable_teacher_weight,
                "decision_weight": sdft_settings.reliable_decision_weight,
                "behavior_weight": sdft_settings.reliable_behavior_weight,
            },
            "ambiguous_digest_open": {
                "callbacks": (
                    "digest open supports INTERRUPT or LATER without "
                    "distinguishing them"
                ),
                "teacher_weight": sdft_settings.ambiguous_teacher_weight,
                "decision_weight": sdft_settings.ambiguous_decision_weight,
                "behavior_weight": sdft_settings.ambiguous_behavior_weight,
            },
            "censored_unknown": "no target or update",
        },
        "teacher_model": model_id,
        "teacher_student_model_sharing": {
            "model_instances": 1,
            "same_base_parameters": True,
            "student_forward": "LoRA adapter enabled",
            "teacher_forward": "same model with LoRA adapter disabled",
            "teacher_reference": "fixed initial frozen base-model policy",
            "separate_teacher_checkpoint": False,
        },
        "teacher_policy": (
            "the student and teacher are one shared Liquid LFM model; teacher "
            "forwards disable the student's LoRA adapter, so the frozen base "
            "is the fixed-initial reference for the full run; notification "
            "title/body + metadata + decision-time importance + delayed "
            "observed user selection are supplied for the executed route "
            "only; explicit delivery-surface guidance keeps a digest open "
            "ambiguous between INTERRUPT and LATER; UNKNOWN stays censored; "
            "no scalar reward, counterfactual, shadow policy, or evaluator "
            "label"
        ),
        "teacher_temperature": sdft_settings.teacher_temperature,
        "evaluation": (
            "prequential one-stream; sampled-preference accuracy and "
            "utility-optimal regret; predict then learn"
        ),
        "update_timing": (
            "feedback is queued until its action-specific observation window "
            "closes; no end-of-horizon flush"
        ),
        "learning_signal": (
            "evidence-reliability-conditioned same-LM soft hindsight target "
            "for Online-SDFT; the teacher's argmax is diagnostic only; "
            "RFT draws one separate categorical teacher candidate and keeps "
            "its hard target only after reliable singleton verification; "
            "ICL/RAG retain direct completed interactions but prompt only "
            "reliable singleton route evidence; learner-only scalar mapped "
            "from the matured executed-surface factual outcome for "
            "REINFORCE, while reported observed reward retains the shared "
            "environment map; hidden sampled preference for shared accuracy "
            "only; utility-optimal route for regret only"
        ),
    }


def main(
    seeds: int = 3,
    model_id: str = MODEL_ID,
    device: str = "auto",
    local_files_only: bool = False,
    seed_start: int = 0,
    output_dir: Path | None = None,
    figure_dir: Path | None = None,
    environment: NotificationRoutingEnvironment = DEFAULT_ENVIRONMENT,
    prompt_style: str = PROMPT_STYLE,
    icl_examples: int = ICL_K,
    rag_examples: int = RAG_K,
    rag_text_weight: float = RAG_TEXT_WEIGHT,
    sdft_settings: OnlineSDFTSettings = DEFAULT_SDFT_SETTINGS,
    reinforce_settings: REINFORCESettings = DEFAULT_REINFORCE_SETTINGS,
    rft_settings: RFTSettings = DEFAULT_RFT_SETTINGS,
) -> None:
    """Run all methods and write raw plus compact experiment artifacts."""
    output_dir = output_dir or OUT
    figure_dir = figure_dir or FIG
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading shared Liquid student/teacher {model_id}", flush=True)

    rollouts_path = output_dir / "rollouts.jsonl"
    curves_path = output_dir / "learning_curves.csv"
    metrics_path = output_dir / "per_seed_metrics.csv"
    curve_fields = [
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
    ]
    metrics = []
    config = None

    with (
        rollouts_path.open("w") as rollout_file,
        curves_path.open("w", newline="") as curve_file,
    ):
        curve_writer = csv.DictWriter(
            curve_file,
            fieldnames=curve_fields,
            lineterminator="\n",
        )
        curve_writer.writeheader()
        for seed_index, seed in enumerate(
            range(seed_start, seed_start + seeds),
            start=1,
        ):
            policy = LiquidLLMPolicy(
                model_id=model_id,
                device=device,
                local_files_only=local_files_only,
                prompt_style=prompt_style,
                sdft_settings=sdft_settings,
            )
            if config is None:
                config = experiment_config(
                    seeds,
                    seed_start,
                    model_id,
                    policy,
                    prompt_style,
                    icl_examples,
                    rag_examples,
                    rag_text_weight,
                    sdft_settings,
                    environment,
                    reinforce_settings,
                    rft_settings,
                )
                print(
                    f"device={policy.device} "
                    "frozen_base_lfm=true "
                    f"online_sdft_lora_trainable="
                    f"{config['online_sdft_trainable_parameters']:,}",
                    flush=True,
                )
            stream = environment.make_stream(seed)
            for method in METHODS:
                metrics.append(
                    run_method(
                        seed,
                        method,
                        stream,
                        policy,
                        rollout_file,
                        curve_writer,
                        environment,
                        icl_examples,
                        rag_examples,
                        rag_text_weight,
                        sdft_settings,
                        reinforce_settings,
                        rft_settings,
                    )
                )
                print(
                    f"seed {seed_index}/{seeds} "
                    f"(id={seed}) · {method}",
                    flush=True,
                )
                # MPS can stall on long multi-method runs; release between arms.
                gc.collect()
                if str(policy.device) == "mps":
                    import torch

                    torch.mps.synchronize()
                    torch.mps.empty_cache()

            del policy
            gc.collect()
            if config["device"] == "mps":
                import torch

                torch.mps.synchronize()
                torch.mps.empty_cache()

    if not metrics or config is None:
        raise ValueError("seeds must be positive")

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=list(metrics[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(metrics)

    with rollouts_path.open() as rollout_file:
        rollouts = [
            json.loads(line)
            for line in rollout_file
        ]
    with curves_path.open() as curve_file:
        curves = list(csv.DictReader(curve_file))

    summary = write_compact_results(
        output_dir,
        config,
        metrics,
        rollouts,
    )
    write_figures(summary, curves, figure_dir)
    print(f"wrote experiment artifacts to {output_dir}")
