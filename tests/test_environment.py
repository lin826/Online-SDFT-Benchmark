"""Causal and phone-observability invariants owned by the environment."""

from collections import Counter
from inspect import signature
import re

import numpy as np
import pytest

from online_sdft.config import (
    ACTIONS,
    DATASET_NUMPY_VERSION,
    FEATURE_DIM,
    FEEDBACK_WINDOWS_MINUTES,
    PREFERENCE_SAMPLING_TEMPERATURE,
    REGIMES,
    STREAM_LENGTH,
)
from online_sdft.environment import (
    NOTIFICATION_SCENARIOS,
    PROJECTS,
    NotificationRoutingEnvironment,
    StudentObservation,
    TeacherObservation,
    factual_feedback,
    gold_action,
    make_stream,
    oracle_utilities,
    probability_power_temperature,
    student_observation,
    stream_fingerprint,
    teacher_observation,
)
from online_sdft.privilege import (
    FactualCallback,
    narrative_mobile_teacher_evidence,
    project_factual_callback,
)


def test_stream_is_one_prequential_sequence():
    stream = make_stream(0)
    assert len(stream) == STREAM_LENGTH
    assert all(event.x.shape == (FEATURE_DIM,) for event in stream)
    assert all(event.title and event.body for event in stream)
    assert len({(event.title, event.body) for event in stream}) > 40
    assert all(
        route.lower() not in f"{event.title} {event.body}".lower()
        for event in stream
        for route in ACTIONS
    )
    assert [
        event.phase
        for event in stream[:1] + stream[80:81] + stream[160:161]
    ] == [0, 1, 2]


def test_canonical_generator_runtime_is_pinned():
    assert np.__version__ == DATASET_NUMPY_VERSION


def test_notification_wording_is_realistic_and_decision_relevant():
    environment = NotificationRoutingEnvironment()
    # Resetting the RNG holds sampling noise constant. Indices 0 and 14 are
    # both manager events but select low-salience and time-sensitive scenarios.
    low = environment.make_event(np.random.default_rng(123), 0, 0, "low")
    urgent = environment.make_event(
        np.random.default_rng(123), 0, 14, "urgent"
    )
    assert low.category == urgent.category == "manager"
    assert "status recap" in low.title
    assert "needs a decision" in urgent.title
    assert "No response is needed" in low.body
    assert "within" in urgent.body
    assert urgent.importance > low.importance
    assert urgent.deadline > low.deadline
    for text in (low.title, low.body, urgent.title, urgent.body):
        assert "INTERRUPT" not in text
        assert "LATER" not in text
        assert "ARCHIVE" not in text


def test_scenario_catalog_has_explicit_semantic_contracts():
    scenario_ids = []
    for category, scenarios in NOTIFICATION_SCENARIOS.items():
        assert len(scenarios) == 3
        assert [scenario.tier for scenario in scenarios] == [
            "low",
            "routine",
            "urgent",
        ]
        for scenario in scenarios:
            scenario_ids.append(scenario.scenario_id)
            assert scenario.scenario_id.startswith(f"{category}_")
            assert 0.0 < scenario.importance_mean < 1.0
            assert 0.0 < scenario.deadline_mean < 1.0
            assert 0.0 < scenario.affinity_mean < 1.0
            has_minutes = "{minutes}" in f"{scenario.title} {scenario.body}"
            assert has_minutes == scenario.uses_relative_minutes
            if scenario.uses_relative_minutes:
                assert scenario.useful_horizon_minutes is None
    assert len(scenario_ids) == len(set(scenario_ids)) == 21


def test_stream_content_is_balanced_varied_and_seeded():
    first = make_stream(0)
    second = make_stream(1)
    category_counts = Counter(event.category for event in first)
    assert max(category_counts.values()) - min(category_counts.values()) <= 1
    assert len({(event.title, event.body) for event in first}) == STREAM_LENGTH
    assert {(event.title, event.body) for event in first} != {
        (event.title, event.body) for event in second
    }

    # Commerce notifications must not reuse internal software-project names.
    commerce = [
        event
        for event in first
        if event.category in {"commerce", "promo"}
    ]
    assert all(
        project not in f"{event.title} {event.body}"
        for event in commerce
        for project in PROJECTS
    )


def test_latent_sampling_is_bounded_without_clipping_point_masses():
    events = [event for seed in range(100) for event in make_stream(seed)]
    for field in ("importance", "deadline", "affinity", "busy"):
        values = [getattr(event, field) for event in events]
        assert all(0.0 < value < 1.0 for value in values)


def test_gold_action_is_sampled_once_from_the_sharpened_utility_distribution():
    class ProportionalEnvironment(NotificationRoutingEnvironment):
        @staticmethod
        def oracle_utilities(event):
            del event
            return np.array([20.0, 30.0, 50.0])

    environment = ProportionalEnvironment()
    captured = {}

    class PreferenceRng:
        @staticmethod
        def choice(size, p):
            captured["size"] = size
            captured["probabilities"] = np.asarray(p, dtype=float)
            return ACTIONS.index("LATER")

    event = environment.make_event(
        np.random.default_rng(123),
        0,
        0,
        "sampled",
        preference_rng=PreferenceRng(),
    )
    base = np.array([0.2, 0.3, 0.5])
    expected = probability_power_temperature(
        base,
        PREFERENCE_SAMPLING_TEMPERATURE,
    )
    assert captured["size"] == len(ACTIONS)
    np.testing.assert_allclose(captured["probabilities"], expected)
    np.testing.assert_allclose(
        environment.gold_action_distribution(event),
        expected,
    )
    assert np.isfinite(expected).all()
    assert np.all(expected > 0.0)
    assert expected.sum() == pytest.approx(1.0)
    assert gold_action(event) == ACTIONS.index("LATER")
    assert gold_action(event) == gold_action(event)


def test_probability_power_temperature_preserves_zeros_ties_and_argmax():
    probabilities = np.array([0.0, 0.4, 0.4, 0.2])
    sharpened = probability_power_temperature(
        probabilities,
        PREFERENCE_SAMPLING_TEMPERATURE,
    )

    assert PREFERENCE_SAMPLING_TEMPERATURE == 1e-2
    assert sharpened[0] == 0.0
    assert sharpened[1] == sharpened[2]
    assert sharpened[3] > 0.0
    assert np.flatnonzero(sharpened == sharpened.max()).tolist() == [1, 2]
    assert sharpened.sum() == pytest.approx(1.0)


def test_probability_power_temperature_is_stable_for_extreme_probabilities():
    probabilities = np.array([1e-300, 1e-200, 1.0])
    with np.errstate(all="raise"):
        sharpened = probability_power_temperature(
            probabilities,
            PREFERENCE_SAMPLING_TEMPERATURE,
        )

    np.testing.assert_array_equal(sharpened, np.array([0.0, 0.0, 1.0]))


def test_negative_utilities_are_shifted_before_proportional_normalization():
    class SignedUtilityEnvironment(NotificationRoutingEnvironment):
        @staticmethod
        def oracle_utilities(event):
            del event
            return np.array([-1.0, 0.0, 2.0])

    environment = SignedUtilityEnvironment()
    event = environment.make_event(np.random.default_rng(7), 0, 0, "signed")
    expected = probability_power_temperature(
        np.array([0.0, 0.25, 0.75]),
        PREFERENCE_SAMPLING_TEMPERATURE,
    )
    np.testing.assert_allclose(
        environment.gold_action_distribution(event),
        expected,
    )


def test_sampled_preferences_are_seeded_and_vary_across_streams():
    first = [gold_action(event) for event in make_stream(0)]
    repeated = [gold_action(event) for event in make_stream(0)]
    second = [gold_action(event) for event in make_stream(1)]
    assert first == repeated
    assert first != second
    assert set(first + second) == set(range(len(ACTIONS)))


def _utility_optimal_action(event):
    return int(np.argmax(oracle_utilities(event)))


def test_relative_minutes_drive_deadline_and_never_expire_in_digest():
    environment = NotificationRoutingEnvironment()
    relative = [
        scenario
        for scenarios in NOTIFICATION_SCENARIOS.values()
        for scenario in scenarios
        if scenario.uses_relative_minutes
    ]
    for scenario in relative:
        means = [
            environment._deadline_mean(scenario, minutes)
            for minutes in (10, 15, 20, 30)
        ]
        assert means == sorted(means, reverse=True)
        assert len(set(means)) == 4

    # The old soft penalty still admitted rare stale-digest wins; scan far
    # beyond the original evaluation seeds to guard the feasibility contract.
    for seed in range(200):
        for event in make_stream(seed):
            if event.useful_horizon_minutes is None:
                continue
            match = re.search(r"within (\d+) minutes|in (\d+) minutes", event.body)
            if match is None:
                match = re.search(r"in (\d+) minutes", event.title)
            if event.useful_horizon_minutes < 120:
                assert match is not None
                stated = next(int(value) for value in match.groups() if value)
                assert event.useful_horizon_minutes == stated
                assert _utility_optimal_action(event) != ACTIONS.index("LATER")


def test_context_boosts_follow_scenario_salience_not_category_alone():
    stream = make_stream(0)
    monitoring = {
        event.scenario_tier: event
        for event in stream
        if event.phase == 1 and event.category == "monitoring"
    }
    social = {
        event.scenario_tier: event
        for event in stream
        if event.phase == 2 and event.category == "social"
    }
    manager = {
        event.scenario_tier: event
        for event in stream
        if event.phase == 0 and event.category == "manager"
    }
    assert {
        tier: monitoring[tier].z["incident_on_call"]
        for tier in ("low", "routine", "urgent")
    } == {"low": 0.0, "routine": 0.5, "urgent": 1.0}
    assert {
        tier: social[tier].z["leisure_social"]
        for tier in ("low", "routine", "urgent")
    } == {"low": 0.0, "routine": 0.5, "urgent": 1.0}
    assert {
        tier: manager[tier].z["manager_focus"]
        for tier in ("low", "routine", "urgent")
    } == {"low": 0.0, "routine": 0.5, "urgent": 1.0}


def test_low_salience_utility_optima_avoid_obviously_bad_routes():
    events = [event for seed in range(25) for event in make_stream(seed)]
    assert all(
        _utility_optimal_action(event) != ACTIONS.index("INTERRUPT")
        for event in events
        if event.scenario_tier == "low"
    )
    assert all(
        _utility_optimal_action(event) != ACTIONS.index("INTERRUPT")
        for event in events
        if event.scenario_id in {
            "monitoring_normal_report",
            "social_suggested_posts",
            "manager_recap",
        }
    )
    tomorrow = [
        event
        for event in events
        if event.scenario_id in {
            "manager_review_tomorrow",
            "calendar_sync_tomorrow",
        }
    ]
    tomorrow_interrupts = sum(
        _utility_optimal_action(event) == ACTIONS.index("INTERRUPT")
        for event in tomorrow
    )
    assert tomorrow_interrupts < 0.02 * len(tomorrow)


def test_fixed_horizon_and_no_impact_rows_are_not_interrupt_optimal():
    events = [event for seed in range(200) for event in make_stream(seed)]
    payment_failures = [
        event
        for event in events
        if event.scenario_id == "commerce_payment_failed"
    ]
    assert payment_failures
    assert all(
        event.useful_horizon_minutes == 24 * 60
        for event in payment_failures
    )
    payment_interrupts = sum(
        _utility_optimal_action(event) == ACTIONS.index("INTERRUPT")
        for event in payment_failures
    )
    assert payment_interrupts < 0.01 * len(payment_failures)

    off_hours_warnings = [
        event
        for event in events
        if event.phase == 2
        and event.scenario_id == "monitoring_latency_warning"
    ]
    assert off_hours_warnings
    warning_interrupts = sum(
        _utility_optimal_action(event) == ACTIONS.index("INTERRUPT")
        for event in off_hours_warnings
    )
    assert warning_interrupts < 0.01 * len(off_hours_warnings)


def test_local_times_are_ordered_inside_each_regime():
    for seed in range(5):
        stream = make_stream(seed)
        for phase in range(3):
            hours = [event.hour for event in stream if event.phase == phase]
            unwrapped = [
                hour + 24.0 if phase == 2 and hour < 18.0 else hour
                for hour in hours
            ]
            assert unwrapped == sorted(unwrapped)
            rendered = []
            for event in (event for event in stream if event.phase == phase):
                match = re.search(
                    r"arrived at (\d{2}):(\d{2}) local time",
                    student_observation(event).text,
                )
                assert match is not None
                minute = 60 * int(match.group(1)) + int(match.group(2))
                if phase == 2 and minute < 18 * 60:
                    minute += 24 * 60
                rendered.append(minute)
            assert rendered == sorted(rendered)
            assert len(rendered) == len(set(rendered))


def test_dataset_fingerprint_locks_inputs_latents_utilities_and_sampled_routes():
    expected = "986cdf1a7d5fcc04c2b33f1bf90a1fc4f24a97ee85e663370382d8a67e4c932d"
    assert stream_fingerprint(range(3)) == expected
    assert stream_fingerprint(range(3)) != stream_fingerprint(range(3, 6))


def _event_for_gold(action: int):
    return next(event for event in make_stream(0) if gold_action(event) == action)


def test_action_specific_user_selection_matrix_never_returns_gold_label():
    expected = {
        ("INTERRUPT", "INTERRUPT"): (
            "OPENED_IMMEDIATELY", "INTERRUPT", "MATCH", 5.0,
        ),
        ("INTERRUPT", "LATER"): (
            "OPENED_AFTER_DELAY", "LATER", "MISS", -1.0,
        ),
        ("INTERRUPT", "ARCHIVE"): (
            "DELETED_NOTIFICATION", "ARCHIVE", "MISS", -2.0,
        ),
        ("LATER", "INTERRUPT"): (
            "OPENED_DIGEST", "LATER", "MATCH", 0.25,
        ),
        ("LATER", "LATER"): (
            "OPENED_DIGEST", "LATER", "MATCH", 0.25,
        ),
        ("LATER", "ARCHIVE"): (
            "DELETED_FROM_DIGEST", "ARCHIVE", "MISS", -1.0,
        ),
        ("ARCHIVE", "INTERRUPT"): (
            "NO_OBSERVABLE_SELECTION", "UNKNOWN", "UNKNOWN", 0.0,
        ),
        ("ARCHIVE", "LATER"): (
            "NO_OBSERVABLE_SELECTION", "UNKNOWN", "UNKNOWN", 0.0,
        ),
        ("ARCHIVE", "ARCHIVE"): (
            "NO_OBSERVABLE_SELECTION", "UNKNOWN", "UNKNOWN", 0.0,
        ),
    }
    for chosen_index, chosen in enumerate(ACTIONS):
        for gold_index, gold in enumerate(ACTIONS):
            feedback = factual_feedback(
                _event_for_gold(gold_index),
                chosen_index,
                np.random.default_rng(99),
            )
            observed = (
                feedback["outcome"],
                feedback["observed_user_selection"],
                feedback["match_status"],
                feedback["reward"],
            )
            assert observed == expected[(chosen, gold)]
            assert "gold" not in feedback
            assert "oracle" not in feedback


def test_digest_read_reveals_later_without_revealing_hidden_preference():
    missed_immediate = factual_feedback(
        _event_for_gold(0), 1, np.random.default_rng(0)
    )
    preferred_later = factual_feedback(
        _event_for_gold(1), 1, np.random.default_rng(0)
    )
    assert missed_immediate == preferred_later
    assert missed_immediate["observed_user_selection"] == "LATER"
    assert missed_immediate["reward"] == 0.25


def test_observable_reward_does_not_make_later_globally_dominant():
    """Balanced hidden preferences must not give one surface free +1 credit."""
    rewards = {
        chosen: [
            factual_feedback(
                _event_for_gold(gold_index),
                chosen_index,
                np.random.default_rng(0),
            )["reward"]
            for gold_index in range(len(ACTIONS))
        ]
        for chosen_index, chosen in enumerate(ACTIONS)
    }
    assert rewards["INTERRUPT"] == [5.0, -1.0, -2.0]
    assert rewards["LATER"] == [0.25, 0.25, -1.0]
    assert rewards["ARCHIVE"] == [0.0, 0.0, 0.0]
    assert np.mean(rewards["LATER"]) < 0.0


def test_feedback_timing_follows_the_observed_user_event():
    immediate = factual_feedback(
        _event_for_gold(0), 0, np.random.default_rng(0)
    )
    delayed = factual_feedback(
        _event_for_gold(1), 0, np.random.default_rng(0)
    )
    dismissed = factual_feedback(
        _event_for_gold(2), 0, np.random.default_rng(0)
    )
    digest = factual_feedback(
        _event_for_gold(1), 1, np.random.default_rng(0)
    )
    archived = factual_feedback(
        _event_for_gold(0), 2, np.random.default_rng(0)
    )
    assert immediate["delay_minutes"] == FEEDBACK_WINDOWS_MINUTES[
        "INTERRUPT_IMMEDIATE"
    ]
    assert dismissed["delay_minutes"] == FEEDBACK_WINDOWS_MINUTES[
        "INTERRUPT_DISMISSAL"
    ]
    assert delayed["delay_minutes"] == FEEDBACK_WINDOWS_MINUTES[
        "INTERRUPT_DELAYED_READ"
    ]
    assert digest["delay_minutes"] == FEEDBACK_WINDOWS_MINUTES["LATER"]
    assert archived["delay_minutes"] == FEEDBACK_WINDOWS_MINUTES["ARCHIVE"]


def test_same_event_and_action_have_identical_feedback_for_all_baselines():
    event = _event_for_gold(1)
    feedback = [
        factual_feedback(event, 0, np.random.default_rng(seed))
        for seed in range(10)
    ]
    assert all(item == feedback[0] for item in feedback)


def test_teacher_view_contains_only_context_and_executed_callback():
    event = make_stream(2)[0]
    action = ACTIONS.index("LATER")
    feedback = factual_feedback(event, action, np.random.default_rng(4))
    sealed_observation = student_observation(event)
    callback = project_factual_callback(feedback)
    observation = teacher_observation(
        sealed_observation,
        action,
        callback,
    )

    assert isinstance(observation, TeacherObservation)
    assert observation.context == sealed_observation.text
    assert "placed the notification in a later digest" in observation.evidence
    assert observation.observed_user_selection == feedback[
        "observed_user_selection"
    ]
    assert "120 minutes later" in observation.evidence
    assert (
        f"revealed {feedback['observed_user_selection']} as the observed "
        "user selection"
    ) in observation.evidence
    assert not any(
        character in observation.evidence
        for character in "();="
    )
    assert "reward=" not in observation.evidence
    assert "oracle" not in observation.evidence.lower()
    assert "counterfactual" not in observation.evidence.lower()
    for hidden in (
        "gold_action=",
        "dataset_label=",
        "busy=",
        "interruption_filter=",
        "match_status=",
        "delivery_channel=",
    ):
        assert hidden not in observation.evidence


def test_callback_projection_seals_out_reward_and_evaluator_bookkeeping():
    feedback = {
        "action_taken": "INTERRUPT",
        "channel": "push_notification",
        "outcome": "OPENED_IMMEDIATELY",
        "observed_user_selection": "INTERRUPT",
        "match_status": "MATCH",
        "delay_minutes": 1,
        "reward": 1.0,
        "gold_action": "INTERRUPT",
    }
    callback = project_factual_callback(feedback)

    assert callback == FactualCallback(
        action_taken="INTERRUPT",
        outcome="OPENED_IMMEDIATELY",
        observed_user_selection="INTERRUPT",
        delay_minutes=1,
    )
    assert set(vars(callback)) == {
        "action_taken",
        "outcome",
        "observed_user_selection",
        "delay_minutes",
    }
    assert tuple(signature(teacher_observation).parameters) == (
        "observation",
        "action",
        "callback",
    )


@pytest.mark.parametrize(
    ("callback", "expected_timing"),
    [
        (
            FactualCallback(
                "INTERRUPT",
                "OPENED_IMMEDIATELY",
                "INTERRUPT",
                1,
            ),
            "The user opened it one minute later.",
        ),
        (
            FactualCallback(
                "INTERRUPT",
                "OPENED_AFTER_DELAY",
                "LATER",
                120,
            ),
            "The user opened it 120 minutes later.",
        ),
        (
            FactualCallback(
                "ARCHIVE",
                "NO_OBSERVABLE_SELECTION",
                "UNKNOWN",
                240,
            ),
            "during the 240 minute observation window.",
        ),
    ],
)
def test_callback_narrative_uses_plain_prose_and_grammatical_timing(
    callback,
    expected_timing,
):
    evidence = narrative_mobile_teacher_evidence(callback)
    assert expected_timing in evidence
    assert "1 minutes" not in evidence
    assert not any(character in evidence for character in "();=")


def test_teacher_rejects_feedback_from_another_route():
    event = make_stream(4)[0]
    feedback = factual_feedback(event, 0, np.random.default_rng(0))
    sealed_observation = student_observation(event)
    callback = project_factual_callback(feedback)
    with pytest.raises(ValueError, match="executed route"):
        teacher_observation(sealed_observation, 2, callback)


def test_oracle_is_evaluation_only_and_student_view_is_compact():
    event = make_stream(5)[0]
    assert oracle_utilities(event).shape == (len(ACTIONS),)
    distribution = NotificationRoutingEnvironment().gold_action_distribution(
        event
    )
    assert distribution.shape == (len(ACTIONS),)
    assert distribution.sum() == pytest.approx(1.0)
    observation = student_observation(event)
    assert isinstance(observation, StudentObservation)
    assert observation.features.shape == (FEATURE_DIM,)
    assert f"The notification title is {event.title}." in observation.text
    assert f"The message says {event.body}" in observation.text
    assert f"This is a {event.category} notification" in observation.text
    assert "local time" in observation.text
    assert f"during the {REGIMES[event.phase]} period" in observation.text
    assert (
        f"on-device importance score is {event.importance:.2f} out of 1"
        in observation.text
    )
    assert len(observation.text) - len(event.title) - len(event.body) <= 200
    assert (
        len(observation.text.split())
        - len(event.title.split())
        - len(event.body.split())
        <= 35
    )
    assert "Metadata:" not in observation.text
    assert "category=" not in observation.text
    assert "hour=" not in observation.text
    assert "regime=" not in observation.text
    assert "importance=" not in observation.text
    assert "affinity" not in observation.text
    assert "urgency" not in observation.text
    assert "deadline" not in observation.text
    assert "interruption_filter" not in observation.text
    assert "z" not in signature(StudentObservation).parameters
