"""Notification-routing environment, outcomes, and mobile evidence views.

The environment owns the causal world: stream generation, action-dependent
feedback, and evaluator-only utilities. It exposes two disjoint projections of
:class:`Event` so that neither the deployed methods nor the language-model
teacher can read state they are not entitled to.

:class:`StudentObservation` is the serving view. Only after the executed route
produces a factual app/OS callback does the environment assemble
:class:`TeacherObservation`. Neither
view ever contains :meth:`NotificationRoutingEnvironment.oracle_utilities`,
which exists only to score the benchmark.

The soft teacher distribution is *not* computed here. The same Liquid LFM used
as the student reads a :class:`TeacherObservation` in
:meth:`online_sdft.methods.LiquidLLMPolicy.teacher_probs`. The evidence
boundary is defined in :mod:`online_sdft.privilege`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import (
    ACTIONS,
    CATEGORIES,
    DIGEST_DELIVERY_DELAY_MINUTES,
    FEEDBACK_WINDOWS_MINUTES,
    OBSERVED_OUTCOME_REWARDS,
    PHASE_LENGTH,
    PREFERENCE_SAMPLING_TEMPERATURE,
    REGIMES,
)
from .privilege import (
    FactualCallback,
    narrative_mobile_teacher_evidence,
)


def probability_power_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Temperature-scale a probability vector in stable log space.

    This computes ``q_i = p_i**(1 / temperature) / Z`` without taking the
    power directly. Exact zeros retain zero mass, equal inputs retain equal
    mass, and the input argmax is therefore unchanged.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("probabilities must have positive finite mass")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")

    normalized = values / total
    positive = normalized > 0.0
    log_probabilities = np.log(normalized[positive]) / temperature
    log_probabilities -= np.max(log_probabilities)
    scaled = np.zeros_like(normalized)
    with np.errstate(over="ignore", under="ignore"):
        scaled[positive] = np.exp(log_probabilities)
    return scaled / scaled.sum()


def one_hot(index: int, size: int) -> np.ndarray:
    values = np.zeros(size)
    values[index] = 1.0
    return values


@dataclass(frozen=True)
class NotificationScenario:
    """One auditable content pattern and its causal latent-state centers."""

    scenario_id: str
    tier: str
    title: str
    body: str
    importance_mean: float
    deadline_mean: float
    affinity_mean: float
    useful_horizon_minutes: int | None = None
    uses_relative_minutes: bool = False

    def __post_init__(self) -> None:
        if self.tier not in {"low", "routine", "urgent"}:
            raise ValueError(f"unknown scenario tier: {self.tier}")
        if not all(
            0.0 < value < 1.0
            for value in (
                self.importance_mean,
                self.deadline_mean,
                self.affinity_mean,
            )
        ):
            raise ValueError("scenario probability centers must lie in (0, 1)")
        if self.useful_horizon_minutes is not None and (
            self.useful_horizon_minutes <= 0
        ):
            raise ValueError("fixed useful horizon must be positive")
        if self.uses_relative_minutes and self.useful_horizon_minutes is not None:
            raise ValueError("relative and fixed useful horizons are exclusive")


PEOPLE = (
    "Maya", "Jordan", "Priya", "Luis", "Avery", "Sam", "Nora", "Eli",
    "Mei", "Omar", "Sofia", "Theo", "Rina", "Dev", "Kai", "Leah",
)
PROJECTS = (
    "Atlas", "Beacon", "Checkout", "Mobile", "Search", "Payments",
    "Identity", "Analytics", "Growth", "Notifications", "Billing",
    "Recommendations", "Messaging", "Accounts", "Reporting", "Platform",
)
MERCHANTS = (
    "Northstar Market", "Harbor Shop", "Cedar & Co.", "Juniper Goods",
    "Maple Market", "Summit Store", "Willow Supply", "Bluebird Outfitters",
    "Pine & Main", "Riverbend Market", "Oak Street Goods", "Lumen Shop",
    "Fieldstone Supply", "Redwood Outfitters", "Seaside Market", "Elm & Co.",
)
PRODUCTS = (
    "wireless earbuds", "trail shoes", "coffee grinder", "desk lamp",
    "travel backpack", "running watch", "camping stove", "standing mat",
    "water bottle", "mechanical keyboard", "yoga mat", "carry-on suitcase",
    "reading light", "bike helmet", "portable speaker", "wool blanket",
)
SOCIAL_TOPICS = (
    "hiking", "photography", "cooking", "cycling", "book club",
    "local event", "gardening", "travel", "running", "art", "music",
    "nature", "film", "volunteering", "design", "science",
)
PROMO_DEPARTMENTS = (
    "home", "outdoor", "electronics", "travel", "kitchen", "fitness",
    "office", "wellness", "audio", "pets", "beauty", "books", "sports",
    "automotive", "toys", "grocery",
)

SCENARIO_SALIENCE = {"low": 0.0, "routine": 0.5, "urgent": 1.0}
RELATIVE_DEADLINE_ADJUSTMENTS = {10: 0.08, 15: 0.04, 20: 0.0, 30: -0.08}
QUIET_HOURS_CATEGORIES = frozenset(
    {"manager", "calendar", "monitoring", "teammate", "commerce"}
)

# Each category contains a low-salience, routine, and time-sensitive scenario.
# Absolute scenario centers avoid the old category-prior pathology in which an
# optional meeting next week had a higher latent deadline than a call starting
# in minutes. The language still never names a route or gold label.
NOTIFICATION_SCENARIOS = {
    "manager": (
        NotificationScenario(
            "manager_recap", "low",
            "{name} shared a {project} status recap",
            "For your records. No response is needed.",
            0.25, 0.10, 0.45,
        ),
        NotificationScenario(
            "manager_review_tomorrow", "routine",
            "{project} plan needs your review",
            "{name}: Please leave comments by tomorrow afternoon.",
            0.65, 0.35, 0.55, 24 * 60,
        ),
        NotificationScenario(
            "manager_release_decision", "urgent",
            "{name} needs a decision on {project}",
            "Please approve or reject the release exception within {minutes} minutes.",
            0.95, 0.93, 0.65, uses_relative_minutes=True,
        ),
    ),
    "calendar": (
        NotificationScenario(
            "calendar_optional_next_week", "low",
            "Optional {project} office hours next week",
            "{name} invited you to a non-required session on Friday at 3:00 PM.",
            0.25, 0.10, 0.35, 7 * 24 * 60,
        ),
        NotificationScenario(
            "calendar_sync_tomorrow", "routine",
            "{project} project sync tomorrow",
            "A 30-minute meeting with {name} is scheduled for 2:00 PM.",
            0.60, 0.35, 0.45, 24 * 60,
        ),
        NotificationScenario(
            "calendar_starting_soon", "urgent",
            "{project} review starts in {minutes} minutes",
            "{name} asked you to join on time. Tap to open the video call.",
            0.95, 0.95, 0.55, uses_relative_minutes=True,
        ),
    ),
    "monitoring": (
        NotificationScenario(
            "monitoring_normal_report", "low",
            "{project} health summary is ready",
            "The latest production checks completed within the normal range.",
            0.20, 0.05, 0.15,
        ),
        NotificationScenario(
            "monitoring_latency_warning", "routine",
            "{project} latency warning",
            "P95 latency has been elevated for 15 minutes with no confirmed "
            "user impact.",
            0.62, 0.40, 0.25,
        ),
        NotificationScenario(
            "monitoring_critical_errors", "urgent",
            "Critical: {project} errors above threshold",
            "Production failures reached 8%. An acknowledgement is "
            "requested within {minutes} minutes.",
            0.95, 0.95, 0.35, uses_relative_minutes=True,
        ),
    ),
    "teammate": (
        NotificationScenario(
            "teammate_lunch_photos", "low",
            "{name} shared photos from the team lunch",
            "New photos were added to the social channel.",
            0.18, 0.05, 0.48,
        ),
        NotificationScenario(
            "teammate_dashboard_question", "routine",
            "{name} asked about the {project} dashboard",
            "When you have a moment, could you check the updated labels?",
            0.40, 0.20, 0.58,
        ),
        NotificationScenario(
            "teammate_blocked", "urgent",
            "{name} is blocked on {project}",
            "Can you confirm the rollback setting within {minutes} minutes?",
            0.82, 0.88, 0.68, uses_relative_minutes=True,
        ),
    ),
    "social": (
        NotificationScenario(
            "social_suggested_posts", "low",
            "New {topic} posts you may have missed",
            "See this week's suggested {topic} posts and community updates.",
            0.08, 0.01, 0.55,
        ),
        NotificationScenario(
            "social_weekend_message", "routine",
            "{name} sent you a message",
            "That trail looks great. Are you free sometime this weekend?",
            0.30, 0.10, 0.78,
        ),
        NotificationScenario(
            "social_live_call", "urgent",
            "{name} invited you to a live event",
            "The private group call starts in {minutes} minutes.",
            0.62, 0.75, 0.88, uses_relative_minutes=True,
        ),
    ),
    "commerce": (
        NotificationScenario(
            "commerce_receipt_available", "low",
            "Your {merchant} receipt is available",
            "A receipt for last week's purchase was added to your account.",
            0.10, 0.02, 0.28,
        ),
        NotificationScenario(
            "commerce_order_shipped", "routine",
            "Your {merchant} order has shipped",
            "The package is expected to arrive tomorrow between 1:00 and 4:00 PM.",
            0.25, 0.10, 0.38, 24 * 60,
        ),
        NotificationScenario(
            "commerce_payment_failed", "urgent",
            "Payment failed for your {merchant} order",
            "Update your payment method within 24 hours to keep the order.",
            0.70, 0.35, 0.48, 24 * 60,
        ),
    ),
    "promo": (
        NotificationScenario(
            "promo_recommendations", "low",
            "New {department} recommendations selected for you",
            "Browse this week's {department} offers whenever you have time.",
            0.03, 0.01, 0.10,
        ),
        NotificationScenario(
            "promo_member_offer", "routine",
            "{merchant} member offer: 15% off",
            "Your member offer is available for the next 12 hours.",
            0.15, 0.20, 0.18, 12 * 60,
        ),
        NotificationScenario(
            "promo_watchlist_price_drop", "urgent",
            "Price drop: {item} from your watchlist",
            "The sale price expires in {minutes} minutes while stock lasts.",
            0.50, 0.70, 0.60, uses_relative_minutes=True,
        ),
    ),
}


@dataclass
class Event:
    """Full simulator event, including evaluator/teacher-only state."""

    event_id: str
    phase: int
    category: str
    scenario_id: str
    scenario_tier: str
    title: str
    body: str
    hour: float
    useful_horizon_minutes: int | None
    importance: float
    deadline: float
    affinity: float
    busy: float
    x: np.ndarray
    z: dict
    sampled_preference: int | None = None


@dataclass(frozen=True)
class StudentObservation:
    """The complete and only view supplied to a deployed method."""

    text: str
    features: np.ndarray


@dataclass(frozen=True)
class TeacherObservation:
    """Hindsight view containing one real trajectory and no answer key."""

    context: str
    evidence: str
    observed_user_selection: str = "UNKNOWN"


class NotificationRoutingEnvironment:
    """Causal contextual-bandit simulator for one notification stream."""

    @staticmethod
    def _sample_probability(
        rng: np.random.Generator,
        mean: float,
        concentration: float = 30.0,
    ) -> float:
        """Sample a strictly bounded logit-normal probability.

        The former clipped Gaussian created large masses at zero and one. A
        beta replacement removed explicit clipping but still returned exact
        endpoints in floating point when a shape parameter fell below one.
        Sampling finite log-odds keeps every generated value strictly inside
        ``(0, 1)``; ``concentration`` retains its variance-control role.
        """
        if concentration <= 0:
            raise ValueError("concentration must be positive")
        mean = float(np.clip(mean, 0.005, 0.995))
        log_odds = math.log(mean) - math.log1p(-mean)
        scale = 0.55 * math.sqrt(30.0 / concentration)
        sampled_log_odds = log_odds + float(rng.normal(0.0, scale))
        if sampled_log_odds >= 0.0:
            return 1.0 / (1.0 + math.exp(-sampled_log_odds))
        odds = math.exp(sampled_log_odds)
        return odds / (1.0 + odds)

    @staticmethod
    def _sample_local_hour_key(
        rng: np.random.Generator,
        phase: int,
    ) -> float:
        """Sample an ordered local-time key appropriate to the regime.

        Off-hours spans one continuous overnight window from 18:00 through
        08:00; values above 24 are wrapped only when rendered.
        """
        if phase == 0:
            return float(rng.uniform(8.0, 18.0))
        if phase == 1:
            return float(rng.uniform(0.0, 24.0))
        return float(rng.uniform(18.0, 32.0))

    @staticmethod
    def _sample_local_hour_keys(
        rng: np.random.Generator,
        phase: int,
        count: int,
    ) -> list[float]:
        """Return distinct, ordered local times with bounded random jitter."""
        if count <= 0:
            return []
        starts = (8.0, 0.0, 18.0)
        ends = (18.0, 24.0, 32.0)
        start, end = starts[phase], ends[phase]
        width = (end - start) / count
        centers = start + (np.arange(count, dtype=float) + 0.5) * width
        jitter = rng.uniform(-0.20 * width, 0.20 * width, size=count)
        return (centers + jitter).tolist()

    @staticmethod
    def _deadline_mean(
        scenario: NotificationScenario,
        rendered_minutes: int,
    ) -> float:
        """Tie a stated relative deadline monotonically to latent urgency."""
        adjustment = (
            RELATIVE_DEADLINE_ADJUSTMENTS[rendered_minutes]
            if scenario.uses_relative_minutes
            else 0.0
        )
        return float(np.clip(scenario.deadline_mean + adjustment, 0.005, 0.995))

    def make_event(
        self,
        rng: np.random.Generator,
        phase: int,
        index: int,
        prefix: str,
        *,
        content_rng: np.random.Generator | None = None,
        entity_affinity: dict[tuple[str, str], float] | None = None,
        used_texts: set[tuple[str, str]] | None = None,
        hour: float | None = None,
        busy: float | None = None,
        preference_rng: np.random.Generator | None = None,
    ) -> Event:
        # Rotate the phase remainder so category counts differ by at most one
        # across the full stream (80 is not divisible by seven).
        category_offset = phase * (PHASE_LENGTH % len(CATEGORIES))
        category = CATEGORIES[(index + category_offset) % len(CATEGORIES)]
        occurrence = index // len(CATEGORIES)
        scenario = NOTIFICATION_SCENARIOS[category][(occurrence + phase) % 3]
        if content_rng is None:
            content_rng = rng
        for _ in range(256):
            content_fields = {
                "name": str(content_rng.choice(PEOPLE)),
                "project": str(content_rng.choice(PROJECTS)),
                "merchant": str(content_rng.choice(MERCHANTS)),
                "item": str(content_rng.choice(PRODUCTS)),
                "topic": str(content_rng.choice(SOCIAL_TOPICS)),
                "department": str(content_rng.choice(PROMO_DEPARTMENTS)),
                "minutes": (10, 15, 20, 30)[(occurrence + phase) % 4],
            }
            title = scenario.title.format(**content_fields)
            body = scenario.body.format(**content_fields)
            if used_texts is None or (title, body) not in used_texts:
                break
        else:  # pragma: no cover - catalog capacity is guarded by tests
            raise RuntimeError("could not render a unique notification")
        if used_texts is not None:
            used_texts.add((title, body))
        rendered_minutes = int(content_fields["minutes"])
        useful_horizon_minutes = (
            rendered_minutes
            if scenario.uses_relative_minutes
            else scenario.useful_horizon_minutes
        )
        if hour is None:
            hour = self._sample_local_hour_key(rng, phase) % 24.0
        importance = self._sample_probability(rng, scenario.importance_mean)
        deadline = self._sample_probability(
            rng,
            self._deadline_mean(scenario, rendered_minutes),
        )
        template = f"{scenario.title} {scenario.body}"
        affinity_offsets = entity_affinity or {}
        referenced_entities = [
            (field, value)
            for field, value in content_fields.items()
            if field != "minutes"
            and f"{{{field}}}" in template
            and (field, value) in affinity_offsets
        ]
        affinity_offset = (
            float(
                np.mean(
                    [affinity_offsets[entity] for entity in referenced_entities]
                )
            )
            if referenced_entities
            else 0.0
        )
        affinity = self._sample_probability(
            rng,
            float(np.clip(scenario.affinity_mean + affinity_offset, 0.02, 0.98)),
        )

        if busy is None:
            busy = self._sample_probability(
                rng,
                (0.68, 0.36, 0.18)[phase],
                concentration=20.0,
            )

        # Context bonuses scale with actual scenario salience. A normal health
        # report is not an incident merely because the user is on call, and
        # generic suggested posts are not a close social interruption merely
        # because they arrive off-hours.
        salience = SCENARIO_SALIENCE[scenario.tier]
        incident_on_call = float(
            phase == 1 and category == "monitoring"
        ) * salience
        leisure_social = float(phase == 2 and category == "social") * salience
        manager_focus = float(phase == 0 and category == "manager") * salience
        off_hours_quiet = float(
            phase == 2 and category in QUIET_HOURS_CATEGORIES
        )

        category_features = one_hot(CATEGORIES.index(category), len(CATEGORIES))
        # Numeric runtime features. Importance can come from app/OS ranking
        # metadata. Exact deadline and affinity remain simulator/evaluator-only
        # utility terms, while title/body provide coarse semantic evidence.
        features = np.concatenate(
            [
                category_features,
                np.array(
                    [
                        importance,
                        math.sin(2 * math.pi * hour / 24),
                        math.cos(2 * math.pi * hour / 24),
                        phase / 2.0,
                        1.0,
                    ]
                ),
            ]
        )
        privileged = {
            "busy": busy,
            "incident_on_call": incident_on_call,
            "leisure_social": leisure_social,
            "manager_focus": manager_focus,
            "off_hours_quiet": off_hours_quiet,
        }
        event = Event(
            event_id=f"{prefix}-{index:04d}",
            phase=phase,
            category=category,
            scenario_id=scenario.scenario_id,
            scenario_tier=scenario.tier,
            title=title,
            body=body,
            hour=hour,
            useful_horizon_minutes=useful_horizon_minutes,
            importance=importance,
            deadline=deadline,
            affinity=affinity,
            busy=busy,
            x=features,
            z=privileged,
        )
        if preference_rng is None:
            preference_rng = rng
        event.sampled_preference = int(
            preference_rng.choice(
                len(ACTIONS),
                p=self.gold_action_distribution(event),
            )
        )
        return event

    def make_stream(self, seed: int) -> list[Event]:
        (
            dynamics_seed,
            content_seed,
            profile_seed,
            order_seed,
            preference_seed,
        ) = (
            np.random.SeedSequence(seed).spawn(5)
        )
        rng = np.random.default_rng(dynamics_seed)
        content_rng = np.random.default_rng(content_seed)
        profile_rng = np.random.default_rng(profile_seed)
        order_rng = np.random.default_rng(order_seed)
        preference_rng = np.random.default_rng(preference_seed)
        entity_pools = {
            "name": PEOPLE,
            "project": PROJECTS,
            "merchant": MERCHANTS,
            "item": PRODUCTS,
            "topic": SOCIAL_TOPICS,
            "department": PROMO_DEPARTMENTS,
        }
        entity_affinity = {
            (field, entity): float(profile_rng.normal(0.0, 0.09))
            for field, pool in entity_pools.items()
            for entity in pool
        }
        used_texts: set[tuple[str, str]] = set()
        events = []
        for phase in range(3):
            indices = np.arange(PHASE_LENGTH)
            order_rng.shuffle(indices)
            hour_keys = self._sample_local_hour_keys(
                rng,
                phase,
                PHASE_LENGTH,
            )
            busy_mean = (0.68, 0.36, 0.18)[phase]
            busy_state = self._sample_probability(
                rng,
                busy_mean,
                concentration=20.0,
            )
            phase_events = []
            for position, index in enumerate(indices.tolist()):
                # Slowly varying interruptibility creates learnable local
                # continuity instead of an iid hidden label-flipping variable.
                busy_draw = self._sample_probability(
                    rng,
                    busy_mean,
                    concentration=20.0,
                )
                busy_state = 0.82 * busy_state + 0.18 * busy_draw
                phase_events.append(
                    self.make_event(
                        rng,
                        phase,
                        index,
                        f"s{seed}-p{phase}",
                        content_rng=content_rng,
                        entity_affinity=entity_affinity,
                        used_texts=used_texts,
                        hour=hour_keys[position] % 24.0,
                        busy=busy_state,
                        preference_rng=preference_rng,
                    )
                )
            events.extend(phase_events)
        return events

    def stream_fingerprint(self, seeds: Iterable[int]) -> str:
        """Hash model inputs, latent state, utilities, and routes."""
        digest = hashlib.sha256()
        for seed in seeds:
            for event in self.make_stream(int(seed)):
                observation = self.student_observation(event)
                record = {
                    "seed": int(seed),
                    "event_id": event.event_id,
                    "phase": event.phase,
                    "category": event.category,
                    "scenario_id": event.scenario_id,
                    "scenario_tier": event.scenario_tier,
                    "title": event.title,
                    "body": event.body,
                    "hour": event.hour.hex(),
                    "useful_horizon_minutes": event.useful_horizon_minutes,
                    "importance": event.importance.hex(),
                    "deadline": event.deadline.hex(),
                    "affinity": event.affinity.hex(),
                    "busy": event.busy.hex(),
                    "z": {
                        key: float(value).hex()
                        for key, value in sorted(event.z.items())
                    },
                    "student_text": observation.text,
                    "student_features": [
                        float(value).hex() for value in observation.features
                    ],
                    "oracle_utilities": [
                        float(value).hex()
                        for value in self.oracle_utilities(event)
                    ],
                    "gold_action_distribution": [
                        float(value).hex()
                        for value in self.gold_action_distribution(event)
                    ],
                    "gold_action": self.gold_action(event),
                }
                digest.update(
                    json.dumps(
                        record,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def student_observation(event: Event) -> StudentObservation:
        """Project a full event onto student-visible information only.

        Title/body and local importance are available to all compared methods.
        Exact deadline and affinity values remain evaluator-only.
        """
        # Floor instead of round so a valid 23:59.x sample cannot render as an
        # unintended 00:00 wrap at the end of the on-call phase.
        total_minutes = int(math.floor(event.hour * 60)) % (24 * 60)
        hour, minute = divmod(total_minutes, 60)
        local_time = f"{hour:02d}:{minute:02d}"
        text = (
            f"The notification title is {event.title}. "
            f"The message says {event.body} "
            f"This is a {event.category} notification that arrived at "
            f"{local_time} local time during the {REGIMES[event.phase]} "
            "period. Its on-device importance score is "
            f"{event.importance:.2f} out of 1."
        )
        return StudentObservation(text=text, features=event.x.copy())

    @staticmethod
    def oracle_utilities(event: Event) -> np.ndarray:
        """Return evaluator-only utility; never a method training target."""
        z = event.z
        urgency = event.importance * event.deadline
        # Imminent valuable events remain interruptible even when the user is
        # busy; routine interruptions continue to pay the full disruption cost.
        busy_cost = 1.20 * z["busy"] * (1.0 - 0.65 * urgency)
        interrupt = (
            1.45 * urgency
            + 0.42 * event.affinity
            - busy_cost
            + 1.00 * z["incident_on_call"]
            + 0.60 * z["manager_focus"]
            + 0.50 * z["leisure_social"]
            - 0.65 * z["off_hours_quiet"] * (1.0 - urgency)
        )
        later = (
            0.72 * event.importance
            + 0.58 * event.affinity
            - 0.62 * urgency
            + 0.22 * z["busy"]
            - 0.62 * z["incident_on_call"]
        )
        archive = (
            0.72 * (1 - event.importance)
            + 0.36 * (1 - event.affinity)
            - 0.80 * urgency
            - 0.50 * z["leisure_social"]
        )
        if (
            event.useful_horizon_minutes is not None
            and event.useful_horizon_minutes < DIGEST_DELIVERY_DELAY_MINUTES
        ):
            missed_fraction = 1.0 - (
                event.useful_horizon_minutes / DIGEST_DELIVERY_DELAY_MINUTES
            )
            # Once the digest arrives after the content's useful horizon, it
            # is stale and adds clutter relative to archiving. This is a route
            # feasibility constraint inside the causal utility—not a post-hoc
            # gold-label override. Valuable imminent content can still favor
            # INTERRUPT; low-value imminent content can favor ARCHIVE.
            stale_digest = archive - missed_fraction * (0.10 + 0.25 * urgency)
            later = min(later, stale_digest)
        return np.array([interrupt, later, archive])

    def gold_action_distribution(self, event: Event) -> np.ndarray:
        """Sharpen proportional evaluator-utility choice probabilities.

        A nonnegative vector such as ``(20, 30, 50)`` maps exactly to
        ``(0.2, 0.3, 0.5)`` before probability-power temperature scaling. The
        current utility model can produce negative values, which are not valid
        sampling weights, so only those vectors are shifted by their minimum
        before normalization. An all-equal vector falls back to the uniform
        distribution. Temperature scaling is performed in log space so the
        configured sharp temperature remains numerically stable.
        """
        weights = np.asarray(self.oracle_utilities(event), dtype=float).copy()
        if weights.shape != (len(ACTIONS),) or not np.isfinite(weights).all():
            raise ValueError(
                "oracle utilities must be one finite value per action"
            )
        minimum = float(weights.min())
        if minimum < 0.0:
            weights -= minimum
        total = float(weights.sum())
        if total <= 0.0:
            return np.full(len(ACTIONS), 1.0 / len(ACTIONS))
        proportional = weights / total
        return probability_power_temperature(
            proportional,
            PREFERENCE_SAMPLING_TEMPERATURE,
        )

    def gold_action(self, event: Event) -> int:
        """Return the sampled hidden preference used by simulator/evaluator.

        The preference is drawn once from :meth:`gold_action_distribution`
        when the event is created. Keeping that draw on the immutable stream
        makes repeated calls and paired benchmark arms agree without exposing
        the preference to feedback, prompts, or updates.
        """
        if event.sampled_preference is None:
            raise ValueError("event preference was not sampled")
        return int(event.sampled_preference)

    def execute(
        self,
        event: Event,
        action: int,
        rng: np.random.Generator,
    ) -> dict:
        """Materialize only the user event observable under ``action``.

        This is a deterministic causal observation matrix conditioned on the
        event's stored preference draw. The returned record contains only what
        the executed delivery surface could reveal:

        * INTERRUPT can reveal immediate open, delayed read, or deletion;
        * LATER can reveal digest read or deletion. A digest read is observed
          as LATER even if the hidden evaluator preference was INTERRUPT; and
        * ARCHIVE exposes no notification interaction and is always UNKNOWN.

        ``rng`` stays in the public signature for injected environments, but
        the default mapping is deterministic so identical event/action pairs
        produce identical feedback for every benchmark arm.
        """
        del rng
        gold = self.gold_action(event)
        if action == 0:
            channel = "push_notification"
            if gold == 0:
                outcome = "OPENED_IMMEDIATELY"
                observed_selection = "INTERRUPT"
                match_status = "MATCH"
                delay = FEEDBACK_WINDOWS_MINUTES["INTERRUPT_IMMEDIATE"]
            elif gold == 1:
                outcome = "OPENED_AFTER_DELAY"
                observed_selection = "LATER"
                match_status = "MISS"
                delay = FEEDBACK_WINDOWS_MINUTES["INTERRUPT_DELAYED_READ"]
            else:
                outcome = "DELETED_NOTIFICATION"
                observed_selection = "ARCHIVE"
                match_status = "MISS"
                delay = FEEDBACK_WINDOWS_MINUTES["INTERRUPT_DISMISSAL"]
        elif action == 1:
            channel = "digest_inbox"
            delay = FEEDBACK_WINDOWS_MINUTES["LATER"]
            if gold == 0:
                # The missed immediate-open preference stays hidden, but the
                # later digest read is itself an observable LATER selection.
                outcome = "OPENED_DIGEST"
                observed_selection = "LATER"
                match_status = "MATCH"
            elif gold == 1:
                outcome = "OPENED_DIGEST"
                observed_selection = "LATER"
                match_status = "MATCH"
            else:
                outcome = "DELETED_FROM_DIGEST"
                observed_selection = "ARCHIVE"
                match_status = "MISS"
        else:
            channel = "notification_not_delivered"
            outcome = "NO_OBSERVABLE_SELECTION"
            observed_selection = "UNKNOWN"
            match_status = "UNKNOWN"
            delay = FEEDBACK_WINDOWS_MINUTES["ARCHIVE"]

        return {
            "action_taken": ACTIONS[action],
            "channel": channel,
            "outcome": outcome,
            "observed_user_selection": observed_selection,
            "match_status": match_status,
            "delay_minutes": delay,
            # Reward the observable delivery outcome, not whether the route
            # name happens to equal the surface-constrained selection. In
            # particular, OPENED_DIGEST receives only partial engagement
            # credit because it cannot distinguish a preferred delay from a
            # missed immediate need.
            "reward": OBSERVED_OUTCOME_REWARDS[outcome],
        }

    @staticmethod
    def teacher_observation(
        observation: StudentObservation,
        action: int,
        callback: FactualCallback,
    ) -> TeacherObservation:
        """Build a hindsight view only after factual feedback is available."""
        if callback.action_taken != ACTIONS[action]:
            raise ValueError("feedback must belong to the executed route")
        return TeacherObservation(
            context=observation.text,
            evidence=narrative_mobile_teacher_evidence(callback),
            observed_user_selection=callback.observed_user_selection,
        )


DEFAULT_ENVIRONMENT = NotificationRoutingEnvironment()


# Compatibility functions keep the public API small while tests and notebooks
# can dependency-inject NotificationRoutingEnvironment directly.
def make_event(
    rng: np.random.Generator,
    phase: int,
    index: int,
    prefix: str,
) -> Event:
    return DEFAULT_ENVIRONMENT.make_event(rng, phase, index, prefix)


def make_stream(seed: int) -> list[Event]:
    return DEFAULT_ENVIRONMENT.make_stream(seed)


def stream_fingerprint(seeds: Iterable[int]) -> str:
    return DEFAULT_ENVIRONMENT.stream_fingerprint(seeds)


def student_observation(event: Event) -> StudentObservation:
    return DEFAULT_ENVIRONMENT.student_observation(event)


def teacher_observation(
    observation: StudentObservation,
    action: int,
    callback: FactualCallback,
) -> TeacherObservation:
    """Project a sealed serving view and narrow callback into hindsight."""
    return DEFAULT_ENVIRONMENT.teacher_observation(
        observation,
        action,
        callback,
    )


def context_text(event: Event) -> str:
    return student_observation(event).text


def oracle_utilities(event: Event) -> np.ndarray:
    return DEFAULT_ENVIRONMENT.oracle_utilities(event)


def gold_action(event: Event) -> int:
    return DEFAULT_ENVIRONMENT.gold_action(event)


def factual_feedback(
    event: Event,
    action: int,
    rng: np.random.Generator,
) -> dict:
    return DEFAULT_ENVIRONMENT.execute(event, action, rng)
