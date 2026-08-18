"""Shared experiment configuration.

This module contains names and hyperparameters only. Environment dynamics live
in :mod:`online_sdft.environment`; learning algorithms live in
:mod:`online_sdft.methods`.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "bandit"
FIG = ROOT / "figures"
DATASET_VERSION = "semantic-title-body-sharp-t001"
DATASET_NUMPY_VERSION = "2.4.6"
PUBLISHED_RESULTS_DATASET_VERSION = DATASET_VERSION
# Sample the evaluator-only user preference from a power-sharpened version of
# the proportional utility distribution: q_i is proportional to p_i**(1 / T).
PREFERENCE_SAMPLING_TEMPERATURE = 1e-2
RFT_PROTOCOL_VERSION = "teacher-categorical-k1-temperature8-singleton"
RFT_SETTINGS_PROVENANCE = (
    "fixed-temperature8-lr7e-4-same-lora-architecture"
)
RFT_CANDIDATE_COUNT = 1
RFT_SAMPLING_TEMPERATURE = 8.0
RFT_SAMPLING_MODE = "categorical"
RFT_LR = 7e-4

ACTIONS = ("INTERRUPT", "LATER", "ARCHIVE")
ACTION_CODES = ("A", "B", "C")
METHODS = (
    "Base",
    "ICL",
    "RAG",
    "REINFORCE",
    "RFT",
    "Online-SDFT",
)
CATEGORIES = (
    "manager",
    "calendar",
    "monitoring",
    "teammate",
    "social",
    "commerce",
    "promo",
)
REGIMES = ("weekday", "on-call", "off-hours")

PHASE_LENGTH = 80
STREAM_LENGTH = PHASE_LENGTH * len(REGIMES)
FEATURE_DIM = len(CATEGORIES) + 5
# Non-trainable RAG retrieval metadata: category one-hot + importance + hour
# sin/cos + regime + bias. LFM route scores come from the rendered prompt.
# Exact deadline and affinity floats stay simulator-only, although scenario
# wording intentionally provides coarse semantic evidence about salience.

# The simulator receives one notification every 15 minutes. A callback is
# released when that particular user event could have been observed. An
# immediate push open is visible quickly; a delayed read or digest interaction
# takes longer; silence after archiving remains uninformative.
DECISION_INTERVAL_MINUTES = 15
# LATER places the item in the next digest. Keeping delivery time separate from
# callback maturation makes the useful-horizon check in the evaluator explicit.
DIGEST_DELIVERY_DELAY_MINUTES = 120
FEEDBACK_WINDOWS_MINUTES = {
    "INTERRUPT_IMMEDIATE": 1,
    "INTERRUPT_DISMISSAL": 15,
    "INTERRUPT_DELAYED_READ": 120,
    "LATER": DIGEST_DELIVERY_DELAY_MINUTES,
    "ARCHIVE": 240,
}

# Reward is computed only from the factual event exposed by the executed
# delivery surface. A digest open is useful engagement evidence, but it is not
# proof that delaying was the right route: an immediately useful notification
# can also be opened later from the digest. Giving that ambiguous event the
# same +1 as an immediate open makes LATER dominate the contextual-bandit
# objective. The smaller engagement credit preserves the ordering supported by
# each observable trajectory without inventing a hidden route label:
#
#   immediate open: INTERRUPT > LATER > ARCHIVE
#   delayed/digest open: LATER > ARCHIVE > INTERRUPT
#   deletion: ARCHIVE > INTERRUPT/LATER
#
# ARCHIVE still exposes no interaction and remains neutral/censored.
OBSERVED_OUTCOME_REWARDS = {
    # Correct interruption is intentionally asymmetric: immediate utility is
    # easy to under-learn from sparse bandit feedback, while an unnecessary
    # interruption has a meaningful user cost. For INTERRUPT trajectories the
    # resulting reward vector is (+5, -1, -2).
    "OPENED_IMMEDIATELY": 5.0,
    "OPENED_AFTER_DELAY": -1.0,
    "DELETED_NOTIFICATION": -2.0,
    "OPENED_DIGEST": 0.25,
    "DELETED_FROM_DIGEST": -1.0,
    "NO_OBSERVABLE_SELECTION": 0.0,
}

EXPLORATION_EPSILON = 0.06
REPLAY_SIZE = 24
ONLINE_BATCH_SIZE = 4
ICL_K = 3
# The canonical memory baselines use three reliable demonstrations.
RAG_K = 3
# Weight assigned to visible title/body token overlap in RAG retrieval. The
# remainder is metadata similarity over category, importance, local hour, and
# regime.
RAG_TEXT_WEIGHT = 0.5
PROMPT_STYLE = "causal_demos"
# Canonical causal-demo prompts must fit this input budget. Other prompt/K
# combinations are supported only when their rendered input also fits; runtime
# validation fails loudly instead of truncating.
PROMPT_TOKEN_BUDGET = 768
TEACHER_PROMPT_VERSION = "concise-causal-v3"
PROMPT_STYLES = (
    "legacy",
    "compact",
    "causal_demos",
    "balanced_routes",
    "interrupt_narrow",
    "interrupt_context",
    "interaction_match",
    "history_guarded",
    "history_relevance",
)
INTERRUPT_PROMPT_SUFFIXES = {
    "causal_demos": (
        "Each assistant route in the history is a reliable observed user "
        "selection from a completed causal interaction. Use it only when "
        "the past notification is relevant to the current one."
    ),
    "balanced_routes": (
        "Treat all three routes as viable. Use the visible context and the "
        "closest past choices; do not default to LATER or ARCHIVE merely "
        "because they are quieter."
    ),
    "interrupt_narrow": (
        "A high-importance monitoring notification during on-call can warrant "
        "an immediate interruption. Otherwise follow visible context and close "
        "past choices."
    ),
    "interrupt_context": (
        "High-importance calendar, manager, or monitoring notifications can "
        "warrant an immediate interruption. Social, commerce, and promotion "
        "items usually should not interrupt unless close past choices support it."
    ),
    "interaction_match": (
        "A known observed selection is useful only when its notification "
        "category and regime resemble the current one. Do not copy unrelated "
        "interactions, and never turn an UNKNOWN selection into a route label."
    ),
    "history_guarded": (
        "Past interactions are weak, case-specific evidence. Never copy the "
        "latest route merely because it appears last. Use a past label only "
        "when its category and regime match the current notification; "
        "otherwise make the same decision you would make without history."
    ),
    "history_relevance": (
        "Each past interaction is tagged EXACT_MATCH or DIFFERENT_CONTEXT. "
        "Only an EXACT_MATCH label is relevant to the current notification. "
        "Ignore every DIFFERENT_CONTEXT label and never label UNKNOWN."
    ),
}

# Online-SDFT: soft targets and a small local replay window (no student ICL).
SDFT_REPLAY_SIZE = 32
SDFT_BATCH_SIZE = 8
SDFT_UPDATE_STEPS = 2
SDFT_WARMUP_EXAMPLES = 4
SDFT_DISTILL_TEMPERATURE = 1.0

MODEL_ID = "LiquidAI/LFM2.5-230M"
# Every adaptive arm resets and trains this PEFT LoRA adapter on the same
# physical Liquid model. Online-SDFT and RFT temporarily disable that adapter
# when the same model performs hindsight inference; REINFORCE has no teacher.
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.0
# LFM2.5 is a hybrid architecture: convolution blocks also expose a module
# named ``out_proj``. Use the qualified attention suffix so PEFT does not
# silently adapt those convolution projections as well.
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "self_attn.out_proj",
)
# LFM2.5-230M alternates convolution and attention blocks. These are its six
# attention-layer indices; pinning them makes the intended adapter shell
# auditable across PEFT releases and prevents hybrid-block name collisions.
LORA_LAYERS_TO_TRANSFORM = (2, 4, 6, 8, 10, 12)
SDFT_LR = 1e-3
SDFT_OPTIMIZER_WEIGHT_DECAY = 0.0
SDFT_MAX_GRAD_NORM = 1.0
# REINFORCE applies a factual-outcome policy-gradient objective to the same
# action-token LoRA adapter. These defaults were selected in-sample on the
# canonical seed-0--2 streams; there is no disjoint confirmation set.
REINFORCE_LR = 1e-4
REINFORCE_BATCH_SIZE = 8
REINFORCE_BASELINE_STEP = 0.0
REINFORCE_ENTROPY_COEF = 1.0
REINFORCE_MAX_GRAD_NORM = 1.0
# Learner-only reward shaping keyed solely by the matured callback from the
# executed delivery surface. This does not replace OBSERVED_OUTCOME_REWARDS in
# rollouts or reported cumulative observed reward. UNKNOWN selections remain
# censored before this map is consulted.
REINFORCE_TRAINING_OUTCOME_REWARDS = {
    "OPENED_IMMEDIATELY": 5.0,
    "OPENED_AFTER_DELAY": -1.0,
    "DELETED_NOTIFICATION": -5.0,
    "OPENED_DIGEST": 0.0,
    "DELETED_FROM_DIGEST": -5.0,
    "NO_OBSERVABLE_SELECTION": 0.0,
}
TEACHER_TEMPERATURE = 1.0
STUDENT_TEMPERATURE = 1.0

SYSTEM_PROMPT = """You are an on-device notification router.
Assess the partial evidence, then choose exactly one route:
A = INTERRUPT now
B = LATER in a digest
C = ARCHIVE without a notification
Use the current notification and any past completed interactions. Do not add explanation."""

TEACHER_SYSTEM_PROMPT = """Choose a route for a similar future notification:
A = INTERRUPT now
B = LATER in a digest
C = ARCHIVE silently
Use the notification and observed callback. No hidden label or unchosen outcome is available. A digest open after LATER leaves INTERRUPT versus LATER unresolved. UNKNOWN supports no route. Keep alternatives possible."""

TEACHER_REASONING_SYSTEM_PROMPT = """In one short paragraph, assess what the notification and observed callback imply for a similar future case.
The executed surface reveals only its observed behavior. Do not invent a hidden label or unchosen outcome. Explain uncertainty without choosing a route or giving a route code."""
