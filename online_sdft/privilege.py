"""Phone-observable evidence used by the hindsight teacher.

The live protocol deliberately avoids simulator latents, evaluator labels,
counterfactual outcomes, hand-written demonstrations, and a shadow policy. It
models an Android OS-integrated or user-authorized notification assistant that
can retain a small local event record:

* the notification and serving information already available to the policy;
* the route the assistant actually executed; and
* a later immediate open, delayed read, deletion, digest interaction, or
  explicit lack of an observable selection.

Simulator state never crosses this boundary. The scalar simulator reward is
intentionally not included: a phone observes events, not the benchmark's
engineered utility function.
"""

from __future__ import annotations

from dataclasses import dataclass

ROUTE_NARRATIVES = {
    "INTERRUPT": "delivered the notification as an immediate interruption",
    "LATER": "placed the notification in a later digest",
    "ARCHIVE": "archived the item without delivering a notification",
}

OUTCOME_NARRATIVES = {
    "OPENED_IMMEDIATELY": "The user opened it",
    "OPENED_AFTER_DELAY": "The user opened it",
    "DELETED_NOTIFICATION": "The user deleted the immediate notification",
    "OPENED_DIGEST": "The user opened it from the digest",
    "DELETED_FROM_DIGEST": "The user deleted it from the digest",
    "NO_OBSERVABLE_SELECTION": (
        "No delivered notification surface revealed a user choice"
    ),
}


@dataclass(frozen=True)
class FactualCallback:
    """Only callback fields permitted to cross into the teacher boundary."""

    action_taken: str
    outcome: str
    observed_user_selection: str
    delay_minutes: int


def project_factual_callback(feedback: dict) -> FactualCallback:
    """Discard reward, evaluator match, and transport bookkeeping eagerly."""
    return FactualCallback(
        action_taken=str(feedback["action_taken"]),
        outcome=str(feedback["outcome"]),
        observed_user_selection=str(feedback["observed_user_selection"]),
        delay_minutes=int(feedback["delay_minutes"]),
    )


def narrative_mobile_teacher_evidence(callback: FactualCallback) -> str:
    """Explain one factual callback as plain prose for the small teacher."""
    route = callback.action_taken
    outcome = callback.outcome
    selection = callback.observed_user_selection
    delay = callback.delay_minutes
    sentences = [
        f"The router {ROUTE_NARRATIVES[route]}.",
    ]
    if outcome == "NO_OBSERVABLE_SELECTION":
        sentences.append(
            f"{OUTCOME_NARRATIVES[outcome]} during the {delay} minute "
            "observation window."
        )
    elif delay == 1:
        sentences.append(f"{OUTCOME_NARRATIVES[outcome]} one minute later.")
    else:
        sentences.append(f"{OUTCOME_NARRATIVES[outcome]} {delay} minutes later.")
    if selection == "UNKNOWN":
        sentences.append(
            "The user's preferred route remains unknown because the "
            "executed surface revealed no selection."
        )
    else:
        sentences.append(
            f"This behavior revealed {selection} as the observed user "
            "selection on the executed surface."
        )
    return " ".join(sentences)
