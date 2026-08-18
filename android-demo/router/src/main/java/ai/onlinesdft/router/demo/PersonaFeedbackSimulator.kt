package ai.onlinesdft.router.demo

import ai.onlinesdft.router.model.DecisionSnapshot
import ai.onlinesdft.router.model.FactualFeedback
import ai.onlinesdft.router.model.FeedbackSource
import ai.onlinesdft.router.model.Outcome
import ai.onlinesdft.router.model.Route

/**
 * A sealed synthetic user for the labeled lab. It emits only callbacks that
 * Android can expose, plus an honestly labeled explicit correction when a
 * delivery surface cannot reveal the persona's desired route.
 */
object PersonaFeedbackSimulator {
    /** A visible, scripted correction in the synthetic curriculum. */
    fun scriptedCorrection(
        decision: DecisionSnapshot,
        simulatedPreference: Route,
        observedAtMillis: Long = System.currentTimeMillis(),
    ): FactualFeedback = FactualFeedback(
        eventId = decision.context.eventId,
        executedRoute = decision.chosenRoute,
        outcome = Outcome.EXPLICIT_USER_CORRECTION,
        observedSelection = simulatedPreference,
        delayMinutes = 0,
        source = FeedbackSource.EXPLICIT_USER_CORRECTION,
        explicitPreference = simulatedPreference,
        observedAtMillis = observedAtMillis,
    )

    fun observe(
        decision: DecisionSnapshot,
        simulatedPreference: Route,
        observedAtMillis: Long = System.currentTimeMillis(),
    ): FactualFeedback {
        val outcome = when (decision.chosenRoute) {
            Route.INTERRUPT -> when (simulatedPreference) {
                Route.INTERRUPT -> Outcome.OPENED_IMMEDIATELY
                Route.LATER -> Outcome.TIMED_OUT_UNTOUCHED
                Route.ARCHIVE -> Outcome.DELETED_NOTIFICATION
            }
            Route.LATER -> when (simulatedPreference) {
                Route.INTERRUPT -> Outcome.EXPLICIT_USER_CORRECTION
                Route.LATER -> Outcome.OPENED_DIGEST
                Route.ARCHIVE -> Outcome.DELETED_FROM_DIGEST
            }
            Route.ARCHIVE -> if (simulatedPreference == Route.ARCHIVE) {
                Outcome.NO_OBSERVABLE_SELECTION
            } else {
                Outcome.EXPLICIT_USER_CORRECTION
            }
        }
        val selection = when (outcome) {
            Outcome.OPENED_IMMEDIATELY -> Route.INTERRUPT
            Outcome.OPENED_AFTER_DELAY, Outcome.OPENED_DIGEST -> Route.LATER
            Outcome.DELETED_NOTIFICATION, Outcome.DELETED_FROM_DIGEST -> Route.ARCHIVE
            Outcome.NO_OBSERVABLE_SELECTION -> null
            Outcome.EXPLICIT_USER_CORRECTION -> simulatedPreference
            Outcome.TIMED_OUT_UNTOUCHED -> Route.LATER
        }
        val delay = when (outcome) {
            Outcome.OPENED_IMMEDIATELY -> 1
            Outcome.DELETED_NOTIFICATION -> 15
            Outcome.NO_OBSERVABLE_SELECTION -> 240
            Outcome.EXPLICIT_USER_CORRECTION -> 0
            Outcome.TIMED_OUT_UNTOUCHED -> 120
            else -> 120
        }
        return FactualFeedback(
            eventId = decision.context.eventId,
            executedRoute = decision.chosenRoute,
            outcome = outcome,
            observedSelection = selection,
            delayMinutes = delay,
            source = if (outcome == Outcome.EXPLICIT_USER_CORRECTION) {
                FeedbackSource.EXPLICIT_USER_CORRECTION
            } else {
                FeedbackSource.SYNTHETIC_LAB
            },
            explicitPreference = if (outcome == Outcome.EXPLICIT_USER_CORRECTION) {
                simulatedPreference
            } else {
                null
            },
            observedAtMillis = observedAtMillis,
        )
    }
}
