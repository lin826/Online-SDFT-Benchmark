package ai.onlinesdft.router.ui

import ai.onlinesdft.router.model.Route
import ai.onlinesdft.router.state.RoutedEventUi
import java.util.Locale

/** Header, assistant summary, setup, and the section heading. */
internal const val ROUTER_ITEMS_BEFORE_EVENT_TIMELINE = 4

internal data class RouterAutoScrollTarget(
    val newestEventId: String,
    val itemIndex: Int,
)

internal data class RouterEventFocusTarget(
    val eventId: String,
    val itemIndex: Int,
)

internal data class RouterPageRequestTarget(
    val itemIndex: Int,
    val highlightedEventId: String?,
)

internal data class SavedAlertsPermissionUi(
    val value: String,
    val healthy: Boolean,
)

internal data class RouteProbabilityShiftUi(
    val route: Route,
    val originalProbability: Float,
    val currentProbability: Float,
    /** Signed percentage-point change captured after this event's own update. */
    val deltaPercentagePoints: Double?,
    val displayText: String,
)

internal fun routeProbabilityShiftRows(event: RoutedEventUi): List<RouteProbabilityShiftUi> {
    val original = event.decision.probabilities
    require(original.size == Route.entries.size)
    val reevaluated = event.postTrainingProbabilities?.also {
        require(it.size == Route.entries.size)
    }
    return Route.entries.map { route ->
        val baseline = original[route.ordinal]
        val current = reevaluated?.get(route.ordinal) ?: baseline
        val delta = reevaluated?.let { (current - baseline).toDouble() * 100.0 }
        RouteProbabilityShiftUi(
            route = route,
            originalProbability = baseline,
            currentProbability = current,
            deltaPercentagePoints = delta,
            displayText = if (reevaluated == null) {
                compactPercent(current)
            } else {
                formatProbabilityWithShift(current, baseline)
            },
        )
    }
}

internal fun formatProbabilityWithShift(current: Float, original: Float): String {
    val currentPercent = current.coerceIn(0f, 1f).toDouble() * 100.0
    val rawDelta = (current - original).toDouble() * 100.0
    val delta = if (kotlin.math.abs(rawDelta) < 0.005) 0.0 else rawDelta
    val signedDelta = when {
        delta > 0.0 -> "+${compactNumber(delta)}"
        delta < 0.0 -> compactNumber(delta)
        else -> "0"
    }
    return "${compactNumber(currentPercent)}% ($signedDelta%)"
}

private fun compactPercent(probability: Float): String =
    "${compactNumber(probability.coerceIn(0f, 1f).toDouble() * 100.0)}%"

private fun compactNumber(value: Double): String {
    val canonical = if (kotlin.math.abs(value) < 0.005) 0.0 else value
    return String.format(Locale.US, "%.2f", canonical)
        .trimEnd('0')
        .trimEnd('.')
}

/**
 * Keys auto-scroll to event identity, so updates to scores or training details
 * in an existing card do not start another scroll animation.
 */
internal fun routerAutoScrollTarget(eventIds: List<String>): RouterAutoScrollTarget? {
    val newestEventId = eventIds.lastOrNull() ?: return null
    return RouterAutoScrollTarget(
        newestEventId = newestEventId,
        itemIndex = ROUTER_ITEMS_BEFORE_EVENT_TIMELINE + eventIds.lastIndex,
    )
}

internal fun routerEventFocusTarget(
    eventIds: List<String>,
    requestedEventId: String?,
): RouterEventFocusTarget? {
    if (requestedEventId == null) return null
    val eventIndex = eventIds.indexOf(requestedEventId)
    if (eventIndex < 0) return null
    return RouterEventFocusTarget(
        eventId = requestedEventId,
        itemIndex = ROUTER_ITEMS_BEFORE_EVENT_TIMELINE + eventIndex,
    )
}

internal fun routerPageRequestTarget(
    eventIds: List<String>,
    highlightedEventId: String?,
): RouterPageRequestTarget? {
    if (highlightedEventId == null) {
        return RouterPageRequestTarget(itemIndex = 0, highlightedEventId = null)
    }
    val focusTarget = routerEventFocusTarget(eventIds, highlightedEventId) ?: return null
    return RouterPageRequestTarget(
        itemIndex = focusTarget.itemIndex,
        highlightedEventId = focusTarget.eventId,
    )
}

internal fun savedAlertsPermissionUi(enabled: Boolean): SavedAlertsPermissionUi =
    SavedAlertsPermissionUi(
        value = if (enabled) "Allowed" else "Action required",
        healthy = enabled,
    )
