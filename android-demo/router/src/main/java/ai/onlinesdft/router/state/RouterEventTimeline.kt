package ai.onlinesdft.router.state

import ai.onlinesdft.router.model.DecisionSnapshot

/**
 * Projects the insertion-ordered runtime map into the top-to-bottom timeline
 * shown by the Router page. Updating an existing key keeps its position while
 * each newly routed notification is appended at the bottom.
 */
internal fun <T> chronologicalEventValues(events: LinkedHashMap<String, T>): List<T> =
    events.values.toList()

/**
 * Freezes the distribution produced immediately after one notification's own
 * training update. Older cards keep their existing snapshot when later
 * notifications train the head, so the timeline remains an iteration-by-
 * iteration history instead of every row drifting to the latest weights.
 */
internal fun capturedEventDistribution(
    events: LinkedHashMap<String, RoutedEventUi>,
    eventId: String,
    updateIndex: Long,
    replaceExisting: Boolean = false,
    reevaluate: (DecisionSnapshot) -> FloatArray?,
): LinkedHashMap<String, RoutedEventUi> {
    val projected = LinkedHashMap(events)
    val event = projected[eventId] ?: return projected
    if (event.postTrainingProbabilities != null && !replaceExisting) return projected
    // UI measurement must never prevent the already-committed learner update
    // or its telemetry from being published.
    val current = runCatching { reevaluate(event.decision) }.getOrNull()
        ?: return projected
    projected[eventId] = event.copy(
        postTrainingProbabilities = current.copyOf(),
        distributionUpdateIndex = updateIndex,
    )
    return projected
}
