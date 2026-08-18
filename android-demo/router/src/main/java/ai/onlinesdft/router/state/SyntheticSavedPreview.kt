package ai.onlinesdft.router.state

import ai.onlinesdft.router.model.DecisionSnapshot
import ai.onlinesdft.router.model.Route
import ai.onlinesdft.router.notification.DigestInboxItem
import ai.onlinesdft.router.notification.DigestInboxOrigin

/**
 * Builds a clearly non-learnable Saved-page preview for a synthetic lab row.
 * No Android notification is cancelled and no assistant alert is posted.
 */
internal fun syntheticLabSavedPreview(
    decision: DecisionSnapshot,
    openToken: String,
): DigestInboxItem? {
    if (decision.chosenRoute != Route.LATER) return null
    return DigestInboxItem(
        eventId = decision.context.eventId,
        openToken = openToken,
        sourcePackage = decision.context.packageName,
        title = decision.context.title.ifBlank { "Saved notification" }.take(512),
        body = decision.context.body.take(4_096),
        routedAtMillis = decision.decidedAtMillis,
        origin = DigestInboxOrigin.SYNTHETIC_LAB,
        learningSnapshot = null,
    )
}
