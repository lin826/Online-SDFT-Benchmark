package ai.onlinesdft.router.lfm

import ai.onlinesdft.router.model.FactualFeedback
import ai.onlinesdft.router.model.FeatureEncoder
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.Outcome
import ai.onlinesdft.router.model.Regime
import java.util.Locale
import kotlin.math.floor

/**
 * The exact serving-view prose used by `NotificationRoutingEnvironment` in
 * the Python package and generated notebook. Android notification fields are
 * projected into that same student-visible schema before tokenization.
 */
fun NotificationContext.toLfmStudentContext(): String {
    // Python's serving projection floors the minute. Rounding here would turn
    // 23:59.x into midnight and would change both prompt tokens and the frozen
    // LFM prior relative to the local/notebook contract.
    val totalMinutes = floor(hourOfDay.toDouble() * 60.0).toInt().mod(24 * 60)
    val hour = totalMinutes / 60
    val minute = totalMinutes % 60
    val localTime = "%02d:%02d".format(Locale.US, hour, minute)
    val regimeText = when (regime) {
        Regime.WEEKDAY -> "weekday"
        Regime.ON_CALL -> "on-call"
        Regime.OFF_HOURS -> "off-hours"
    }
    val semanticCategory = FeatureEncoder.normalizeCategory(category)
    return "The notification title is $title. " +
        "The message says $body " +
        "This is a $semanticCategory notification that arrived at $localTime local time " +
        "during the $regimeText period. Its on-device importance score is " +
        "%.2f out of 1.".format(Locale.US, importance)
}

/** Factual phone evidence only; no reward, oracle utility, or gold route. */
fun FactualFeedback.toLfmTeacherEvidence(): String {
    val routeNarrative = when (executedRoute.name) {
        "INTERRUPT" -> "delivered the notification as an immediate interruption"
        "LATER" -> "placed the notification in a later digest"
        "ARCHIVE" -> "archived the item without delivering a notification"
        else -> error("unsupported executed route")
    }
    val outcomeNarrative = when (outcome) {
        Outcome.OPENED_IMMEDIATELY,
        Outcome.OPENED_AFTER_DELAY,
        -> "The user opened it"
        Outcome.DELETED_NOTIFICATION -> "The user deleted the immediate notification"
        Outcome.OPENED_DIGEST -> "The user opened it from the digest"
        Outcome.DELETED_FROM_DIGEST -> "The user deleted it from the digest"
        Outcome.NO_OBSERVABLE_SELECTION ->
            "No delivered notification surface revealed a user choice"
        Outcome.EXPLICIT_USER_CORRECTION ->
            "The user explicitly corrected the router"
        Outcome.TIMED_OUT_UNTOUCHED ->
            "The notification expired without a user gesture"
    }
    val timing = when {
        outcome == Outcome.NO_OBSERVABLE_SELECTION ||
            outcome == Outcome.TIMED_OUT_UNTOUCHED ->
            "$outcomeNarrative during the $delayMinutes minute observation window."
        delayMinutes == 1 -> "$outcomeNarrative one minute later."
        else -> "$outcomeNarrative $delayMinutes minutes later."
    }
    val selection = explicitPreference ?: observedSelection
    val selectionNarrative = if (selection == null) {
        "The user's preferred route remains unknown because the executed surface " +
            "revealed no selection."
    } else {
        "This behavior revealed ${selection.name} as the observed user selection " +
            "on the executed surface."
    }
    return "The router $routeNarrative. $timing $selectionNarrative"
}
