package ai.onlinesdft.router.model

import ai.onlinesdft.router.lfm.toLfmStudentContext
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/** UI and safety-only projections; LoRA training consumes tokenized prompts directly. */
object FeatureEncoder {
    const val DECISION_DIM = 18
    const val FEATURE_DIM = 29
    const val ONGOING_INDEX = 23
    const val FOREGROUND_SERVICE_INDEX = 24
    const val CALL_INDEX = 25
    const val MEDIA_INDEX = 26
    const val GROUP_SUMMARY_INDEX = 27
    const val NON_CLEARABLE_INDEX = 28

    val categories = listOf(
        "manager",
        "calendar",
        "monitoring",
        "teammate",
        "social",
        "commerce",
        "promo",
    )

    fun student(context: NotificationContext): FloatArray {
        val vector = FloatArray(FEATURE_DIM)
        val categoryIndex = categories.indexOf(normalizeCategory(context.category))
        if (categoryIndex >= 0) vector[categoryIndex] = 1f
        vector[7] = context.importance.coerceIn(0f, 1f)
        val angle = 2.0 * PI * context.hourOfDay.coerceIn(0f, 24f) / 24.0
        vector[8] = sin(angle).toFloat()
        vector[9] = cos(angle).toFloat()
        vector[10] = context.regime.ordinal / 2f
        vector[11] = 1f
        vector[ONGOING_INDEX] = context.isOngoing.asFeature()
        vector[FOREGROUND_SERVICE_INDEX] = context.isForegroundService.asFeature()
        vector[CALL_INDEX] = context.isCall.asFeature()
        vector[MEDIA_INDEX] = context.isMedia.asFeature()
        vector[GROUP_SUMMARY_INDEX] = context.isGroupSummary.asFeature()
        vector[NON_CLEARABLE_INDEX] = (!context.isClearable || context.isNoClear).asFeature()
        return vector
    }

    fun canonicalStudentContext(context: NotificationContext): String = context.toLfmStudentContext()

    /** Retained for display-only teacher vectors; the optimizer never reads this array. */
    fun teacher(sealedStudent: FloatArray, feedback: FactualFeedback): FloatArray {
        require(sealedStudent.size == FEATURE_DIM)
        val vector = sealedStudent.copyOf()
        val routeEvidence = feedback.explicitPreference ?: feedback.executedRoute
        vector[12 + routeEvidence.ordinal] = 1f
        if (feedback.outcome != Outcome.EXPLICIT_USER_CORRECTION) {
            vector[15 + feedback.outcome.ordinal] = 1f
        }
        vector[21] = feedback.delayMinutes.coerceIn(0, 240) / 240f
        vector[22] = 1f
        return vector
    }

    fun normalizeCategory(raw: String): String {
        val value = raw.trim().lowercase()
        return when {
            value in categories -> value
            "calendar" in value || "event" in value -> "calendar"
            "monitor" in value || "alert" in value || "incident" in value ||
                "security" in value || "fraud" in value -> "monitoring"
            "manager" in value || "boss" in value -> "manager"
            "chat" in value || "message" in value || "team" in value -> "teammate"
            "social" in value -> "social"
            "receipt" in value || "order" in value || "shipping" in value ||
                "delivery" in value || "package" in value || "commerce" in value -> "commerce"
            else -> "promo"
        }
    }

    private fun Boolean.asFeature(): Float = if (this) 1f else 0f
}
