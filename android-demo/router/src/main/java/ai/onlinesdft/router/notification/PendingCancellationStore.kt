package ai.onlinesdft.router.notification

import ai.onlinesdft.router.model.DecisionSnapshot
import ai.onlinesdft.router.model.ExecutionConstraint
import ai.onlinesdft.router.model.FeatureEncoder
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.Regime
import ai.onlinesdft.router.model.Route
import android.content.Context
import android.util.Base64
import android.util.Log
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Small app-private journal for the interval between cancelNotification() and
 * Android's listener-cancel confirmation. Raw title/body and sealed learner
 * inputs are retained for outstanding Later/Archive storage obligations and
 * deleted on resolution/lockdown.
 */
internal object PendingCancellationStore {
    data class Stored(
        val key: String,
        val decision: DecisionSnapshot,
        val confirmed: Boolean,
    )

    fun save(
        context: Context,
        key: String,
        decision: DecisionSnapshot,
        confirmed: Boolean = false,
    ): Boolean = runCatching {
        val notification = decision.context
        val learningSnapshot = if (
            decision.chosenRoute == Route.LATER || decision.chosenRoute == Route.ARCHIVE
        ) {
            runCatching { decision.toDigestLearningSnapshot() }.getOrNull()
        } else {
            null
        }
        val value = JSONObject()
            .put("version", VERSION)
            .put("key", key)
            .put("event_id", notification.eventId)
            .put("package_name", notification.packageName)
            .put("title", notification.title)
            .put("body", notification.body)
            .put("category", notification.category)
            .put("importance", notification.importance.toDouble())
            .put("regime", notification.regime.name)
            .put("hour_of_day", notification.hourOfDay.toDouble())
            .put("posted_at_ms", notification.postedAtMillis)
            .put("case_id", notification.caseId ?: JSONObject.NULL)
            .put("is_clearable", notification.isClearable)
            .put("is_ongoing", notification.isOngoing)
            .put("is_foreground_service", notification.isForegroundService)
            .put("is_call", notification.isCall)
            .put("is_media", notification.isMedia)
            .put("is_group_summary", notification.isGroupSummary)
            .put("is_no_clear", notification.isNoClear)
            .put("can_publish_digest", notification.canPublishDigest)
            .put("chosen_route", decision.chosenRoute.name)
            .put("decided_at_ms", decision.decidedAtMillis)
            .put("run_epoch", decision.runEpoch)
            .put("confirmed", confirmed)
        learningSnapshot?.let { snapshot ->
            value
                .put("learning_foundation_model_id", snapshot.foundationModelId)
                .put("learning_checkpoint_index", snapshot.checkpointIndex)
                .put("learning_student_prompt", snapshot.studentPrompt)
                .put(
                    "learning_foundation_probabilities",
                    encodeDoubles(snapshot.foundationProbabilitiesFp64),
                )
                .put(
                    "learning_decision_probabilities",
                    encodeDoubles(snapshot.adaptiveDecisionProbabilities),
                )
        }
        preferences(context).edit().putString(storageKey(key), value.toString()).commit()
    }.getOrElse { error ->
        Log.w(TAG, "Unable to persist pending cancellation", error)
        false
    }

    fun markConfirmed(context: Context, key: String): Boolean = runCatching {
        val preferenceKey = storageKey(key)
        val raw = preferences(context).getString(preferenceKey, null) ?: return false
        val value = JSONObject(raw).put("confirmed", true)
        preferences(context).edit().putString(preferenceKey, value.toString()).commit()
    }.getOrElse { error ->
        Log.w(TAG, "Unable to seal cancellation confirmation", error)
        false
    }

    fun remove(context: Context, key: String): Boolean =
        preferences(context).edit().remove(storageKey(key)).commit()

    fun clear(context: Context): Boolean = preferences(context).edit().clear().commit()

    fun load(context: Context): List<Stored> = preferences(context).all.values.mapNotNull { raw ->
        runCatching { decode(raw as String) }
            .onFailure { Log.w(TAG, "Ignoring malformed pending cancellation", it) }
            .getOrNull()
    }

    private fun decode(raw: String): Stored {
        val value = JSONObject(raw)
        require(value.getInt("version") == VERSION)
        val key = value.getString("key")
        val route = Route.fromWire(value.getString("chosen_route"))
            ?: error("Unknown pending route")
        require(route != Route.INTERRUPT)
        val notification = NotificationContext(
            eventId = value.getString("event_id"),
            packageName = value.getString("package_name"),
            title = value.optString("title"),
            body = value.optString("body"),
            category = value.optString("category", "restored-pending"),
            importance = value.optDouble("importance", 0.0).toFloat(),
            regime = runCatching {
                Regime.valueOf(value.optString("regime", Regime.WEEKDAY.name))
            }.getOrDefault(Regime.WEEKDAY),
            hourOfDay = value.optDouble("hour_of_day", 0.0).toFloat(),
            postedAtMillis = value.getLong("posted_at_ms"),
            caseId = value.optNullableString("case_id"),
            isClearable = value.optBoolean("is_clearable", true),
            isOngoing = value.optBoolean("is_ongoing", false),
            isForegroundService = value.optBoolean("is_foreground_service", false),
            isCall = value.optBoolean("is_call", false),
            isMedia = value.optBoolean("is_media", false),
            isGroupSummary = value.optBoolean("is_group_summary", false),
            isNoClear = value.optBoolean("is_no_clear", false),
            canPublishDigest = value.optBoolean("can_publish_digest", true),
        )
        val runEpoch = value.optLong("run_epoch", -1L)
        val restoredLearningDecision = if (
            route != Route.INTERRUPT && value.has("learning_student_prompt")
        ) {
            val snapshot = DigestLearningSnapshot(
                context = notification,
                foundationModelId = value.getString("learning_foundation_model_id"),
                decidedAtMillis = value.getLong("decided_at_ms"),
                checkpointIndex = value.getLong("learning_checkpoint_index"),
                runEpoch = runEpoch,
                studentPrompt = value.getString("learning_student_prompt"),
                foundationProbabilitiesFp64 = decodeDoubles(
                    value.getString("learning_foundation_probabilities"),
                    Route.entries.size,
                ),
                adaptiveDecisionProbabilities = decodeDoubles(
                    value.getString("learning_decision_probabilities"),
                    Route.entries.size,
                ),
            )
            DigestInboxItem(
                eventId = notification.eventId,
                openToken = "pending-cancellation",
                sourcePackage = notification.packageName,
                title = notification.title.ifBlank { "Saved notification" }.take(512),
                body = notification.body.take(4_096),
                routedAtMillis = snapshot.decidedAtMillis,
                origin = if (route == Route.ARCHIVE) {
                    DigestInboxOrigin.ROUTER_ARCHIVE
                } else {
                    DigestInboxOrigin.LIVE_NOTIFICATION
                },
                learningSnapshot = snapshot,
            ).toDecision(runEpoch)
        } else {
            null
        }
        return Stored(
            key = key,
            decision = restoredLearningDecision ?: DecisionSnapshot(
                context = notification,
                studentFeatures = FloatArray(FeatureEncoder.FEATURE_DIM),
                probabilities = oneHot(route),
                baseProbabilities = oneHot(route),
                chosenRoute = route,
                recommendedRoute = route,
                executionConstraint = ExecutionConstraint.NONE,
                baseRoute = route,
                baseRecommendedRoute = route,
                checkpointIndex = 0L,
                adapterChecksum = "restored-pending",
                decidedAtMillis = value.getLong("decided_at_ms"),
                inferenceLatencyMillis = 0.0,
                runEpoch = runEpoch,
            ),
            confirmed = value.optBoolean("confirmed", false),
        )
    }

    private fun oneHot(route: Route): FloatArray =
        FloatArray(Route.entries.size) { index -> if (index == route.ordinal) 1f else 0f }

    private fun encodeDoubles(values: DoubleArray): String {
        val buffer = ByteBuffer.allocate(Int.SIZE_BYTES + values.size * Double.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(values.size)
        values.forEach(buffer::putDouble)
        return Base64.encodeToString(buffer.array(), Base64.NO_WRAP)
    }

    private fun decodeDoubles(encoded: String, expectedSize: Int): DoubleArray {
        val bytes = Base64.decode(encoded, Base64.NO_WRAP)
        require(bytes.size == Int.SIZE_BYTES + expectedSize * Double.SIZE_BYTES)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(buffer.int == expectedSize)
        return DoubleArray(expectedSize) { buffer.double }.also {
            require(!buffer.hasRemaining())
            require(it.all(Double::isFinite))
        }
    }

    private fun JSONObject.optNullableString(key: String): String? =
        if (has(key) && !isNull(key)) getString(key) else null

    private fun storageKey(key: String): String = Base64.encodeToString(
        key.toByteArray(Charsets.UTF_8),
        Base64.NO_WRAP or Base64.URL_SAFE,
    )

    private fun preferences(context: Context) =
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)

    private const val PREFERENCES = "pending_notification_cancellations_v1"
    private const val VERSION = 3
    private const val TAG = "OnlineSdftProof"
}
