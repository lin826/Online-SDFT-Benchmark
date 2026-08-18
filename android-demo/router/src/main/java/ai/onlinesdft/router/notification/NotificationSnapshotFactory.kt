package ai.onlinesdft.router.notification

import ai.onlinesdft.router.model.FeatureEncoder
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.Regime
import android.app.Notification
import android.app.NotificationManager
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import java.security.MessageDigest
import java.util.Calendar

object NotificationSnapshotFactory {
    fun eligible(sbn: StatusBarNotification, routerPackage: String): Boolean {
        // Digest notifications and any future router-owned surfaces must never
        // feed back into the router. Every other notification delivered by the
        // OS listener filter is observed, including contentless/custom views.
        return sbn.packageName != routerPackage
    }

    fun create(
        sbn: StatusBarNotification,
        rankingMap: NotificationListenerService.RankingMap?,
    ): NotificationContext {
        val notification = sbn.notification
        val extras = notification.extras
        val trustedDemoExtras = sbn.packageName in DEMO_PUBLISHER_PACKAGES
        val requestedTimeoutMillis = if (trustedDemoExtras) {
            extras.getLong(EXTRA_TIMEOUT_AFTER_MILLIS, 0L)
                .takeIf { it in 1L..MAX_DEMO_TIMEOUT_MILLIS }
        } else {
            null
        }
        val trustedTimeoutMillis = requestedTimeoutMillis?.takeIf {
            notification.timeoutAfter == it
        }
        val semanticDelayMinutes = if (trustedTimeoutMillis != null) {
            extras.getInt(EXTRA_SEMANTIC_DELAY_MINUTES, 0)
                .takeIf { it in 1..MAX_SEMANTIC_DELAY_MINUTES }
        } else {
            null
        }
        val ranking = NotificationListenerService.Ranking()
        val ranked = rankingMap?.getRanking(sbn.key, ranking) == true
        val rankingImportance = if (ranked) {
            (ranking.importance / NotificationManager.IMPORTANCE_MAX.toFloat()).coerceIn(0f, 1f)
        } else {
            0.5f
        }
        val customImportance = if (trustedDemoExtras) {
            extras.getString(EXTRA_IMPORTANCE)?.let(::parseImportance)
        } else {
            null
        }
        val customCategory = if (trustedDemoExtras) extras.getString(EXTRA_CATEGORY) else null
        val category = FeatureEncoder.normalizeCategory(
            customCategory ?: notification.category ?: sbn.packageName,
        )
        val calendar = Calendar.getInstance().apply { timeInMillis = sbn.postTime }
        val hour = calendar.get(Calendar.HOUR_OF_DAY) + calendar.get(Calendar.MINUTE) / 60f
        val flags = notification.flags
        val template = extras.getString(Notification.EXTRA_TEMPLATE).orEmpty()
        val isCall = notification.category == Notification.CATEGORY_CALL ||
            extras.containsKey(Notification.EXTRA_CALL_TYPE) ||
            template.endsWith("CallStyle")
        val isMedia = notification.category == Notification.CATEGORY_TRANSPORT ||
            extras.containsKey(Notification.EXTRA_MEDIA_SESSION) ||
            template.endsWith("MediaStyle") ||
            template.endsWith("DecoratedMediaCustomViewStyle")
        val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString()
            ?.takeIf { it.isNotBlank() }
            ?: notification.tickerText?.toString()?.takeIf { it.isNotBlank() }
            ?: "Notification from ${sbn.packageName.substringAfterLast('.')}"
        return NotificationContext(
            eventId = if (trustedDemoExtras) {
                canonicalPublisherEventId(extras.getString(EXTRA_EVENT_ID), sbn)
            } else {
                opaqueEventId(sbn)
            },
            packageName = sbn.packageName,
            title = title,
            body = body(extras),
            category = category,
            importance = customImportance ?: rankingImportance,
            regime = Regime.fromWire(
                if (trustedDemoExtras) extras.getString(EXTRA_REGIME) else null,
            ),
            hourOfDay = hour,
            postedAtMillis = sbn.postTime,
            caseId = if (trustedDemoExtras) extras.getString(EXTRA_CASE_ID) else null,
            isClearable = sbn.isClearable,
            isOngoing = flags and Notification.FLAG_ONGOING_EVENT != 0,
            isForegroundService = flags and Notification.FLAG_FOREGROUND_SERVICE != 0,
            isCall = isCall,
            isMedia = isMedia,
            isGroupSummary = flags and Notification.FLAG_GROUP_SUMMARY != 0,
            isNoClear = flags and Notification.FLAG_NO_CLEAR != 0,
            demoTimeoutMillis = trustedTimeoutMillis,
            semanticDelayMinutes = semanticDelayMinutes,
        )
    }

    /**
     * Demo publishers expose a human-readable case id for capture scripts, but
     * Android permits the same value to be posted more than once. Preserve the
     * first id for readable proof logs and disambiguate only collisions so two
     * distinct source notifications can never collapse into one decision or
     * Saved-for-later row.
     */
    fun disambiguateEventId(
        context: NotificationContext,
        reservedEventIds: Set<String>,
    ): NotificationContext {
        if (context.eventId !in reservedEventIds) return context
        val stableInput = listOf(
            context.eventId,
            context.packageName,
            context.postedAtMillis.toString(),
            context.title,
            context.body,
        ).joinToString("\u0000")
        val suffix = MessageDigest.getInstance("SHA-256")
            .digest(stableInput.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
            .take(16)
        val stem = context.eventId.take(MAX_EVENT_ID_CHARS - suffix.length - 1)
        var candidate = "$stem-$suffix"
        var collision = 2
        while (candidate in reservedEventIds) {
            val counter = "-$collision"
            candidate = "$stem-$suffix".take(MAX_EVENT_ID_CHARS - counter.length) + counter
            collision += 1
        }
        return context.copy(
            eventId = candidate,
            caseId = context.caseId ?: context.eventId,
        )
    }

    private fun opaqueEventId(sbn: StatusBarNotification): String {
        val stableInput = "${sbn.key}:${sbn.postTime}"
        val hash = MessageDigest.getInstance("SHA-256")
            .digest(stableInput.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
        return "live-${hash.take(24)}"
    }

    private fun body(extras: android.os.Bundle): String =
        extras.getCharSequence(Notification.EXTRA_BIG_TEXT)?.toString()
            ?: extras.getCharSequence(Notification.EXTRA_TEXT)?.toString()
            ?: ""

    private fun parseImportance(raw: String): Float? = raw.toFloatOrNull()
        ?.takeIf(Float::isFinite)
        ?.let { value ->
        (if (value > 1f) value / 5f else value).coerceIn(0f, 1f)
    } ?: when (raw.trim().lowercase()) {
        "urgent", "critical", "max", "high" -> 0.95f
        "low", "min", "silent" -> 0.20f
        "default" -> 0.60f
        else -> null
    }

    private fun canonicalPublisherEventId(raw: String?, sbn: StatusBarNotification): String {
        val candidate = raw?.trim().orEmpty()
        if (
            candidate.isNotEmpty() &&
            candidate.length <= MAX_EVENT_ID_CHARS &&
            candidate.all { it.isLetterOrDigit() || it in EVENT_ID_PUNCTUATION }
        ) return candidate
        return opaqueEventId(sbn)
    }

    private const val EXTRA_CASE_ID = "case_id"
    private const val EXTRA_CATEGORY = "category"
    private const val EXTRA_IMPORTANCE = "importance"
    private const val EXTRA_REGIME = "regime"
    private const val EXTRA_EVENT_ID = "event_id"
    private const val EXTRA_TIMEOUT_AFTER_MILLIS = "timeout_after_millis"
    private const val EXTRA_SEMANTIC_DELAY_MINUTES = "semantic_delay_minutes"
    private const val MAX_DEMO_TIMEOUT_MILLIS = 60_000L
    private const val MAX_SEMANTIC_DELAY_MINUTES = 24 * 60
    private const val MAX_EVENT_ID_CHARS = 200
    private const val EVENT_ID_PUNCTUATION = "-_.:"
    private val DEMO_PUBLISHER_PACKAGES = setOf(
        "ai.onlinesdft.publisher.chat",
        "ai.onlinesdft.publisher.calendar",
        "ai.onlinesdft.publisher.mail",
    )

    fun isTrustedDemoTimeout(context: NotificationContext): Boolean =
        context.packageName in DEMO_PUBLISHER_PACKAGES &&
            context.demoTimeoutMillis == 2_000L &&
            context.semanticDelayMinutes == 120
}
