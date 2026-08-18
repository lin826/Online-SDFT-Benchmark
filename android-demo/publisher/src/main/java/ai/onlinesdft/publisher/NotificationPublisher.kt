package ai.onlinesdft.publisher

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Person
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.drawable.Icon
import android.media.session.MediaSession
import android.media.session.PlaybackState
import android.os.Build
import android.os.Bundle
import java.util.concurrent.ConcurrentHashMap

private const val MAX_DEMO_TIMEOUT_AFTER_MILLIS = 60_000L
private const val MAX_SEMANTIC_DELAY_MINUTES = 24 * 60

internal enum class NotificationSurface(val wireName: String) {
    STANDARD("standard"),
    ONGOING("ongoing"),
    FOREGROUND_SERVICE("foreground-service"),
    CALL("call"),
    MEDIA("media");

    companion object {
        fun fromWire(raw: String?): NotificationSurface = entries.firstOrNull {
            it.wireName == raw?.trim()?.lowercase()
        } ?: STANDARD
    }
}

internal data class CasePayload(
    val caseId: String,
    val title: String,
    val body: String,
    val category: String,
    val importance: String,
    val regime: String,
    val eventId: String,
    val surface: NotificationSurface,
    val timeoutAfterMillis: Long,
    val semanticDelayMinutes: Int,
) {
    fun asNotificationExtras(): Bundle = Bundle().apply {
        putString(CaseNotificationReceiver.EXTRA_CASE_ID, caseId)
        putString(CaseNotificationReceiver.EXTRA_TITLE, title)
        putString(CaseNotificationReceiver.EXTRA_BODY, body)
        putString(CaseNotificationReceiver.EXTRA_CATEGORY, category)
        putString(CaseNotificationReceiver.EXTRA_IMPORTANCE, importance)
        putString(CaseNotificationReceiver.EXTRA_REGIME, regime)
        putString(CaseNotificationReceiver.EXTRA_EVENT_ID, eventId)
        putString(CaseNotificationReceiver.EXTRA_SURFACE, surface.wireName)
        putLong(CaseNotificationReceiver.EXTRA_TIMEOUT_AFTER_MILLIS, timeoutAfterMillis)
        putInt(CaseNotificationReceiver.EXTRA_SEMANTIC_DELAY_MINUTES, semanticDelayMinutes)
    }

    companion object {
        fun fromIntent(context: Context, intent: Intent): CasePayload {
            val now = System.currentTimeMillis()
            val defaultCategory = context.getString(R.string.publisher_category)
            val category = intent.textExtra(CaseNotificationReceiver.EXTRA_CATEGORY)
                ?: defaultCategory
            val caseId = intent.textExtra(CaseNotificationReceiver.EXTRA_CASE_ID)
                ?: "$defaultCategory-$now"

            return CasePayload(
                caseId = caseId,
                title = intent.textExtra(CaseNotificationReceiver.EXTRA_TITLE)
                    ?: "New $category item",
                body = intent.textExtra(CaseNotificationReceiver.EXTRA_BODY)
                    ?: "No message body supplied.",
                category = category,
                importance = intent.textExtra(CaseNotificationReceiver.EXTRA_IMPORTANCE)
                    ?: "default",
                regime = intent.textExtra(CaseNotificationReceiver.EXTRA_REGIME)
                    ?: "demo",
                eventId = intent.textExtra(CaseNotificationReceiver.EXTRA_EVENT_ID)
                    ?: caseId,
                surface = NotificationSurface.fromWire(
                    intent.textExtra(CaseNotificationReceiver.EXTRA_SURFACE),
                ),
                timeoutAfterMillis = intent
                    .getLongExtra(CaseNotificationReceiver.EXTRA_TIMEOUT_AFTER_MILLIS, 0L)
                    .coerceIn(0L, MAX_DEMO_TIMEOUT_AFTER_MILLIS),
                semanticDelayMinutes = intent
                    .getIntExtra(CaseNotificationReceiver.EXTRA_SEMANTIC_DELAY_MINUTES, 0)
                    .coerceIn(0, MAX_SEMANTIC_DELAY_MINUTES),
            )
        }

        private fun Intent.textExtra(key: String): String? =
            getStringExtra(key)?.trim()?.takeIf(String::isNotEmpty)
    }
}

internal object NotificationPublisher {
    enum class PostResult {
        POSTED,
        PERMISSION_REQUIRED,
        NOTIFICATIONS_DISABLED,
        POST_FAILED,
    }

    private data class ChannelSpec(
        val id: String,
        val name: String,
        val importance: Int,
    )

    private val highChannel = ChannelSpec(
        id = "online_sdft_cases_high",
        name = "High-priority demo cases",
        importance = NotificationManager.IMPORTANCE_HIGH,
    )
    private val defaultChannel = ChannelSpec(
        id = "online_sdft_cases_default",
        name = "Demo cases",
        importance = NotificationManager.IMPORTANCE_DEFAULT,
    )
    private val lowChannel = ChannelSpec(
        id = "online_sdft_cases_low",
        name = "Low-priority demo cases",
        importance = NotificationManager.IMPORTANCE_LOW,
    )

    fun ensureChannels(context: Context) {
        val manager = context.getSystemService(NotificationManager::class.java)
        listOf(highChannel, defaultChannel, lowChannel).forEach { spec ->
            val channel = NotificationChannel(spec.id, spec.name, spec.importance).apply {
                description =
                    "Real Android notifications posted by a separate Online SDFT demo source app"
            }
            manager.createNotificationChannel(channel)
        }
    }

    fun post(context: Context, payload: CasePayload): PostResult {
        ensureChannels(context)
        if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            return PostResult.PERMISSION_REQUIRED
        }

        val manager = context.getSystemService(NotificationManager::class.java)
        if (!manager.areNotificationsEnabled()) return PostResult.NOTIFICATIONS_DISABLED

        if (
            payload.surface == NotificationSurface.FOREGROUND_SERVICE ||
            payload.surface == NotificationSurface.CALL
        ) {
            return if (DemoForegroundService.start(context, payload)) {
                PostResult.POSTED
            } else {
                PostResult.POST_FAILED
            }
        }

        val notificationId = payload.eventId.hashCode() and Int.MAX_VALUE
        return try {
            manager.notify(
                "online-sdft:${payload.eventId}",
                notificationId,
                buildNotification(context, payload, notificationId),
            )
            PostResult.POSTED
        } catch (_: RuntimeException) {
            PostResult.POST_FAILED
        }
    }

    fun cleanup(context: Context) {
        DemoForegroundService.stop(context)
        context.getSystemService(NotificationManager::class.java).cancelAll()
        mediaSessions.values.forEach { session ->
            session.isActive = false
            session.release()
        }
        mediaSessions.clear()
    }

    fun buildNotification(
        context: Context,
        payload: CasePayload,
        notificationId: Int = payload.eventId.hashCode() and Int.MAX_VALUE,
    ): Notification {
        val channel = channelFor(payload.importance)
        val openIntent = Intent(context, MainActivity::class.java).apply {
            putExtras(payload.asNotificationExtras())
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val contentIntent = PendingIntent.getActivity(
            context,
            notificationId,
            openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        val builder = Notification.Builder(context, channel.id)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(payload.title)
            .setContentText(payload.body)
            .setStyle(Notification.BigTextStyle().bigText(payload.body))
            .setSubText("${payload.category} · ${payload.regime}")
            .setCategory(androidCategory(payload))
            .setVisibility(Notification.VISIBILITY_PRIVATE)
            .setAutoCancel(true)
            .setContentIntent(contentIntent)
            .setExtras(payload.asNotificationExtras())

        if (
            payload.surface == NotificationSurface.STANDARD &&
            payload.timeoutAfterMillis > 0L
        ) {
            builder.setTimeoutAfter(payload.timeoutAfterMillis)
        }

        when (payload.surface) {
            NotificationSurface.STANDARD -> Unit
            NotificationSurface.ONGOING -> builder.setOngoing(true).setAutoCancel(false)
            // startForeground() supplies FLAG_FOREGROUND_SERVICE. Keep this
            // fixture independent from FLAG_ONGOING_EVENT so the router proves
            // both traits and constraints separately.
            NotificationSurface.FOREGROUND_SERVICE -> builder.setAutoCancel(false)
            NotificationSurface.CALL -> applyCallStyle(
                context,
                payload,
                notificationId,
                builder,
            )
            NotificationSurface.MEDIA -> applyMediaStyle(
                context,
                payload,
                notificationId,
                builder,
            )
        }
        return builder.build()
    }

    private fun applyCallStyle(
        context: Context,
        payload: CasePayload,
        notificationId: Int,
        builder: Notification.Builder,
    ) {
        builder.setCategory(Notification.CATEGORY_CALL).setOngoing(true).setAutoCancel(false)
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.S) return
        val caller = Person.Builder().setName(payload.title).setImportant(true).build()
        val decline = actionIntent(context, payload, notificationId + 1, "decline")
        val answer = actionIntent(context, payload, notificationId + 2, "answer")
        builder.setStyle(Notification.CallStyle.forIncomingCall(caller, decline, answer))
    }

    private fun applyMediaStyle(
        context: Context,
        payload: CasePayload,
        notificationId: Int,
        builder: Notification.Builder,
    ) {
        val session = mediaSessions.computeIfAbsent(payload.eventId) {
            MediaSession(context, "OnlineSdftDemo:${payload.eventId}").apply {
                setPlaybackState(
                    PlaybackState.Builder()
                        .setState(PlaybackState.STATE_PLAYING, 0L, 1f)
                        .setActions(PlaybackState.ACTION_PLAY_PAUSE)
                        .build(),
                )
                isActive = true
            }
        }
        builder
            .setCategory(Notification.CATEGORY_TRANSPORT)
            .setOngoing(true)
            .setAutoCancel(false)
            .addAction(
                Notification.Action.Builder(
                    Icon.createWithResource(context, android.R.drawable.ic_media_pause),
                    "Pause",
                    actionIntent(context, payload, notificationId + 3, "pause"),
                ).build(),
            )
            .setStyle(
                Notification.MediaStyle()
                    .setMediaSession(session.sessionToken)
                    .setShowActionsInCompactView(0),
            )
    }

    private fun actionIntent(
        context: Context,
        payload: CasePayload,
        requestCode: Int,
        action: String,
    ): PendingIntent {
        val intent = Intent(context, MainActivity::class.java).apply {
            putExtras(payload.asNotificationExtras())
            putExtra("demo_action", action)
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        return PendingIntent.getActivity(
            context,
            requestCode,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    private fun channelFor(importance: String): ChannelSpec =
        when (importance.lowercase()) {
            "urgent", "critical", "max", "high", "4", "5" -> highChannel
            "min", "low", "silent", "1", "2" -> lowChannel
            else -> defaultChannel
        }

    private fun androidCategory(payload: CasePayload): String = when (payload.surface) {
        NotificationSurface.CALL -> Notification.CATEGORY_CALL
        NotificationSurface.MEDIA -> Notification.CATEGORY_TRANSPORT
        else -> when (payload.category.lowercase()) {
            "chat", "message", "messages" -> Notification.CATEGORY_MESSAGE
            "calendar", "event", "meeting" -> Notification.CATEGORY_EVENT
            "mail", "email" -> Notification.CATEGORY_EMAIL
            else -> Notification.CATEGORY_STATUS
        }

    }

    private val mediaSessions = ConcurrentHashMap<String, MediaSession>()
}
