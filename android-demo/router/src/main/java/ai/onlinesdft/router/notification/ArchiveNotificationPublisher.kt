package ai.onlinesdft.router.notification

import ai.onlinesdft.router.MainActivity
import ai.onlinesdft.router.R
import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build

/** Silent Android-shade projection of unread, model-selected archives. */
object ArchiveNotificationPublisher {
    const val CHANNEL_ID = "online_sdft_archive_history"
    const val GROUP_KEY = "ai.onlinesdft.router.ARCHIVED_NOTIFICATIONS"

    fun sync(context: Context, items: List<DigestInboxItem>) {
        val archived = archiveGroupItems(items)
        val manager = managerIfEnabled(context) ?: return
        if (archived.isEmpty()) {
            manager.cancel(SUMMARY_TAG, SUMMARY_ID)
            return
        }
        archived.forEach { item ->
            manager.notify(childTag(item.eventId), childId(item.eventId), child(context, item))
        }
        manager.notify(SUMMARY_TAG, SUMMARY_ID, summary(context, archived))
    }

    fun cancel(context: Context, eventId: String, remainingItems: List<DigestInboxItem>) {
        val manager = context.getSystemService(NotificationManager::class.java)
        manager.cancel(childTag(eventId), childId(eventId))
        sync(context, remainingItems)
    }

    private fun managerIfEnabled(context: Context): NotificationManager? {
        if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) return null
        val manager = context.getSystemService(NotificationManager::class.java)
        if (!manager.areNotificationsEnabled()) return null
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Silenced notifications",
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = "A quiet group for notifications Tact decided not to interrupt you with"
                setSound(null, null)
                enableVibration(false)
                setShowBadge(true)
            },
        )
        if (manager.getNotificationChannel(CHANNEL_ID)?.importance == NotificationManager.IMPORTANCE_NONE) {
            return null
        }
        return manager
    }

    private fun child(context: Context, item: DigestInboxItem): Notification {
        val text = item.body.ifBlank { "Kept quiet for you" }
        return Notification.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_router)
            .setContentTitle(item.title)
            .setContentText(text)
            .setStyle(Notification.BigTextStyle().bigText(text))
            .setSubText("Silenced by Tact")
            .setCategory(Notification.CATEGORY_STATUS)
            .setGroup(GROUP_KEY)
            .setGroupAlertBehavior(Notification.GROUP_ALERT_SUMMARY)
            .setContentIntent(savedIntent(context, item.eventId))
            .setVisibility(Notification.VISIBILITY_SECRET)
            .setOnlyAlertOnce(true)
            .setAutoCancel(false)
            .build()
    }

    private fun summary(context: Context, archived: List<DigestInboxItem>): Notification {
        val count = archived.size
        val style = Notification.InboxStyle()
            .setBigContentTitle("$count notification${if (count == 1) "" else "s"} kept quiet")
        archived.take(SUMMARY_LINES).forEach { item -> style.addLine(item.title) }
        if (count > SUMMARY_LINES) style.setSummaryText("+${count - SUMMARY_LINES} more")
        return Notification.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_router)
            .setContentTitle("Silenced by Tact")
            .setContentText("$count kept quiet · swipe down to see them")
            .setStyle(style)
            .setCategory(Notification.CATEGORY_STATUS)
            .setGroup(GROUP_KEY)
            .setGroupSummary(true)
            .setGroupAlertBehavior(Notification.GROUP_ALERT_SUMMARY)
            .setContentIntent(savedIntent(context, SUMMARY_EVENT_ID))
            .setVisibility(Notification.VISIBILITY_SECRET)
            .setOnlyAlertOnce(true)
            .setAutoCancel(false)
            .setNumber(count)
            .build()
    }

    private fun savedIntent(context: Context, eventId: String): PendingIntent {
        val uri = Uri.Builder()
            .scheme("onlinesdft")
            .authority("archive-history")
            .appendPath(eventId)
            .build()
        return PendingIntent.getActivity(
            context,
            eventId.hashCode() and Int.MAX_VALUE,
            Intent(context, MainActivity::class.java).apply {
                action = MainActivity.ACTION_SHOW_SAVED
                data = uri
                addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    private fun childTag(eventId: String) = "online-sdft-archive:$eventId"
    private fun childId(eventId: String) = eventId.hashCode() and Int.MAX_VALUE

    private const val SUMMARY_TAG = "online-sdft-archive-summary"
    private const val SUMMARY_ID = 0x41524348
    private const val SUMMARY_EVENT_ID = "summary"
    private const val SUMMARY_LINES = 5
}

internal fun archiveGroupItems(items: List<DigestInboxItem>): List<DigestInboxItem> = items.filter {
    it.origin == DigestInboxOrigin.ROUTER_ARCHIVE && it.isUnread
}
