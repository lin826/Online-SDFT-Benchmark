package ai.onlinesdft.router.notification

import ai.onlinesdft.router.MainActivity
import ai.onlinesdft.router.OnlineSdftApplication
import ai.onlinesdft.router.R
import ai.onlinesdft.router.model.DecisionSnapshot
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

object DigestNotificationPublisher {
    const val CHANNEL_ID = "online_sdft_later_digest"

    fun canPublish(context: Context): Boolean {
        if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) return false
        val manager = context.getSystemService(NotificationManager::class.java)
        if (!manager.areNotificationsEnabled()) return false
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Saved for later alerts",
                NotificationManager.IMPORTANCE_DEFAULT,
            ).apply {
                description = "Reminders about notifications Tact saved for later"
            },
        )
        return manager.getNotificationChannel(CHANNEL_ID)?.importance !=
            NotificationManager.IMPORTANCE_NONE
    }

    fun publish(
        context: Context,
        decision: DecisionSnapshot,
        allowStaleDelivery: Boolean = false,
    ): DigestDeliveryResult {
        val runtime = OnlineSdftApplication.runtime(context)
        val record = runtime.recordDigest(decision, allowStaleDelivery)
            ?: return DigestDeliveryResult(saved = false, alertPosted = false)
        val digestEventId = record.eventId
        val openToken = record.openToken
        if (!canPublish(context)) {
            return DigestDeliveryResult(saved = true, alertPosted = false)
        }
        val manager = context.getSystemService(NotificationManager::class.java)
        val requestCode = digestEventId.hashCode() and Int.MAX_VALUE
        val eventUri = Uri.Builder()
            .scheme("onlinesdft")
            .authority("saved-for-later")
            .appendPath(digestEventId)
            .appendPath(openToken)
            .build()
        val openActivity = PendingIntent.getActivity(
            context,
            requestCode,
            Intent(context, MainActivity::class.java).apply {
                action = MainActivity.ACTION_OPEN_DIGEST
                data = eventUri
                putExtra(MainActivity.EXTRA_DIGEST_EVENT_ID, digestEventId)
                putExtra(MainActivity.EXTRA_DIGEST_OPEN_TOKEN, openToken)
                addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val delete = PendingIntent.getBroadcast(
            context,
            requestCode xor 0x51f7,
            Intent(context, DigestActionReceiver::class.java).apply {
                action = DigestActionReceiver.ACTION_DELETE
                data = eventUri.buildUpon().authority("saved-for-later-delete").build()
                putExtra(DigestActionReceiver.EXTRA_EVENT_ID, digestEventId)
                putExtra(DigestActionReceiver.EXTRA_OPEN_TOKEN, openToken)
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = Notification.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_router)
            .setContentTitle("Saved for later")
            .setContentText("Something is waiting for you when you have a moment.")
            .setSubText("Tact")
            .setCategory(Notification.CATEGORY_REMINDER)
            // Source apps can mark content secret. A generic, SECRET alert
            // never weakens that boundary; full content stays in the unlocked,
            // app-private inbox.
            .setVisibility(Notification.VISIBILITY_SECRET)
            .setAutoCancel(true)
            .setContentIntent(openActivity)
            .setDeleteIntent(delete)
            .build()
        val alertPosted = runtime.withDigestRecord(digestEventId, openToken) {
            runCatching {
                manager.notify(
                    "online-sdft-digest:$digestEventId",
                    requestCode,
                    notification,
                )
                true
            }.getOrDefault(false)
        }
        return DigestDeliveryResult(
            saved = alertPosted != null,
            alertPosted = alertPosted == true,
        )
    }

    fun clearAll(context: Context) {
        context.getSystemService(NotificationManager::class.java).cancelAll()
    }

    fun cancel(context: Context, eventId: String) {
        context.getSystemService(NotificationManager::class.java).cancel(
            "online-sdft-digest:$eventId",
            eventId.hashCode() and Int.MAX_VALUE,
        )
    }
}

data class DigestDeliveryResult(
    val saved: Boolean,
    val alertPosted: Boolean,
)
