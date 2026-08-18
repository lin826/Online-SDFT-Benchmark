package ai.onlinesdft.publisher

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * Shell-only demo ingress used by the local capture scripts.
 *
 * The exported component requires Android's DUMP permission. ADB shell automation
 * can still post a genuine notification from this separate package, while ordinary
 * installed apps cannot mint trusted demo-timeout extras through the receiver.
 */
class CaseNotificationReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == ACTION_CLEANUP) {
            NotificationPublisher.cleanup(context)
            Log.i(TAG, "DEMO_SURFACES_CLEANED source_package=${context.packageName}")
            return
        }
        if (intent.action != ACTION_POST_CASE) return

        val payload = CasePayload.fromIntent(context, intent)
        when (NotificationPublisher.post(context, payload)) {
            NotificationPublisher.PostResult.POSTED -> {
                Log.i(
                    TAG,
                    "REAL_SOURCE_POST event_id=${payload.eventId} " +
                        "source_package=${context.packageName} " +
                        "demo_timeout_ms=${payload.timeoutAfterMillis} " +
                        "semantic_delay_minutes=${payload.semanticDelayMinutes}",
                )
            }
            NotificationPublisher.PostResult.PERMISSION_REQUIRED -> {
                Log.w(TAG, "Notification permission is required; open the publisher app once")
            }
            NotificationPublisher.PostResult.NOTIFICATIONS_DISABLED -> {
                Log.w(TAG, "Notifications are disabled for ${context.packageName}")
            }
            NotificationPublisher.PostResult.POST_FAILED -> {
                Log.e(TAG, "Android rejected the requested notification surface")
            }
        }
    }

    companion object {
        const val ACTION_POST_CASE = "ai.onlinesdft.publisher.POST_CASE"
        const val ACTION_CLEANUP = "ai.onlinesdft.publisher.CLEANUP"

        const val EXTRA_CASE_ID = "case_id"
        const val EXTRA_TITLE = "title"
        const val EXTRA_BODY = "body"
        const val EXTRA_CATEGORY = "category"
        const val EXTRA_IMPORTANCE = "importance"
        const val EXTRA_REGIME = "regime"
        const val EXTRA_EVENT_ID = "event_id"
        const val EXTRA_SURFACE = "surface"
        const val EXTRA_TIMEOUT_AFTER_MILLIS = "timeout_after_millis"
        const val EXTRA_SEMANTIC_DELAY_MINUTES = "semantic_delay_minutes"

        private const val TAG = "OnlineSdftPublisher"
    }
}
