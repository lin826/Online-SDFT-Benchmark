package ai.onlinesdft.router.notification

import ai.onlinesdft.router.OnlineSdftApplication
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class DigestActionReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val eventId = intent.getStringExtra(EXTRA_EVENT_ID) ?: return
        val openToken = intent.getStringExtra(EXTRA_OPEN_TOKEN) ?: return
        val runtime = OnlineSdftApplication.runtime(context)
        when (intent.action) {
            ACTION_OPEN -> runtime.openDigestFromNotification(eventId, openToken)
            ACTION_DELETE -> runtime.dismissDigestFromNotification(eventId, openToken)
        }
    }

    companion object {
        const val ACTION_OPEN = "ai.onlinesdft.router.DIGEST_OPEN"
        const val ACTION_DELETE = "ai.onlinesdft.router.DIGEST_DELETE"
        const val EXTRA_EVENT_ID = "event_id"
        const val EXTRA_OPEN_TOKEN = "open_token"
    }
}
