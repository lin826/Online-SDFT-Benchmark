package ai.onlinesdft.publisher

import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log

/** Genuine foreground-service fixture used to exercise listener classification. */
class DemoForegroundService : Service() {
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val sourceIntent = intent ?: return START_NOT_STICKY
        val payload = CasePayload.fromIntent(this, sourceIntent)
        NotificationPublisher.ensureChannels(this)
        val notificationId = payload.eventId.hashCode() and Int.MAX_VALUE
        startForeground(
            notificationId,
            NotificationPublisher.buildNotification(this, payload, notificationId),
        )
        Log.i(
            TAG,
            "REAL_FOREGROUND_SERVICE event_id=${payload.eventId} source_package=$packageName",
        )
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
        super.onDestroy()
    }

    companion object {
        internal fun start(context: Context, payload: CasePayload): Boolean = try {
            val intent = Intent(context, DemoForegroundService::class.java).apply {
                putExtras(payload.asNotificationExtras())
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
            true
        } catch (error: RuntimeException) {
            Log.e(TAG, "Unable to start demo foreground service", error)
            false
        }

        internal fun stop(context: Context) {
            context.stopService(Intent(context, DemoForegroundService::class.java))
        }

        private const val TAG = "OnlineSdftPublisher"
    }
}
