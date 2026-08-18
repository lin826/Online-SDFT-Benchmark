package ai.onlinesdft.router

import ai.onlinesdft.router.model.Route
import ai.onlinesdft.router.notification.DigestNotificationPublisher
import ai.onlinesdft.router.notification.RouterNotificationListenerService
import ai.onlinesdft.router.ui.OnlineSdftApp
import android.Manifest
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.app.NotificationManagerCompat

class MainActivity : ComponentActivity() {
    private val runtime by lazy { OnlineSdftApplication.runtime(this) }
    private var listenerAccess by mutableStateOf(false)
    private var digestNotificationsEnabled by mutableStateOf(false)
    private val notificationPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { refreshDigestAvailability() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        refreshListenerAccess()
        refreshDigestAvailability()
        handleDigestIntent(intent)
        setContent {
            OnlineSdftApp(
                stateFlow = runtime.state,
                listenerAccess = listenerAccess,
                digestNotificationsEnabled = digestNotificationsEnabled,
                onOpenNotificationAccess = ::openNotificationAccess,
                onRequestDigestPermission = ::requestDigestNotifications,
                onReset = runtime::reset,
                onShowRouter = runtime::showRouter,
                onShowDigest = runtime::showDigest,
                onShowScores = runtime::showScores,
                onPageVisible = runtime::pageVisible,
                onSelectRouterEvent = runtime::selectRouterEvent,
                onOpenDigest = runtime::openDigest,
                onRemoveDigest = runtime::dismissDigest,
                onArchivePreference = runtime::submitArchivedPreference,
                onCorrection = { eventId, route ->
                    runtime.submitExplicitCorrection(eventId, route)
                },
            )
        }
    }

    override fun onResume() {
        super.onResume()
        refreshListenerAccess()
        refreshDigestAvailability()
        runtime.syncArchiveNotifications()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleDigestIntent(intent)
    }

    private fun refreshListenerAccess() {
        listenerAccess = NotificationManagerCompat
            .getEnabledListenerPackages(this)
            .contains(packageName)
    }

    private fun refreshDigestAvailability() {
        digestNotificationsEnabled = DigestNotificationPublisher.canPublish(this)
    }

    private fun requestDigestNotifications() {
        if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            val preferences = getSharedPreferences(PERMISSION_PREFERENCES, MODE_PRIVATE)
            val requestedBefore = preferences.getBoolean(KEY_DIGEST_PERMISSION_REQUESTED, false)
            if (
                !requestedBefore ||
                shouldShowRequestPermissionRationale(Manifest.permission.POST_NOTIFICATIONS)
            ) {
                preferences.edit().putBoolean(KEY_DIGEST_PERMISSION_REQUESTED, true).apply()
                notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
            } else {
                openAppNotificationSettings()
            }
            return
        }
        val intent = Intent(Settings.ACTION_CHANNEL_NOTIFICATION_SETTINGS)
            .putExtra(Settings.EXTRA_APP_PACKAGE, packageName)
            .putExtra(Settings.EXTRA_CHANNEL_ID, DigestNotificationPublisher.CHANNEL_ID)
        runCatching { startActivity(intent) }.onFailure {
            openAppNotificationSettings()
        }
    }

    private fun openAppNotificationSettings() {
        startActivity(
            Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
                .putExtra(Settings.EXTRA_APP_PACKAGE, packageName),
        )
    }

    private fun handleDigestIntent(intent: Intent?) {
        when (intent?.action) {
            ACTION_SHOW_SAVED -> {
                runtime.showDigest()
                intent.action = null
            }
            ACTION_OPEN_DIGEST -> {
                val eventId = intent.getStringExtra(EXTRA_DIGEST_EVENT_ID) ?: return
                val openToken = intent.getStringExtra(EXTRA_DIGEST_OPEN_TOKEN) ?: return
                runtime.openDigestFromNotification(eventId, openToken)
                intent.action = null
            }
        }
    }

    private fun openNotificationAccess() {
        val fallback = Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
        val intent = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Intent(Settings.ACTION_NOTIFICATION_LISTENER_DETAIL_SETTINGS).putExtra(
                Settings.EXTRA_NOTIFICATION_LISTENER_COMPONENT_NAME,
                ComponentName(this, RouterNotificationListenerService::class.java)
                    .flattenToString(),
            )
        } else {
            fallback
        }
        runCatching { startActivity(intent) }.onFailure { startActivity(fallback) }
    }

    companion object {
        const val ACTION_OPEN_DIGEST = "ai.onlinesdft.router.OPEN_DIGEST"
        const val ACTION_SHOW_SAVED = "ai.onlinesdft.router.SHOW_SAVED"
        const val EXTRA_DIGEST_EVENT_ID = "digest_event_id"
        const val EXTRA_DIGEST_OPEN_TOKEN = "digest_open_token"
        private const val PERMISSION_PREFERENCES = "customer_permissions"
        private const val KEY_DIGEST_PERMISSION_REQUESTED = "digest_notifications_requested"
    }
}
