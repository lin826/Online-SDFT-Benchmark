package ai.onlinesdft.publisher

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.util.Log

class MainActivity : Activity() {
    private lateinit var permissionStatus: TextView
    private lateinit var deliveryStatus: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        NotificationPublisher.ensureChannels(this)
        setContentView(createContentView())
        showOpenedNotification(intent)
        requestNotificationPermissionIfNeeded()
    }

    override fun onResume() {
        super.onResume()
        if (::permissionStatus.isInitialized) updatePermissionStatus()
        handleSurfaceLaunch(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        showOpenedNotification(intent)
        handleSurfaceLaunch(intent)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_NOTIFICATIONS) updatePermissionStatus()
    }

    private fun createContentView(): ScrollView {
        val category = getString(R.string.publisher_category)
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(dp(24), dp(40), dp(24), dp(32))
        }

        content.addView(textView(getString(R.string.app_name), 28f, Color.rgb(28, 31, 42)).apply {
            gravity = Gravity.CENTER
        })
        content.addView(textView(
            "Separate source app · posts genuine Android $category notifications for " +
                "the on-device Online SDFT routing demo.",
            17f,
            Color.rgb(72, 77, 94),
        ).withTopMargin(dp(16)))

        permissionStatus = textView("", 15f, Color.rgb(72, 77, 94)).apply {
            gravity = Gravity.CENTER
        }
        content.addView(permissionStatus.withTopMargin(dp(28)))

        content.addView(Button(this).apply {
            text = "Enable notifications"
            isAllCaps = false
            setOnClickListener { requestNotificationPermissionIfNeeded() }
        }.withTopMargin(dp(12)))

        content.addView(Button(this).apply {
            text = "Post a realistic sample"
            isAllCaps = false
            setOnClickListener { postSample(category) }
        }.withTopMargin(dp(20)))

        deliveryStatus = textView(
            "External scripts can also send ai.onlinesdft.publisher.POST_CASE to this package.",
            14f,
            Color.rgb(92, 97, 112),
        ).apply { gravity = Gravity.CENTER }
        content.addView(deliveryStatus.withTopMargin(dp(20)))

        return ScrollView(this).apply {
            setBackgroundColor(Color.rgb(247, 248, 252))
            addView(
                content,
                ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
        }
    }

    private fun postSample(category: String) {
        val sample = sampleFor(category)
        val now = System.currentTimeMillis()
        val eventId = "$category-ui-$now"
        val broadcast = Intent(CaseNotificationReceiver.ACTION_POST_CASE).apply {
            setClass(this@MainActivity, CaseNotificationReceiver::class.java)
            putExtra(CaseNotificationReceiver.EXTRA_CASE_ID, "$category-case-$now")
            putExtra(CaseNotificationReceiver.EXTRA_TITLE, sample.first)
            putExtra(CaseNotificationReceiver.EXTRA_BODY, sample.second)
            putExtra(CaseNotificationReceiver.EXTRA_CATEGORY, category)
            putExtra(CaseNotificationReceiver.EXTRA_IMPORTANCE, sample.third)
            putExtra(CaseNotificationReceiver.EXTRA_REGIME, "demo-personal")
            putExtra(CaseNotificationReceiver.EXTRA_EVENT_ID, eventId)
        }
        sendBroadcast(broadcast)
        deliveryStatus.text = "Broadcast sent for event $eventId"
    }

    private fun sampleFor(category: String): Triple<String, String, String> =
        when (category) {
            "chat" -> Triple(
                "Maya · Dinner moved to 7:30",
                "Can you confirm the reservation still works? No rush if you are in focus time.",
                "high",
            )
            "calendar" -> Triple(
                "Design review starts in 15 minutes",
                "Room Atlas · Notification routing prototype and on-device learning review.",
                "default",
            )
            else -> Triple(
                "Weekly product metrics are ready",
                "The router shows local accuracy, regret, and adaptation after feedback.",
                "low",
            )
        }

    private fun showOpenedNotification(intent: Intent) {
        if (!::deliveryStatus.isInitialized) return
        val eventId = intent.getStringExtra(CaseNotificationReceiver.EXTRA_EVENT_ID) ?: return
        val caseId = intent.getStringExtra(CaseNotificationReceiver.EXTRA_CASE_ID).orEmpty()
        deliveryStatus.text = "Opened notification · event $eventId · case $caseId"
    }

    private fun handleSurfaceLaunch(intent: Intent) {
        if (intent.action != ACTION_START_SURFACE) return
        // Consume the launch exactly once even when onNewIntent is immediately
        // followed by onResume.
        intent.action = null
        val requestedPayload = CasePayload.fromIntent(this, intent)
        if (
            requestedPayload.surface != NotificationSurface.FOREGROUND_SERVICE &&
            requestedPayload.surface != NotificationSurface.CALL
        ) {
            deliveryStatus.text = "Unsupported external surface fixture"
            Log.w(
                "OnlineSdftPublisher",
                "SURFACE_LAUNCH_REJECTED source_package=$packageName " +
                    "surface=${requestedPayload.surface.wireName}",
            )
            return
        }
        // The exported launcher exists only because Android requires a visible
        // start for the call/foreground-service fixtures. Never let it mint the
        // trusted standard-notification timeout used by capture automation;
        // that path is exclusively the DUMP-protected receiver.
        val payload = requestedPayload.copy(
            timeoutAfterMillis = 0L,
            semanticDelayMinutes = 0,
        )
        val result = NotificationPublisher.post(this, payload)
        deliveryStatus.text = "${payload.surface.wireName} fixture · ${result.name.lowercase()}"
        Log.i(
            "OnlineSdftPublisher",
            "REAL_SOURCE_POST event_id=${payload.eventId} source_package=$packageName " +
                "surface=${payload.surface.wireName} result=${result.name}",
        )
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQUEST_NOTIFICATIONS)
        } else {
            updatePermissionStatus()
        }
    }

    private fun updatePermissionStatus() {
        val granted =
            Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
                checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) ==
                PackageManager.PERMISSION_GRANTED
        permissionStatus.text = if (granted) {
            "Notifications enabled"
        } else {
            "Notification permission is required before cases can appear"
        }
        permissionStatus.setTextColor(
            if (granted) Color.rgb(23, 112, 70) else Color.rgb(172, 75, 40),
        )
    }

    private fun textView(text: String, size: Float, color: Int): TextView =
        TextView(this).apply {
            this.text = text
            textSize = size
            setTextColor(color)
            setLineSpacing(0f, 1.15f)
        }

    private fun <T : android.view.View> T.withTopMargin(margin: Int): T = apply {
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ).apply { topMargin = margin }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    companion object {
        const val ACTION_START_SURFACE = "ai.onlinesdft.publisher.START_SURFACE"
        private const val REQUEST_NOTIFICATIONS = 42
    }
}
