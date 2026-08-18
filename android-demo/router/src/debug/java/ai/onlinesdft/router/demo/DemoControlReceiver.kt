package ai.onlinesdft.router.demo

import ai.onlinesdft.router.OnlineSdftApplication
import ai.onlinesdft.router.model.Route
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.util.Log

/** Debug-build automation ingress used by the reproducible capture script. */
class DemoControlReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) {
            Log.w(TAG, "Ignoring demo control in a non-debuggable build")
            return
        }
        if (intent.action != ACTION_DEMO_CONTROL) return
        val runtime = OnlineSdftApplication.runtime(context)
        val rawCommand = intent.getStringExtra(EXTRA_COMMAND)?.trim().orEmpty()
        when {
            rawCommand.equals(COMMAND_RESET, ignoreCase = true) -> runtime.reset()
            rawCommand.equals(COMMAND_RUN_ACCELERATED, ignoreCase = true) ->
                runtime.runAcceleratedLab()
            rawCommand.equals(COMMAND_RUN_PREFERENCE_SHIFT, ignoreCase = true) ->
                runtime.runPreferenceShiftDemo()
            rawCommand.equals(COMMAND_SHOW_SCORES, ignoreCase = true) -> runtime.showScores()
            rawCommand.equals(COMMAND_SHOW_DIGEST, ignoreCase = true) -> runtime.showDigest()
            rawCommand.equals(COMMAND_SHOW_ROUTER_EVENT, ignoreCase = true) ->
                runtime.showRouterEvent(intent.getStringExtra(EXTRA_EVENT_ID))
            rawCommand.equals(COMMAND_SHOW_ROUTER, ignoreCase = true) -> runtime.showRouter()
            rawCommand.equals(COMMAND_FEEDBACK, ignoreCase = true) ||
                rawCommand.startsWith("FEEDBACK(", ignoreCase = true) -> {
                val compactRoute = rawCommand
                    .substringAfter('(', missingDelimiterValue = "")
                    .substringBefore(')', missingDelimiterValue = "")
                val route = Route.fromWire(
                    intent.getStringExtra(EXTRA_ROUTE) ?: compactRoute,
                ) ?: return
                runtime.submitExplicitCorrection(
                    intent.getStringExtra(EXTRA_EVENT_ID),
                    route,
                )
            }
            else -> Log.w(TAG, "Unknown demo command: $rawCommand")
        }
    }

    companion object {
        const val ACTION_DEMO_CONTROL = "ai.onlinesdft.router.DEMO_CONTROL"
        const val EXTRA_COMMAND = "command"
        const val EXTRA_ROUTE = "route"
        const val EXTRA_EVENT_ID = "event_id"
        const val COMMAND_RESET = "RESET"
        const val COMMAND_FEEDBACK = "FEEDBACK"
        const val COMMAND_RUN_ACCELERATED = "RUN_ACCELERATED"
        const val COMMAND_RUN_PREFERENCE_SHIFT = "RUN_PREFERENCE_SHIFT"
        const val COMMAND_SHOW_SCORES = "SHOW_SCORES"
        const val COMMAND_SHOW_DIGEST = "SHOW_DIGEST"
        const val COMMAND_SHOW_ROUTER = "SHOW_ROUTER"
        const val COMMAND_SHOW_ROUTER_EVENT = "SHOW_ROUTER_EVENT"
        private const val TAG = "OnlineSdftControl"
    }
}
