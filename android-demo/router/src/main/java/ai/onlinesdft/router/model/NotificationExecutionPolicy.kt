package ai.onlinesdft.router.model

data class RouteExecutionPlan(
    val recommendedRoute: Route,
    val effectiveRoute: Route,
    val constraint: ExecutionConstraint,
)

/**
 * Keeps the model recommendation distinct from what a normal Android app can
 * safely apply. Protected surfaces are still observed, scored, and available
 * for explicit preference learning, but remain visible to preserve their
 * controls and active-service state.
 */
object NotificationExecutionPolicy {
    fun plan(context: NotificationContext, recommendedRoute: Route): RouteExecutionPlan {
        val surfaceConstraint = constraint(context)
        val constraint = if (
            surfaceConstraint == ExecutionConstraint.NONE &&
            recommendedRoute == Route.LATER &&
            !context.canPublishDigest
        ) {
            ExecutionConstraint.DIGEST_UNAVAILABLE
        } else {
            surfaceConstraint
        }
        val effectiveRoute = if (
            constraint != ExecutionConstraint.NONE && recommendedRoute != Route.INTERRUPT
        ) {
            Route.INTERRUPT
        } else {
            recommendedRoute
        }
        return RouteExecutionPlan(recommendedRoute, effectiveRoute, constraint)
    }

    fun constraint(context: NotificationContext): ExecutionConstraint = when {
        context.isCall -> ExecutionConstraint.CALL
        context.isMedia -> ExecutionConstraint.MEDIA
        context.isGroupSummary -> ExecutionConstraint.GROUP_SUMMARY
        context.isOngoing -> ExecutionConstraint.ONGOING
        context.isForegroundService -> ExecutionConstraint.FOREGROUND_SERVICE
        !context.isClearable || context.isNoClear -> ExecutionConstraint.NON_CLEARABLE
        else -> ExecutionConstraint.NONE
    }
}
