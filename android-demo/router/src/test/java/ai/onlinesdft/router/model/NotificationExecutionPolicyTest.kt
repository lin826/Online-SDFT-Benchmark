package ai.onlinesdft.router.model

import org.junit.Assert.assertEquals
import org.junit.Test

class NotificationExecutionPolicyTest {
    @Test
    fun `ordinary notification applies the model recommendation`() {
        Route.entries.forEach { route ->
            val plan = NotificationExecutionPolicy.plan(context(), route)
            assertEquals(route, plan.recommendedRoute)
            assertEquals(route, plan.effectiveRoute)
            assertEquals(ExecutionConstraint.NONE, plan.constraint)
        }
    }

    @Test
    fun `every protected surface is observed but destructive routes pass through`() {
        val protected = listOf(
            context(isOngoing = true) to ExecutionConstraint.ONGOING,
            context(isForegroundService = true) to ExecutionConstraint.FOREGROUND_SERVICE,
            context(isCall = true) to ExecutionConstraint.CALL,
            context(isMedia = true) to ExecutionConstraint.MEDIA,
            context(isGroupSummary = true) to ExecutionConstraint.GROUP_SUMMARY,
            context(isClearable = false) to ExecutionConstraint.NON_CLEARABLE,
            context(isNoClear = true) to ExecutionConstraint.NON_CLEARABLE,
        )

        protected.forEach { (notification, expectedConstraint) ->
            listOf(Route.LATER, Route.ARCHIVE).forEach { recommendation ->
                val plan = NotificationExecutionPolicy.plan(notification, recommendation)
                assertEquals(recommendation, plan.recommendedRoute)
                assertEquals(Route.INTERRUPT, plan.effectiveRoute)
                assertEquals(expectedConstraint, plan.constraint)
            }
            val interruptPlan = NotificationExecutionPolicy.plan(notification, Route.INTERRUPT)
            assertEquals(Route.INTERRUPT, interruptPlan.effectiveRoute)
            assertEquals(expectedConstraint, interruptPlan.constraint)
        }
    }

    @Test
    fun `missing digest capability clamps only Later`() {
        val notification = context().copy(canPublishDigest = false)
        val later = NotificationExecutionPolicy.plan(notification, Route.LATER)
        assertEquals(Route.INTERRUPT, later.effectiveRoute)
        assertEquals(ExecutionConstraint.DIGEST_UNAVAILABLE, later.constraint)

        val archive = NotificationExecutionPolicy.plan(notification, Route.ARCHIVE)
        assertEquals(Route.ARCHIVE, archive.effectiveRoute)
        assertEquals(ExecutionConstraint.NONE, archive.constraint)
    }

    @Test
    fun `surface traits are independent decision-visible features`() {
        val features = FeatureEncoder.student(
            context(
                isClearable = false,
                isOngoing = true,
                isForegroundService = true,
                isCall = true,
                isMedia = true,
                isGroupSummary = true,
                isNoClear = true,
            ),
        )

        assertEquals(FeatureEncoder.FEATURE_DIM, features.size)
        listOf(
            FeatureEncoder.ONGOING_INDEX,
            FeatureEncoder.FOREGROUND_SERVICE_INDEX,
            FeatureEncoder.CALL_INDEX,
            FeatureEncoder.MEDIA_INDEX,
            FeatureEncoder.GROUP_SUMMARY_INDEX,
            FeatureEncoder.NON_CLEARABLE_INDEX,
        ).forEach { index -> assertEquals(1f, features[index], 0f) }
    }

    private fun context(
        isClearable: Boolean = true,
        isOngoing: Boolean = false,
        isForegroundService: Boolean = false,
        isCall: Boolean = false,
        isMedia: Boolean = false,
        isGroupSummary: Boolean = false,
        isNoClear: Boolean = false,
    ) = NotificationContext(
        eventId = "event",
        packageName = "source.app",
        title = "title",
        body = "body",
        category = "promo",
        importance = 0.5f,
        regime = Regime.WEEKDAY,
        hourOfDay = 12f,
        postedAtMillis = 1L,
        isClearable = isClearable,
        isOngoing = isOngoing,
        isForegroundService = isForegroundService,
        isCall = isCall,
        isMedia = isMedia,
        isGroupSummary = isGroupSummary,
        isNoClear = isNoClear,
    )
}
