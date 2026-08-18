package ai.onlinesdft.router.lfm

import ai.onlinesdft.router.model.FactualFeedback
import ai.onlinesdft.router.model.FeedbackSource
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.Outcome
import ai.onlinesdft.router.model.Regime
import ai.onlinesdft.router.model.Route
import org.junit.Assert.assertEquals
import org.junit.Test

class LfmNotificationTextTest {
    @Test
    fun `Android serving view matches Python environment prose`() {
        val context = NotificationContext(
            eventId = "event",
            packageName = "example.app",
            title = "Checkout errors",
            body = "Error rate is elevated.",
            category = "monitoring",
            importance = 0.914f,
            regime = Regime.ON_CALL,
            hourOfDay = 1.5f,
            postedAtMillis = 1L,
        )

        assertEquals(
            "The notification title is Checkout errors. " +
                "The message says Error rate is elevated. " +
                "This is a monitoring notification that arrived at 01:30 local time " +
                "during the on-call period. Its on-device importance score is 0.91 out of 1.",
            context.toLfmStudentContext(),
        )
    }

    @Test
    fun `serving projection floors minutes and uses semantic-v2 commerce category`() {
        val context = NotificationContext(
            eventId = "receipt",
            packageName = "example.mail",
            title = "Trip receipt",
            body = "Your receipt is ready.",
            category = "receipt",
            importance = 0.31f,
            regime = Regime.WEEKDAY,
            hourOfDay = 23.999f,
            postedAtMillis = 1L,
        )

        assertEquals(
            "The notification title is Trip receipt. " +
                "The message says Your receipt is ready. " +
                "This is a commerce notification that arrived at 23:59 local time " +
                "during the weekday period. Its on-device importance score is 0.31 out of 1.",
            context.toLfmStudentContext(),
        )
    }

    @Test
    fun `teacher evidence contains only executed factual trajectory`() {
        val feedback = FactualFeedback(
            eventId = "event",
            executedRoute = Route.LATER,
            outcome = Outcome.OPENED_DIGEST,
            observedSelection = Route.LATER,
            delayMinutes = 120,
            source = FeedbackSource.DIGEST_CALLBACK,
            observedAtMillis = 2L,
        )

        assertEquals(
            "The router placed the notification in a later digest. " +
                "The user opened it from the digest 120 minutes later. " +
                "This behavior revealed LATER as the observed user selection on the executed surface.",
            feedback.toLfmTeacherEvidence(),
        )
    }
}
