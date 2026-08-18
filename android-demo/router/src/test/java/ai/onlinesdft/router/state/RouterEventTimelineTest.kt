package ai.onlinesdft.router.state

import ai.onlinesdft.router.model.DecisionSnapshot
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.Regime
import ai.onlinesdft.router.model.Route
import ai.onlinesdft.router.notification.DigestInboxOrigin
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Test

class RouterEventTimelineTest {
    @Test
    fun `only an actually chosen Later lab row becomes a non learnable Saved preview`() {
        val preview = requireNotNull(
            syntheticLabSavedPreview(
                decision = decision("lab-session-004", Route.LATER),
                openToken = "preview-token",
            ),
        )

        assertEquals(DigestInboxOrigin.SYNTHETIC_LAB, preview.origin)
        assertEquals("preview-token", preview.openToken)
        assertNull(preview.learningSnapshot)
        assertNull(
            syntheticLabSavedPreview(
                decision = decision("lab-session-005", Route.ARCHIVE),
                openToken = "unused",
            ),
        )
    }

    @Test
    fun `events project oldest first and new events append at bottom`() {
        val events = linkedMapOf(
            "first" to "oldest",
            "second" to "middle",
        )

        assertEquals(listOf("oldest", "middle"), chronologicalEventValues(events))

        events["third"] = "newest"

        assertEquals(
            listOf("oldest", "middle", "newest"),
            chronologicalEventValues(events),
        )
    }

    @Test
    fun `updating an existing event preserves its timeline position`() {
        val events = linkedMapOf(
            "first" to "waiting",
            "second" to "waiting",
        )

        events["first"] = "trained"

        assertEquals(listOf("trained", "waiting"), chronologicalEventValues(events))
    }

    @Test
    fun `each card freezes its own post update distribution and later updates do not change it`() {
        val firstSnapshot = floatArrayOf(0.49f, 0.38f, 0.13f)
        val first = RoutedEventUi(
            decision = decision("first"),
            postTrainingProbabilities = firstSnapshot.copyOf(),
            distributionUpdateIndex = 2L,
        )
        val second = RoutedEventUi(decision("second"))
        val third = RoutedEventUi(decision("third"))
        val events = linkedMapOf("first" to first, "second" to second, "third" to third)
        val reevaluatedIds = mutableListOf<String>()

        val afterSecond = capturedEventDistribution(
            events = events,
            eventId = "second",
            updateIndex = 4L,
        ) { decision ->
            reevaluatedIds += decision.context.eventId
            floatArrayOf(0.41f, 0.33f, 0.26f)
        }
        val afterThird = capturedEventDistribution(
            events = afterSecond,
            eventId = "third",
            updateIndex = 6L,
        ) { decision ->
            reevaluatedIds += decision.context.eventId
            floatArrayOf(0.35f, 0.28f, 0.37f)
        }

        assertEquals(listOf("first", "second", "third"), afterThird.keys.toList())
        assertEquals(listOf("second", "third"), reevaluatedIds)
        assertArrayEquals(
            firstSnapshot,
            afterThird.getValue("first").postTrainingProbabilities,
            0f,
        )
        assertEquals(2L, afterThird.getValue("first").distributionUpdateIndex)
        assertArrayEquals(
            floatArrayOf(0.41f, 0.33f, 0.26f),
            afterThird.getValue("second").postTrainingProbabilities,
            0f,
        )
        assertEquals(4L, afterThird.getValue("second").distributionUpdateIndex)
        assertArrayEquals(
            floatArrayOf(0.35f, 0.28f, 0.37f),
            afterThird.getValue("third").postTrainingProbabilities,
            0f,
        )
        assertEquals(6L, afterThird.getValue("third").distributionUpdateIndex)
        assertSame(first, afterSecond.getValue("first"))
        assertSame(second, events.getValue("second"))
        assertNull(events.getValue("second").postTrainingProbabilities)
    }

    @Test
    fun `buffered or unavailable event remains unchanged`() {
        val event = RoutedEventUi(decision("buffered"))
        val events = linkedMapOf("buffered" to event)

        val unavailable = capturedEventDistribution(
            events = events,
            eventId = "buffered",
            updateIndex = 2L,
        ) { null }
        val missing = capturedEventDistribution(
            events = unavailable,
            eventId = "not-present",
            updateIndex = 4L,
        ) { error("missing events must not be evaluated") }

        assertSame(event, unavailable.getValue("buffered"))
        assertSame(event, missing.getValue("buffered"))
        assertNull(missing.getValue("buffered").postTrainingProbabilities)
    }

    @Test
    fun `manual correction replaces only that cards frozen snapshot`() {
        val first = RoutedEventUi(
            decision = decision("first"),
            postTrainingProbabilities = floatArrayOf(0.55f, 0.39f, 0.06f),
            distributionUpdateIndex = 2L,
        )
        val secondSnapshot = floatArrayOf(0.41f, 0.33f, 0.26f)
        val second = RoutedEventUi(
            decision = decision("second"),
            postTrainingProbabilities = secondSnapshot.copyOf(),
            distributionUpdateIndex = 4L,
        )

        val replaced = capturedEventDistribution(
            events = linkedMapOf("first" to first, "second" to second),
            eventId = "first",
            updateIndex = 10L,
            replaceExisting = true,
        ) {
            floatArrayOf(0.24f, 0.31f, 0.45f)
        }

        assertArrayEquals(
            floatArrayOf(0.24f, 0.31f, 0.45f),
            replaced.getValue("first").postTrainingProbabilities,
            0f,
        )
        assertEquals(10L, replaced.getValue("first").distributionUpdateIndex)
        assertArrayEquals(
            secondSnapshot,
            replaced.getValue("second").postTrainingProbabilities,
            0f,
        )
        assertEquals(4L, replaced.getValue("second").distributionUpdateIndex)
    }

    private fun decision(
        eventId: String,
        chosenRoute: Route = Route.INTERRUPT,
    ) = DecisionSnapshot(
        context = NotificationContext(
            eventId = eventId,
            packageName = "demo.source",
            title = "Notification $eventId",
            body = "Body",
            category = "teammate",
            importance = 0.6f,
            regime = Regime.WEEKDAY,
            hourOfDay = 9f,
            postedAtMillis = 1L,
        ),
        studentFeatures = floatArrayOf(),
        probabilities = floatArrayOf(0.312f, 0.688f, 0f),
        baseProbabilities = floatArrayOf(0.312f, 0.688f, 0f),
        chosenRoute = chosenRoute,
        recommendedRoute = chosenRoute,
        baseRoute = Route.INTERRUPT,
        checkpointIndex = 0L,
        adapterChecksum = "sealed",
        decidedAtMillis = 1L,
        inferenceLatencyMillis = 1.0,
    )
}
