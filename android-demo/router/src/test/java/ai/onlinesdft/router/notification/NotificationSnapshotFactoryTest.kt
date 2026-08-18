package ai.onlinesdft.router.notification

import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.Regime
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NotificationSnapshotFactoryTest {
    @Test
    fun `listener reconnect ignores backlog except durable pending cancellations`() {
        assertEquals(
            setOf("ordinary", "protected"),
            backlogKeysToIgnore(
                activeKeys = setOf("ordinary", "protected", "pending"),
                pendingCancellationKeys = setOf("pending"),
            ),
        )
    }

    @Test
    fun `reused publisher id becomes a bounded collision-proof decision id`() {
        val first = context("shared-id", "ai.onlinesdft.publisher.mail", 10L, "Package")
        assertEquals(
            first,
            NotificationSnapshotFactory.disambiguateEventId(first, emptySet()),
        )

        val second = context("shared-id", "ai.onlinesdft.publisher.calendar", 20L, "Review")
        val unique = NotificationSnapshotFactory.disambiguateEventId(
            second,
            setOf("shared-id"),
        )

        assertNotEquals("shared-id", unique.eventId)
        assertTrue(unique.eventId.startsWith("shared-id-"))
        assertTrue(unique.eventId.length <= 200)
        assertEquals("shared-id", unique.caseId)
    }

    @Test
    fun `a hash collision receives a deterministic numeric suffix`() {
        val duplicate = context("same", "publisher", 7L, "Same")
        val once = NotificationSnapshotFactory.disambiguateEventId(duplicate, setOf("same"))
        val twice = NotificationSnapshotFactory.disambiguateEventId(
            duplicate,
            setOf("same", once.eventId),
        )
        assertNotEquals(once.eventId, twice.eventId)
        assertTrue(twice.eventId.endsWith("-2"))
    }

    @Test
    fun `only exact trusted publisher timeout metadata activates accelerated horizon`() {
        val trusted = context(
            "timeout",
            "ai.onlinesdft.publisher.mail",
            7L,
            "Receipt",
        ).copy(
            demoTimeoutMillis = 2_000L,
            semanticDelayMinutes = 120,
        )

        assertTrue(NotificationSnapshotFactory.isTrustedDemoTimeout(trusted))
        assertFalse(
            NotificationSnapshotFactory.isTrustedDemoTimeout(
                trusted.copy(packageName = "com.example.untrusted"),
            ),
        )
        assertFalse(
            NotificationSnapshotFactory.isTrustedDemoTimeout(
                trusted.copy(demoTimeoutMillis = 2_001L),
            ),
        )
        assertFalse(
            NotificationSnapshotFactory.isTrustedDemoTimeout(
                trusted.copy(semanticDelayMinutes = 0),
            ),
        )
    }

    private fun context(
        eventId: String,
        packageName: String,
        postedAtMillis: Long,
        title: String,
    ) = NotificationContext(
        eventId = eventId,
        packageName = packageName,
        title = title,
        body = "Body",
        category = "promo",
        importance = 0.5f,
        regime = Regime.WEEKDAY,
        hourOfDay = 9f,
        postedAtMillis = postedAtMillis,
    )
}
