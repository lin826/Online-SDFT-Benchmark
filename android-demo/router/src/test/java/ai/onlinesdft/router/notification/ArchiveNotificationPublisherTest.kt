package ai.onlinesdft.router.notification

import org.junit.Assert.assertEquals
import org.junit.Test

class ArchiveNotificationPublisherTest {
    @Test
    fun `shade group contains only unread router archives in inbox order`() {
        val unreadArchive = item("archive-unread", DigestInboxOrigin.ROUTER_ARCHIVE)
        val savedForLater = item("later", DigestInboxOrigin.LIVE_NOTIFICATION)
        val readArchive = item(
            eventId = "archive-read",
            origin = DigestInboxOrigin.ROUTER_ARCHIVE,
            readAtMillis = 100L,
        )
        val anotherUnreadArchive = item("archive-another", DigestInboxOrigin.ROUTER_ARCHIVE)

        val grouped = archiveGroupItems(
            listOf(unreadArchive, savedForLater, readArchive, anotherUnreadArchive),
        )

        assertEquals(
            listOf("archive-unread", "archive-another"),
            grouped.map(DigestInboxItem::eventId),
        )
    }

    private fun item(
        eventId: String,
        origin: DigestInboxOrigin,
        readAtMillis: Long? = null,
    ) = DigestInboxItem(
        eventId = eventId,
        openToken = "open-$eventId",
        sourcePackage = "ai.publisher.mail",
        title = "Title $eventId",
        body = "Body $eventId",
        routedAtMillis = 10L,
        origin = origin,
        readAtMillis = readAtMillis,
    )
}
