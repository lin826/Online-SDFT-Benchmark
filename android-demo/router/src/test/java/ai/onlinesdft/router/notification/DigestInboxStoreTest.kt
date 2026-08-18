package ai.onlinesdft.router.notification

import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.Regime
import ai.onlinesdft.router.model.Route
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.security.MessageDigest

class DigestInboxStoreTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `items persist across store recreation`() {
        val file = inboxFile()
        val store = DigestInboxStore(file)
        assertTrue(store.upsert(item("first", routedAtMillis = 10L)))
        assertTrue(store.upsert(item("second", routedAtMillis = 20L)))
        assertTrue(store.markRead("first", atMillis = 30L))

        val restored = DigestInboxStore(file).items()
        assertEquals(listOf("second", "first"), restored.map(DigestInboxItem::eventId))
        assertEquals("open-second", restored.first().openToken)
        assertTrue(restored.first().isUnread)
        assertEquals(30L, restored.last().readAtMillis)
    }

    @Test
    fun `synthetic lab preview persists but can never become a learning callback`() {
        val file = inboxFile()
        val store = DigestInboxStore(file)
        assertTrue(
            store.upsert(
                item(
                    eventId = "lab-session-004",
                    origin = DigestInboxOrigin.SYNTHETIC_LAB,
                ),
            ),
        )

        val restored = DigestInboxStore(file).items().single()
        assertEquals(DigestInboxOrigin.SYNTHETIC_LAB, restored.origin)
        assertNull(restored.learningSnapshot)
        assertNull(restored.toDecision(runEpoch = 0L))
    }

    @Test
    fun `router archive persists the sealed Archive decision for later correction`() {
        val file = inboxFile()
        val context = NotificationContext(
            eventId = "archived",
            packageName = "com.example.mail",
            title = "Newsletter",
            body = "A long-form update the user may want later",
            category = "promo",
            importance = 0.3f,
            regime = Regime.WEEKDAY,
            hourOfDay = 14f,
            postedAtMillis = 1_000L,
        )
        val snapshot = DigestLearningSnapshot(
            context = context,
            foundationModelId = "LiquidAI/LFM2.5-230M@test",
            decidedAtMillis = 1_100L,
            checkpointIndex = 3L,
            runEpoch = 9L,
            studentPrompt = "<|startoftext|>archived prompt",
            foundationProbabilitiesFp64 = doubleArrayOf(0.2, 0.3, 0.5),
            adaptiveDecisionProbabilities = doubleArrayOf(0.1, 0.2, 0.7),
        )
        assertTrue(
            DigestInboxStore(file).upsert(
                item(
                    eventId = context.eventId,
                    sourcePackage = context.packageName,
                    title = context.title,
                    body = context.body,
                    origin = DigestInboxOrigin.ROUTER_ARCHIVE,
                    learningSnapshot = snapshot,
                ),
            ),
        )

        val restored = DigestInboxStore(file).items().single()
        assertEquals(DigestInboxOrigin.ROUTER_ARCHIVE, restored.origin)
        val decision = requireNotNull(restored.toDecision(runEpoch = 9L))
        assertEquals(Route.ARCHIVE, decision.chosenRoute)
        assertEquals(snapshot.studentPrompt, decision.studentPrompt)
        val resealed = decision.toDigestLearningSnapshot()
        assertEquals(9L, resealed.runEpoch)
        assertEquals(snapshot.context, resealed.context)
    }

    @Test
    fun `pre LoRA inbox is rejected instead of becoming a malformed training row`() {
        val file = inboxFile()
        requireNotNull(file.parentFile).mkdirs()
        file.writeBytes(versionFourFixture())

        assertTrue(DigestInboxStore(file).items().isEmpty())
        assertFalse(file.exists())
    }

    @Test
    fun `synthetic preview never evicts a live saved notification`() {
        val store = DigestInboxStore(maxItems = 1)
        assertTrue(store.upsert(item(eventId = "live", routedAtMillis = 10L)))

        assertFalse(
            store.upsert(
                item(
                    eventId = "lab",
                    routedAtMillis = 20L,
                    origin = DigestInboxOrigin.SYNTHETIC_LAB,
                ),
            ),
        )
        assertEquals(listOf("live"), store.items().map(DigestInboxItem::eventId))
    }

    @Test
    fun `live saved notification evicts a synthetic preview before another live item`() {
        val store = DigestInboxStore(maxItems = 2)
        assertTrue(store.upsert(item(eventId = "live-old", routedAtMillis = 10L)))
        assertTrue(
            store.upsert(
                item(
                    eventId = "lab-new",
                    routedAtMillis = 20L,
                    origin = DigestInboxOrigin.SYNTHETIC_LAB,
                ),
            ),
        )

        assertTrue(store.upsert(item(eventId = "live-new", routedAtMillis = 30L)))
        assertEquals(
            listOf("live-new", "live-old"),
            store.items().map(DigestInboxItem::eventId),
        )
    }

    @Test
    fun `open token and exact sealed learner inputs survive restart`() {
        val file = inboxFile()
        val context = NotificationContext(
            eventId = "sealed",
            packageName = "com.example.secure_mail",
            title = "Quarterly results ✓",
            body = "Revenue and forecast details",
            category = "business_update",
            importance = 0.73f,
            regime = Regime.OFF_HOURS,
            hourOfDay = 22.75f,
            postedAtMillis = 1_234_567L,
            caseId = "case-sealed",
            isClearable = true,
            isOngoing = false,
            isForegroundService = true,
            isCall = false,
            isMedia = true,
            isGroupSummary = false,
            isNoClear = true,
            canPublishDigest = false,
        )
        val studentPrompt = "<|startoftext|><|system|>route<|user|>sealed prompt"
        val foundation = doubleArrayOf(0.125, 0.75, 0.125)
        val residualDecision = doubleArrayOf(0.2, 0.7, 0.1)
        val snapshot = DigestLearningSnapshot(
            context = context,
            foundationModelId = "LiquidAI/LFM2.5-230M@test",
            decidedAtMillis = 1_234_999L,
            checkpointIndex = 42L,
            runEpoch = 77L,
            studentPrompt = studentPrompt,
            foundationProbabilitiesFp64 = foundation,
            adaptiveDecisionProbabilities = residualDecision,
        )
        val token = "550e8400-e29b-41d4-a716-446655440000"
        val store = DigestInboxStore(file)
        assertTrue(
            store.upsert(
                item(
                    eventId = context.eventId,
                    sourcePackage = context.packageName,
                    title = context.title,
                    routedAtMillis = snapshot.decidedAtMillis,
                    body = context.body,
                    openToken = token,
                    learningSnapshot = snapshot,
                ),
            ),
        )

        val restoredItem = DigestInboxStore(file).items().single()
        val restored = assertNotNull(restoredItem.learningSnapshot).let {
            requireNotNull(restoredItem.learningSnapshot)
        }
        assertEquals(token, restoredItem.openToken)
        assertEquals(context, restored.context)
        assertEquals(snapshot.foundationModelId, restored.foundationModelId)
        assertEquals(snapshot.decidedAtMillis, restored.decidedAtMillis)
        assertEquals(snapshot.checkpointIndex, restored.checkpointIndex)
        assertEquals(snapshot.runEpoch, restored.runEpoch)
        assertEquals(studentPrompt, restored.studentPrompt)
        assertArrayEquals(foundation, restored.foundationProbabilitiesFp64, 0.0)
        assertArrayEquals(residualDecision, restored.adaptiveDecisionProbabilities, 0.0)

        val decision = requireNotNull(restoredItem.toDecision(runEpoch = 77L))
        assertEquals(context, decision.context)
        assertEquals(Route.LATER, decision.chosenRoute)
        assertEquals(snapshot.foundationModelId, decision.foundationModelId)
        assertEquals(snapshot.decidedAtMillis, decision.decidedAtMillis)
        assertEquals(snapshot.checkpointIndex, decision.checkpointIndex)
        assertEquals(77L, decision.runEpoch)
        assertEquals(studentPrompt, decision.studentPrompt)
        assertArrayEquals(
            foundation,
            decision.foundationProbabilitiesFp64,
            0.0,
        )
        assertArrayEquals(
            residualDecision,
            decision.adaptiveDecisionProbabilities,
            0.0,
        )
        val resealed = decision.toDigestLearningSnapshot()
        assertEquals(context, resealed.context)
        assertEquals(77L, resealed.runEpoch)
        assertEquals(studentPrompt, resealed.studentPrompt)
        assertArrayEquals(foundation, resealed.foundationProbabilitiesFp64, 0.0)
        assertArrayEquals(residualDecision, resealed.adaptiveDecisionProbabilities, 0.0)
    }

    @Test
    fun `item from an old run stays visible but cannot reconstruct in a new epoch`() {
        val file = inboxFile()
        val eventId = "old-run"
        val sourcePackage = "ai.publisher.mail"
        val oldEpoch = 4L
        val snapshot = DigestLearningSnapshot(
            context = NotificationContext(
                eventId = eventId,
                packageName = sourcePackage,
                title = "Saved before reset",
                body = "Still useful to read",
                category = "mail",
                importance = 0.5f,
                regime = Regime.WEEKDAY,
                hourOfDay = 9f,
                postedAtMillis = 100L,
            ),
            foundationModelId = "LiquidAI/LFM2.5-230M@test",
            decidedAtMillis = 110L,
            checkpointIndex = 2L,
            runEpoch = oldEpoch,
            studentPrompt = "<|startoftext|>old run prompt",
            foundationProbabilitiesFp64 = doubleArrayOf(0.2, 0.7, 0.1),
            adaptiveDecisionProbabilities = doubleArrayOf(0.1, 0.8, 0.1),
        )
        assertTrue(
            DigestInboxStore(file).upsert(
                item(
                    eventId = eventId,
                    sourcePackage = sourcePackage,
                    learningSnapshot = snapshot,
                ),
            ),
        )

        val restored = DigestInboxStore(file).items().single()
        assertEquals(eventId, restored.eventId)
        assertEquals(oldEpoch, restored.learningSnapshot?.runEpoch)
        assertNull(restored.toDecision(runEpoch = oldEpoch + 1L))
        assertNotNull(restored.toDecision(runEpoch = oldEpoch))
    }

    @Test
    fun `upsert is idempotent by event id`() {
        val store = DigestInboxStore()
        assertTrue(store.upsert(item("same", title = "Original")))
        assertFalse(store.upsert(item("same", title = "Replacement", routedAtMillis = 99L)))

        assertEquals(1, store.items().size)
        assertEquals("Original", store.items().single().title)
    }

    @Test
    fun `terminal action claim survives restart and rejects its opposite`() {
        val file = inboxFile()
        val token = "550e8400-e29b-41d4-a716-446655440001"
        val store = DigestInboxStore(file)
        assertTrue(store.upsert(item("claimed", openToken = token)))

        val first = requireNotNull(
            store.claimAction(
                eventId = "claimed",
                openToken = token,
                action = DigestPendingAction.OPEN,
                atMillis = 100L,
            ),
        )
        assertEquals(DigestPendingAction.OPEN, first.pendingAction)
        assertEquals(100L, first.pendingActionAtMillis)

        val restored = DigestInboxStore(file)
        val retry = requireNotNull(
            restored.claimAction(
                eventId = "claimed",
                openToken = token,
                action = DigestPendingAction.OPEN,
                atMillis = 999L,
            ),
        )
        assertEquals(DigestPendingAction.OPEN, retry.pendingAction)
        assertEquals(100L, retry.pendingActionAtMillis)
        assertNull(
            restored.claimAction(
                eventId = "claimed",
                openToken = token,
                action = DigestPendingAction.REMOVE,
                atMillis = 101L,
            ),
        )
        assertNull(
            restored.claimAction(
                eventId = "claimed",
                openToken = "wrong-token",
                action = DigestPendingAction.OPEN,
                atMillis = 101L,
            ),
        )
    }

    @Test
    fun `mark read completes and clears a durable open claim`() {
        val file = inboxFile()
        val token = "open-claimed-read"
        val store = DigestInboxStore(file)
        assertTrue(store.upsert(item("claimed-read", openToken = token)))
        assertNotNull(
            store.claimAction(
                eventId = "claimed-read",
                openToken = token,
                action = DigestPendingAction.OPEN,
                atMillis = 20L,
            ),
        )

        assertTrue(store.markRead("claimed-read", atMillis = 21L))
        val restored = DigestInboxStore(file).items().single()
        assertEquals(21L, restored.readAtMillis)
        assertNull(restored.pendingAction)
        assertNull(restored.pendingActionAtMillis)
    }

    @Test
    fun `remove deletes an item with a durable remove claim`() {
        val file = inboxFile()
        val token = "open-claimed-remove"
        val store = DigestInboxStore(file)
        assertTrue(store.upsert(item("claimed-remove", openToken = token)))
        assertNotNull(
            store.claimAction(
                eventId = "claimed-remove",
                openToken = token,
                action = DigestPendingAction.REMOVE,
                atMillis = 30L,
            ),
        )

        assertTrue(store.remove("claimed-remove"))
        assertTrue(DigestInboxStore(file).items().isEmpty())
    }

    @Test
    fun `ui-only stale delivery cannot be reconstructed for learning`() {
        val stale = item("stale-after-reset", learningSnapshot = null)

        assertNull(stale.toDecision(runEpoch = 99L))
    }

    @Test
    fun `items are newest first and retention keeps newest events`() {
        val store = DigestInboxStore(maxItems = 3)
        assertTrue(store.upsert(item("oldest", routedAtMillis = 10L)))
        assertTrue(store.upsert(item("newest", routedAtMillis = 40L)))
        assertTrue(store.upsert(item("middle-new", routedAtMillis = 30L)))
        assertTrue(store.upsert(item("middle-old", routedAtMillis = 20L)))

        assertEquals(
            listOf("newest", "middle-new", "middle-old"),
            store.items().map(DigestInboxItem::eventId),
        )
    }

    @Test
    fun `older confirmed incoming item is retained when inbox is full`() {
        val store = DigestInboxStore(maxItems = 2)
        assertTrue(store.upsert(item("newest-existing", routedAtMillis = 300L)))
        assertTrue(store.upsert(item("oldest-existing", routedAtMillis = 200L)))

        assertTrue(store.upsert(item("older-incoming", routedAtMillis = 100L)))
        assertEquals(
            listOf("newest-existing", "older-incoming"),
            store.items().map(DigestInboxItem::eventId),
        )
    }

    @Test
    fun `mark read transitions once and preserves the first read time`() {
        val store = DigestInboxStore()
        assertTrue(store.upsert(item("read-me")))
        assertFalse(store.markRead("missing", atMillis = 20L))
        assertTrue(store.markRead("read-me", atMillis = 21L))
        assertFalse(store.markRead("read-me", atMillis = 22L))

        assertFalse(store.items().single().isUnread)
        assertEquals(21L, store.items().single().readAtMillis)
    }

    @Test
    fun `remove and clear remain cleared after restart`() {
        val file = inboxFile()
        val store = DigestInboxStore(file)
        assertTrue(store.upsert(item("one", routedAtMillis = 1L)))
        assertTrue(store.upsert(item("two", routedAtMillis = 2L)))
        assertFalse(store.remove("missing"))
        assertTrue(store.remove("one"))
        assertFalse(store.remove("one"))
        assertEquals(listOf("two"), DigestInboxStore(file).items().map { it.eventId })

        assertTrue(store.clear())
        assertTrue(store.items().isEmpty())
        assertTrue(DigestInboxStore(file).items().isEmpty())
    }

    @Test
    fun `clear forgets memory when durable purge cannot be guaranteed`() {
        val file = inboxFile()
        val store = DigestInboxStore(file)
        assertTrue(store.upsert(item("sensitive")))
        assertTrue(file.delete())
        assertTrue(file.mkdirs())
        File(file, "undeletable-child").writeText("blocks replacement and directory deletion")

        assertFalse(store.clear())
        assertTrue(store.items().isEmpty())
        assertTrue(file.exists())
    }

    @Test
    fun `tampered durable state fails closed and is deleted`() {
        val file = inboxFile()
        val store = DigestInboxStore(file)
        assertTrue(store.upsert(item("tampered")))
        val bytes = file.readBytes()
        bytes[bytes.lastIndex] = (bytes.last().toInt() xor 0x01).toByte()
        file.writeBytes(bytes)

        assertTrue(DigestInboxStore(file).items().isEmpty())
        assertFalse(file.exists())
    }

    @Test
    fun `summary is deterministic bounded and uses friendly source labels`() {
        val store = DigestInboxStore()
        assertTrue(store.upsert(item("mail-1", "ai.publisher.mail", "Weekly report", 10L)))
        assertTrue(store.upsert(item("calendar", "ai.publisher.calendar", "Team sync", 20L)))
        assertTrue(store.upsert(item("chat", "ai.publisher.team_chat", "Alex replied", 30L)))
        assertTrue(store.upsert(item("security", "ai.publisher.security-alerts", "Sign-in", 40L)))
        assertTrue(store.upsert(item("mail-2", "ai.publisher.mail", "Budget", 50L)))
        assertTrue(store.markRead("calendar", atMillis = 60L))

        assertEquals(
            DigestInboxSummary(
                total = 5,
                unread = 4,
                sourceCounts = listOf("Mail" to 2, "Calendar" to 1, "Security Alerts" to 1),
                previewTitles = listOf("Budget", "Sign-in", "Alex replied"),
            ),
            store.summary(),
        )
    }

    @Test
    fun `invalid and oversized records are rejected`() {
        val store = DigestInboxStore()
        assertThrows(IllegalArgumentException::class.java) {
            store.upsert(item(eventId = ""))
        }
        assertThrows(IllegalArgumentException::class.java) {
            store.upsert(item(eventId = "large", body = "x".repeat(4_097)))
        }
        assertThrows(IllegalArgumentException::class.java) {
            store.upsert(item(eventId = "token", openToken = " "))
        }
        assertThrows(IllegalArgumentException::class.java) {
            store.upsert(item(eventId = "token", openToken = "x".repeat(129)))
        }
        assertThrows(IllegalArgumentException::class.java) {
            DigestInboxStore(maxItems = 0)
        }
    }

    private fun inboxFile(): File = File(temporaryFolder.root, "digest/inbox-v1.bin")

    private fun versionFourFixture(): ByteArray {
        fun DataOutputStream.writeString(value: String) {
            val encoded = value.toByteArray(Charsets.UTF_8)
            writeInt(encoded.size)
            write(encoded)
        }
        val payload = ByteArrayOutputStream().use { bytes ->
            DataOutputStream(bytes).use { output ->
                output.writeInt(1)
                output.writeString("legacy-live")
                output.writeString("legacy-token")
                output.writeString("ai.publisher.mail")
                output.writeString("Legacy title")
                output.writeString("Legacy body")
                output.writeLong(42L)
                output.writeBoolean(false) // readAtMillis
                output.writeBoolean(false) // pending action
                output.writeBoolean(false) // learning snapshot
            }
            bytes.toByteArray()
        }
        val digest = MessageDigest.getInstance("SHA-256").digest(payload)
        return ByteArrayOutputStream().use { bytes ->
            DataOutputStream(bytes).use { output ->
                output.writeInt(0x44494749)
                output.writeInt(4)
                output.writeInt(payload.size)
                output.write(payload)
                output.write(digest)
            }
            bytes.toByteArray()
        }
    }

    private fun item(
        eventId: String,
        sourcePackage: String = "ai.publisher.mail",
        title: String = "Title $eventId",
        routedAtMillis: Long = 10L,
        body: String = "Body $eventId",
        openToken: String = "open-$eventId",
        origin: DigestInboxOrigin = DigestInboxOrigin.LIVE_NOTIFICATION,
        learningSnapshot: DigestLearningSnapshot? = null,
    ) = DigestInboxItem(
        eventId = eventId,
        openToken = openToken,
        sourcePackage = sourcePackage,
        title = title,
        body = body,
        routedAtMillis = routedAtMillis,
        origin = origin,
        learningSnapshot = learningSnapshot,
    )
}
