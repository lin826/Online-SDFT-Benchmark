package ai.onlinesdft.router.model

import java.io.File
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class LoraReplayStoreTest {
    @get:Rule val temporary = TemporaryFolder()

    @Test
    fun `replay prompt target rng and adapter identity survive restart`() {
        val file = File(temporary.root, "model/replay.bin")
        val store = LoraReplayStore(file)
        val value = checkpoint()

        val receipt = store.save(value)
        val restored = requireNotNull(LoraReplayStore(file).load())

        assertEquals(1L, receipt.generation)
        assertEquals(1L, restored.generation)
        assertEquals(7L, restored.adapterUpdateIndex)
        assertEquals("adapter-checksum", restored.adapterChecksum)
        assertEquals("sealed tokenized prompt", restored.replay.single().prompt)
        assertArrayEquals(doubleArrayOf(0.1, 0.8, 0.1), restored.replay.single().target, 0.0)
        assertEquals(listOf("feedback-1"), restored.processedFeedbackFingerprints)
    }

    @Test
    fun `tampered newest slot falls back to previous committed replay`() {
        val file = File(temporary.root, "model/replay.bin")
        val store = LoraReplayStore(file)
        store.save(checkpoint(adapterUpdateIndex = 1L))
        store.save(checkpoint(adapterUpdateIndex = 2L))
        val newest = File("${file.path}.0")
        val bytes = newest.readBytes()
        bytes[bytes.lastIndex] = (bytes.last().toInt() xor 1).toByte()
        newest.writeBytes(bytes)

        assertEquals(1L, LoraReplayStore(file).load()?.adapterUpdateIndex)
    }

    @Test
    fun `clear removes both durable slots`() {
        val file = File(temporary.root, "model/replay.bin")
        val store = LoraReplayStore(file)
        store.save(checkpoint())
        store.clear()
        assertNull(store.load())
        assertTrue(listOf(File("${file.path}.0"), File("${file.path}.1")).none(File::exists))
    }

    private fun checkpoint(adapterUpdateIndex: Long = 7L) = LoraReplayCheckpoint(
        generation = 0L,
        modelId = LiquidOrtFoundationModel.MODEL_ID,
        adapterUpdateIndex = adapterUpdateIndex,
        adapterChecksum = "adapter-checksum",
        replay = listOf(
            LoraTrainingExample(
                eventId = "event-1",
                prompt = "sealed tokenized prompt",
                target = doubleArrayOf(0.1, 0.8, 0.1),
                replayLabel = "LATER",
            ),
        ),
        replayRng = NumpyReplayRng.canonicalSeed57().snapshot(),
        processedFeedbackFingerprints = listOf("feedback-1"),
    )
}
