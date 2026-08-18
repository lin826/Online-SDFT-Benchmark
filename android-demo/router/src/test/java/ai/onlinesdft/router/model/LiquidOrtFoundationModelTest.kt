package ai.onlinesdft.router.model

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LiquidOrtFoundationModelTest {
    @Test
    fun `deployment contract pins model tokenizer and action rows`() {
        assertEquals("LiquidAI/LFM2.5-230M", LiquidOrtFoundationModel.MODEL_ID)
        assertEquals(
            "13a53837c4906b4f7405932532ba85d182bb013b",
            LiquidOrtFoundationModel.MODEL_REVISION,
        )
        assertEquals("fp32", LiquidOrtFoundationModel.PRECISION)
        assertArrayEquals(longArrayOf(542, 543, 544), LiquidOrtFoundationModel.ACTION_TOKEN_IDS)
        assertEquals("ai.onlinesdft.lfm_lora_ort_bundle", LiquidOrtFoundationModel.MANIFEST_SCHEMA)
        assertEquals(172_032, LoraAdapterStatus.EXPECTED_PARAMETERS)
        assertEquals(48, LoraAdapterStatus.EXPECTED_TENSORS)
    }

    @Test
    fun `stable action softmax returns finite normalized ABC probabilities`() {
        val probabilities = frozenActionSoftmax(floatArrayOf(1_000f, 999f, -1_000f))

        assertTrue(probabilities.all(Float::isFinite))
        assertEquals(1f, probabilities.sum(), 1e-6f)
        assertTrue(probabilities[0] > probabilities[1])
        assertEquals(0f, probabilities[2], 0f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `action softmax rejects full vocabulary output`() {
        frozenActionSoftmax(FloatArray(65_536))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `action softmax rejects non finite output`() {
        frozenActionSoftmax(floatArrayOf(0f, Float.NaN, 1f))
    }

    @Test
    fun `prompt token budget leaves short prompts unchanged`() {
        assertArrayEquals(
            longArrayOf(1, 20, 21, 22),
            fitPromptTokenBudget(
                encoded = intArrayOf(1, 20, 21, 22),
                marker = intArrayOf(90, 91),
                maxSequenceLength = 8,
            ),
        )
    }

    @Test
    fun `prompt token budget preserves title prefix and route suffix`() {
        val encoded = IntArray(20) { it + 1 }

        val fitted = fitPromptTokenBudget(
            encoded = encoded,
            marker = intArrayOf(90, 91),
            maxSequenceLength = 12,
        )

        assertArrayEquals(
            longArrayOf(1, 2, 3, 4, 5, 6, 90, 91, 17, 18, 19, 20),
            fitted,
        )
        assertEquals(1L, fitted.first())
        assertEquals(encoded.last().toLong(), fitted.last())
    }
}
