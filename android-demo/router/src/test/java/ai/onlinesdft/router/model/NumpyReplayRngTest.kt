package ai.onlinesdft.router.model

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class NumpyReplayRngTest {
    @Test
    fun `seed 57 raw stream matches NumPy 2 4 6 PCG64`() {
        val rng = NumpyReplayRng.canonicalSeed57()
        val expected = arrayOf(
            12_620_648_076_863_891_989uL,
            8_638_831_024_364_687_375uL,
            15_518_022_212_100_399_819uL,
            6_277_792_398_783_293_611uL,
            7_264_726_011_671_283_770uL,
            2_885_845_252_504_734_324uL,
            7_917_677_138_843_397_617uL,
            13_398_597_681_927_043_336uL,
            5_365_670_219_334_981_681uL,
            6_217_982_342_897_412_676uL,
        )

        expected.forEach { value -> assertEquals(value, rng.nextRawForTest()) }
    }

    @Test
    fun `weighted choice and persisted state match NumPy without replacement`() {
        val weights = doubleArrayOf(
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
            0.00390625,
        )
        val rng = NumpyReplayRng.canonicalSeed57()

        assertArrayEquals(
            intArrayOf(1, 0, 2, 3, 5),
            rng.weightedChoiceWithoutReplacement(weights, 5),
        )
        assertEquals(
            "336308513773973887114601364716314102817",
            rng.snapshot().state,
        )
        assertArrayEquals(
            intArrayOf(0, 4, 5, 2, 1),
            rng.weightedChoiceWithoutReplacement(weights, 5),
        )
        assertEquals(
            "71569012008654870697562576287528510316",
            rng.snapshot().state,
        )
    }

    @Test
    fun `restored snapshot continues the exact same draw stream`() {
        val live = NumpyReplayRng.canonicalSeed57()
        live.weightedChoiceWithoutReplacement(doubleArrayOf(1.0, 2.0, 3.0, 4.0), 3)
        val restored = NumpyReplayRng.fromSnapshot(live.snapshot())

        repeat(20) {
            assertEquals(live.nextRawForTest(), restored.nextRawForTest())
        }
    }
}
