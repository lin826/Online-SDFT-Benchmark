package ai.onlinesdft.router.model

import java.math.BigInteger

data class NumpyPcg64Snapshot(
    val state: String,
    val increment: String,
)

/** Minimal NumPy-2.4.6 PCG64 + weighted choice port for replay parity. */
class NumpyReplayRng private constructor(
    state: BigInteger,
    private val increment: BigInteger,
) {
    private var state = state.and(MASK_128)

    init {
        require(increment.signum() >= 0 && increment.bitLength() <= 128)
        require(increment.testBit(0))
    }

    @Synchronized
    fun weightedChoiceWithoutReplacement(
        weights: DoubleArray,
        sampleSize: Int,
    ): IntArray {
        require(sampleSize in 0..weights.size)
        weights.forEach { require(it.isFinite() && it >= 0.0) }
        if (sampleSize == 0) return intArrayOf()
        require(weights.count { it > 0.0 } >= sampleSize)
        val probabilities = weights.copyOf()
        normalize(probabilities)
        val selected = ArrayList<Int>(sampleSize)
        while (selected.size < sampleSize) {
            selected.forEach { probabilities[it] = 0.0 }
            normalize(probabilities)
            val draws = DoubleArray(sampleSize - selected.size) { nextDoubleLocked() }
            val cumulative = DoubleArray(probabilities.size)
            var running = 0.0
            for (index in probabilities.indices) {
                running += probabilities[index]
                cumulative[index] = running
            }
            cumulative[cumulative.lastIndex] = 1.0
            val newIndices = draws.map { draw -> searchRight(cumulative, draw) }
            newIndices.forEach { index ->
                if (index !in selected) selected += index
            }
        }
        return selected.toIntArray()
    }

    @Synchronized
    fun snapshot(): NumpyPcg64Snapshot = NumpyPcg64Snapshot(
        state = state.toString(),
        increment = increment.toString(),
    )

    @Synchronized
    fun restore(snapshot: NumpyPcg64Snapshot) {
        val restoredIncrement = BigInteger(snapshot.increment)
        require(restoredIncrement == increment) { "PCG64 stream changed" }
        val restoredState = BigInteger(snapshot.state)
        require(restoredState.signum() >= 0 && restoredState.bitLength() <= 128)
        state = restoredState
    }

    @Synchronized
    internal fun nextRawForTest(): ULong = nextRawLocked().toString().toULong()

    private fun nextDoubleLocked(): Double {
        val top53 = nextRawLocked().shiftRight(11).toLong()
        return top53 * TWO_POW_NEGATIVE_53
    }

    /** NumPy PCG64 advances first, then applies XSL-RR to the new state. */
    private fun nextRawLocked(): BigInteger {
        state = state.multiply(MULTIPLIER).add(increment).and(MASK_128)
        val high = state.shiftRight(64)
        val low = state.and(MASK_64)
        val xorshifted = high.xor(low).and(MASK_64)
        val rotation = state.shiftRight(122).toInt()
        if (rotation == 0) return xorshifted
        return xorshifted.shiftRight(rotation)
            .or(xorshifted.shiftLeft(64 - rotation))
            .and(MASK_64)
    }

    private fun normalize(values: DoubleArray) {
        val total = values.sum()
        require(total.isFinite() && total > 0.0)
        for (index in values.indices) values[index] /= total
    }

    private fun searchRight(cumulative: DoubleArray, value: Double): Int {
        var low = 0
        var high = cumulative.size
        while (low < high) {
            val middle = (low + high) ushr 1
            if (value < cumulative[middle]) high = middle else low = middle + 1
        }
        return low.coerceAtMost(cumulative.lastIndex)
    }

    companion object {
        private val MASK_128 = BigInteger.ONE.shiftLeft(128).subtract(BigInteger.ONE)
        private val MASK_64 = BigInteger.ONE.shiftLeft(64).subtract(BigInteger.ONE)
        private val MULTIPLIER = BigInteger("2360ed051fc65da44385df649fccf645", 16)
        private const val TWO_POW_NEGATIVE_53 = 1.0 / 9_007_199_254_740_992.0
        private const val SEED_57_STATE = "53849766128814384326592283228154713017"
        private const val SEED_57_INCREMENT = "103383364936357617232830043403980227967"

        fun canonicalSeed57(): NumpyReplayRng = NumpyReplayRng(
            BigInteger(SEED_57_STATE),
            BigInteger(SEED_57_INCREMENT),
        )

        fun fromSnapshot(snapshot: NumpyPcg64Snapshot): NumpyReplayRng =
            NumpyReplayRng(BigInteger(snapshot.state), BigInteger(snapshot.increment))
    }
}
