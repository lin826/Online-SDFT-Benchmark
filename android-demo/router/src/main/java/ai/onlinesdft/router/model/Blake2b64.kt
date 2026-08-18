package ai.onlinesdft.router.model

import java.lang.Long.rotateRight

/** BLAKE2b with an eight-byte digest, matching hashlib.blake2b(..., digest_size=8). */
internal object Blake2b64 {
    private val iv = longArrayOf(
        0x6a09e667f3bcc908uL.toLong(),
        0xbb67ae8584caa73buL.toLong(),
        0x3c6ef372fe94f82buL.toLong(),
        0xa54ff53a5f1d36f1uL.toLong(),
        0x510e527fade682d1uL.toLong(),
        0x9b05688c2b3e6c1fuL.toLong(),
        0x1f83d9abfb41bd6buL.toLong(),
        0x5be0cd19137e2179uL.toLong(),
    )
    private val sigma = arrayOf(
        intArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        intArrayOf(14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3),
        intArrayOf(11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4),
        intArrayOf(7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8),
        intArrayOf(9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13),
        intArrayOf(2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9),
        intArrayOf(12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11),
        intArrayOf(13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10),
        intArrayOf(6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5),
        intArrayOf(10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0),
        intArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        intArrayOf(14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3),
    )

    fun digest(input: ByteArray): ByteArray {
        val state = iv.copyOf()
        state[0] = state[0] xor 0x01010008L
        if (input.isEmpty()) {
            compress(state, ByteArray(BLOCK_BYTES), 0L, true)
        } else {
            var offset = 0
            while (offset < input.size) {
                val length = minOf(BLOCK_BYTES, input.size - offset)
                val block = ByteArray(BLOCK_BYTES)
                input.copyInto(block, 0, offset, offset + length)
                offset += length
                compress(state, block, offset.toLong(), offset == input.size)
            }
        }
        return ByteArray(DIGEST_BYTES) { index ->
            ((state[0] ushr (index * 8)) and 0xffL).toByte()
        }
    }

    private fun compress(
        state: LongArray,
        block: ByteArray,
        count: Long,
        last: Boolean,
    ) {
        val message = LongArray(16) { index -> littleEndianLong(block, index * 8) }
        val work = LongArray(16)
        state.copyInto(work, 0)
        iv.copyInto(work, 8)
        work[12] = work[12] xor count
        if (last) work[14] = work[14].inv()
        repeat(12) { round ->
            val s = sigma[round]
            mix(work, 0, 4, 8, 12, message[s[0]], message[s[1]])
            mix(work, 1, 5, 9, 13, message[s[2]], message[s[3]])
            mix(work, 2, 6, 10, 14, message[s[4]], message[s[5]])
            mix(work, 3, 7, 11, 15, message[s[6]], message[s[7]])
            mix(work, 0, 5, 10, 15, message[s[8]], message[s[9]])
            mix(work, 1, 6, 11, 12, message[s[10]], message[s[11]])
            mix(work, 2, 7, 8, 13, message[s[12]], message[s[13]])
            mix(work, 3, 4, 9, 14, message[s[14]], message[s[15]])
        }
        for (index in state.indices) {
            state[index] = state[index] xor work[index] xor work[index + 8]
        }
    }

    private fun mix(
        work: LongArray,
        a: Int,
        b: Int,
        c: Int,
        d: Int,
        x: Long,
        y: Long,
    ) {
        work[a] = work[a] + work[b] + x
        work[d] = rotateRight(work[d] xor work[a], 32)
        work[c] += work[d]
        work[b] = rotateRight(work[b] xor work[c], 24)
        work[a] = work[a] + work[b] + y
        work[d] = rotateRight(work[d] xor work[a], 16)
        work[c] += work[d]
        work[b] = rotateRight(work[b] xor work[c], 63)
    }

    private fun littleEndianLong(bytes: ByteArray, offset: Int): Long {
        var value = 0L
        repeat(8) { index ->
            value = value or ((bytes[offset + index].toLong() and 0xffL) shl (8 * index))
        }
        return value
    }

    private const val BLOCK_BYTES = 128
    private const val DIGEST_BYTES = 8
}
