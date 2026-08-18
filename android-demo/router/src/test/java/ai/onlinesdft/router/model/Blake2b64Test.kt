package ai.onlinesdft.router.model

import org.junit.Assert.assertEquals
import org.junit.Test

class Blake2b64Test {
    @Test
    fun `digest matches Python hashlib at empty text unicode and block boundaries`() {
        val cases = linkedMapOf(
            "" to "e4a6a0577479b2b4",
            "title" to "8b340982acd28268",
            "design_review" to "0aa2755e4b1c3d8a",
            "Maya" to "68cc1ac207da7b9a",
            "a".repeat(127) to "03d6a9be11f3f18a",
            "a".repeat(128) to "f06643fe9c7e18da",
            "a".repeat(129) to "e462535bb0c5a299",
            "notification 🚀" to "66ce9d0012576fc4",
        )

        cases.forEach { (value, expected) ->
            val actual = Blake2b64.digest(value.toByteArray(Charsets.UTF_8)).toHex()
            assertEquals("digest for ${value.take(24)}", expected, actual)
        }
    }

    private fun ByteArray.toHex(): String = joinToString("") { byte ->
        "%02x".format(byte.toInt() and 0xff)
    }
}
