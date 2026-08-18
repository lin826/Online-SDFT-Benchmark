package ai.onlinesdft.router.telemetry

import java.security.MessageDigest

internal data class RawJson(val value: String)

internal fun jsonObject(vararg fields: Pair<String, Any?>): String = fields.joinToString(
    prefix = "{",
    postfix = "}",
    separator = ",",
) { (key, value) -> "\"${escapeJson(key)}\":${jsonValue(value)}" }

internal fun jsonValue(value: Any?): String = when (value) {
    null -> "null"
    is RawJson -> value.value
    is String -> "\"${escapeJson(value)}\""
    is Boolean, is Number -> value.toString()
    is FloatArray -> value.joinToString(prefix = "[", postfix = "]") { it.toString() }
    is DoubleArray -> value.joinToString(prefix = "[", postfix = "]") { it.toString() }
    else -> "\"${escapeJson(value.toString())}\""
}

internal fun routeProbabilities(values: FloatArray): RawJson {
    require(values.size == 3)
    val total = values.sum().coerceAtLeast(1e-8f)
    return RawJson(
        jsonObject(
            "INTERRUPT" to values[0] / total,
            "LATER" to values[1] / total,
            "ARCHIVE" to values[2] / total,
        ),
    )
}

internal fun escapeJson(value: String): String = buildString(value.length + 8) {
    value.forEach { character ->
        when (character) {
            '\\' -> append("\\\\")
            '"' -> append("\\\"")
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            else -> if (character.code < 0x20) {
                append("\\u%04x".format(character.code))
            } else {
                append(character)
            }
        }
    }
}

internal fun redactedHash(value: String): String = MessageDigest
    .getInstance("SHA-256")
    .digest(value.toByteArray(Charsets.UTF_8))
    .joinToString("") { "%02x".format(it) }
    .take(16)
