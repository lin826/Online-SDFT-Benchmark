package ai.onlinesdft.router.lfm

import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.charset.StandardCharsets
import java.util.concurrent.ConcurrentHashMap
import java.util.regex.Pattern

/**
 * Pure Kotlin reader for the Hugging Face `tokenizer.json` used by LFM2.5.
 *
 * The implementation deliberately supports the tokenizer pipeline declared by
 * the pinned Liquid checkpoint: added-token extraction, the Unicode-aware GPT
 * pre-tokenizer regex, byte-level remapping, ranked BPE merges, the BOS template
 * processor, and byte-level decoding. It has no Android framework dependency,
 * so the exact codec can be exercised by both JVM and device tests.
 */
class LfmByteLevelBpeTokenizer private constructor(
    private val vocabulary: Map<String, Int>,
    private val mergeRanks: Map<SymbolPair, Int>,
    addedTokens: List<AddedToken>,
    preTokenizerPattern: String,
    val bosTokenId: Int,
) {
    private val addedById = addedTokens.associateBy(AddedToken::id)
    private val addedByFirstCharacter = addedTokens
        .filter { it.content.isNotEmpty() }
        .groupBy { it.content.first() }
        .mapValues { (_, tokens) ->
            tokens.sortedWith(
                compareByDescending<AddedToken> { it.content.length }
                    .thenBy { it.id },
            )
        }
    private val specialTokenIds = addedTokens.filter(AddedToken::special).mapTo(hashSetOf()) {
        it.id
    }
    private val idToToken: Array<String?>
    private val preTokenizer = compilePreTokenizerPattern(preTokenizerPattern)
    private val bpeCache = ConcurrentHashMap<String, List<String>>()

    init {
        val maximumId = maxOf(
            vocabulary.values.maxOrNull() ?: -1,
            addedTokens.maxOfOrNull(AddedToken::id) ?: -1,
        )
        idToToken = arrayOfNulls(maximumId + 1)
        vocabulary.forEach { (token, id) ->
            require(id >= 0) { "token id cannot be negative" }
            require(idToToken[id] == null || idToToken[id] == token) {
                "duplicate tokenizer id $id"
            }
            idToToken[id] = token
        }
        addedTokens.forEach { token ->
            require(token.id in idToToken.indices) { "added token id is out of range" }
            val existing = idToToken[token.id]
            require(existing == null || existing == token.content) {
                "added token ${token.content} conflicts with vocabulary id ${token.id}"
            }
            idToToken[token.id] = token.content
        }
        require(bosTokenId in idToToken.indices) { "BOS token is missing" }
    }

    val vocabularySize: Int get() = idToToken.size

    fun tokenId(token: String): Int? = addedById.values.firstOrNull {
        it.content == token
    }?.id ?: vocabulary[token]

    /** Equivalent to the fast tokenizer's single-sequence `encode`. */
    fun encode(
        text: String,
        addSpecialTokens: Boolean = true,
    ): IntArray {
        val result = ArrayList<Int>()
        if (addSpecialTokens) result += bosTokenId
        var cursor = 0
        while (cursor < text.length) {
            val match = findNextAddedToken(text, cursor)
            if (match == null) {
                encodePlainText(text.substring(cursor), result)
                break
            }
            if (match.consumedStart > cursor) {
                encodePlainText(text.substring(cursor, match.consumedStart), result)
            }
            result += match.token.id
            cursor = match.consumedEnd
        }
        return result.toIntArray()
    }

    /** Equivalent to byte-level `decode`, with optional special-token removal. */
    fun decode(
        tokenIds: IntArray,
        skipSpecialTokens: Boolean = false,
    ): String {
        val result = StringBuilder()
        val bytes = ByteArrayOutputStream()

        fun flushBytes() {
            if (bytes.size() == 0) return
            result.append(bytes.toByteArray().toString(StandardCharsets.UTF_8))
            bytes.reset()
        }

        tokenIds.forEach { id ->
            // Hugging Face convert_ids_to_tokens returns null for the LFM
            // model's padded vocabulary rows (64,402..65,535), and decode
            // silently omits those null entries. Matching that behavior keeps
            // the Android teacher path byte-compatible with Python.
            if (id !in idToToken.indices || idToToken[id] == null) {
                return@forEach
            }
            val added = addedById[id]
            if (added != null) {
                flushBytes()
                if (!(skipSpecialTokens && id in specialTokenIds)) {
                    result.append(added.content)
                }
                return@forEach
            }
            val token = requireNotNull(idToToken[id])
            token.codePoints().forEach { codePoint ->
                val byte = BYTE_DECODER[codePoint]
                    ?: error("token $id contains a non-byte-level symbol")
                bytes.write(byte)
            }
        }
        flushBytes()
        return result.toString()
    }

    private fun encodePlainText(text: String, output: MutableList<Int>) {
        if (text.isEmpty()) return
        val matcher = preTokenizer.matcher(text)
        var cursor = 0
        while (matcher.find()) {
            require(matcher.start() == cursor) {
                "pre-tokenizer left text unmatched at UTF-16 offset $cursor"
            }
            encodePiece(matcher.group(), output)
            cursor = matcher.end()
        }
        require(cursor == text.length) {
            "pre-tokenizer left text unmatched at UTF-16 offset $cursor"
        }
    }

    private fun encodePiece(piece: String, output: MutableList<Int>) {
        val byteEncoded = buildString {
            piece.toByteArray(StandardCharsets.UTF_8).forEach { signedByte ->
                append(BYTE_ENCODER[signedByte.toInt() and 0xff])
            }
        }
        bpe(byteEncoded).forEach { symbol ->
            output += requireNotNull(vocabulary[symbol]) {
                "BPE symbol is absent from vocabulary: $symbol"
            }
        }
    }

    private fun bpe(byteEncoded: String): List<String> = bpeCache.computeIfAbsent(byteEncoded) {
        var symbols = byteEncoded.codePoints().toArray().map { codePoint ->
            String(Character.toChars(codePoint))
        }
        while (symbols.size > 1) {
            var bestPair: SymbolPair? = null
            var bestRank = Int.MAX_VALUE
            for (index in 0 until symbols.lastIndex) {
                val pair = SymbolPair(symbols[index], symbols[index + 1])
                val rank = mergeRanks[pair] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestPair = pair
                }
            }
            val selected = bestPair ?: break
            val merged = ArrayList<String>(symbols.size)
            var index = 0
            while (index < symbols.size) {
                if (
                    index < symbols.lastIndex &&
                    symbols[index] == selected.left &&
                    symbols[index + 1] == selected.right
                ) {
                    merged += selected.left + selected.right
                    index += 2
                } else {
                    merged += symbols[index]
                    index += 1
                }
            }
            symbols = merged
        }
        symbols
    }

    private fun findNextAddedToken(text: String, start: Int): AddedMatch? {
        var contentStart = start
        while (contentStart < text.length) {
            val candidates = addedByFirstCharacter[text[contentStart]].orEmpty()
            for (token in candidates) {
                if (!text.regionMatches(contentStart, token.content, 0, token.content.length)) {
                    continue
                }
                val contentEnd = contentStart + token.content.length
                if (token.singleWord && !hasWordBoundaries(text, contentStart, contentEnd)) {
                    continue
                }
                var consumedStart = contentStart
                if (token.leftStrip) {
                    while (consumedStart > start) {
                        val codePoint = text.codePointBefore(consumedStart)
                        if (!Character.isWhitespace(codePoint)) break
                        consumedStart -= Character.charCount(codePoint)
                    }
                }
                var consumedEnd = contentEnd
                if (token.rightStrip) {
                    while (consumedEnd < text.length) {
                        val codePoint = text.codePointAt(consumedEnd)
                        if (!Character.isWhitespace(codePoint)) break
                        consumedEnd += Character.charCount(codePoint)
                    }
                }
                return AddedMatch(token, consumedStart, consumedEnd)
            }
            val codePoint = text.codePointAt(contentStart)
            contentStart += Character.charCount(codePoint)
        }
        return null
    }

    private fun hasWordBoundaries(text: String, start: Int, end: Int): Boolean {
        val startsInsideWord = start > 0 && isWordCodePoint(text.codePointBefore(start))
        val endsInsideWord = end < text.length && isWordCodePoint(text.codePointAt(end))
        return !startsInsideWord && !endsInsideWord
    }

    private fun isWordCodePoint(codePoint: Int): Boolean =
        codePoint == '_'.code || Character.isLetterOrDigit(codePoint)

    private data class AddedToken(
        val id: Int,
        val content: String,
        val singleWord: Boolean,
        val leftStrip: Boolean,
        val rightStrip: Boolean,
        val special: Boolean,
    )

    private data class AddedMatch(
        val token: AddedToken,
        val consumedStart: Int,
        val consumedEnd: Int,
    )

    private data class SymbolPair(val left: String, val right: String)

    companion object {
        /*
         * Android's ICU-backed java.util.regex implementation is Unicode-aware,
         * but rejects the Java 7 UNICODE_CHARACTER_CLASS flag itself. Probe the
         * flag independently so an invalid tokenizer regex still fails instead
         * of being mistaken for an Android compatibility case.
         */
        private val UNICODE_CHARACTER_CLASS_SUPPORTED = runCatching {
            Pattern.compile("", Pattern.UNICODE_CHARACTER_CLASS)
        }.isSuccess

        private fun compilePreTokenizerPattern(pattern: String): Pattern =
            if (UNICODE_CHARACTER_CLASS_SUPPORTED) {
                Pattern.compile(pattern, Pattern.UNICODE_CHARACTER_CLASS)
            } else {
                Pattern.compile(pattern)
            }

        fun fromFile(tokenizerJson: File): LfmByteLevelBpeTokenizer {
            require(tokenizerJson.isFile) {
                "tokenizer.json does not exist: ${tokenizerJson.absolutePath}"
            }
            val root = JsonParser(tokenizerJson.readText(Charsets.UTF_8)).parseObject()
            require(root["normalizer"] == null) {
                "only the unnormalized LFM byte-level tokenizer is supported"
            }

            val model = root.requiredObject("model")
            require(model.requiredString("type") == "BPE") { "tokenizer model must be BPE" }
            require(model["dropout"] == null) { "BPE dropout is unsupported" }
            require(model["unk_token"] == null) { "this codec expects no unknown token" }
            require(model["continuing_subword_prefix"] == null)
            require(model["end_of_word_suffix"] == null)
            require(model.booleanOrDefault("byte_fallback", false).not())
            require(model.booleanOrDefault("ignore_merges", false).not())

            val vocabulary = LinkedHashMap<String, Int>()
            model.requiredObject("vocab").forEach { (token, rawId) ->
                vocabulary[token] = rawId.requiredInt("vocabulary id")
            }
            val mergeRanks = HashMap<SymbolPair, Int>()
            model.requiredArray("merges").forEachIndexed { rank, rawMerge ->
                val merge = rawMerge.requiredArray("merge")
                require(merge.size == 2) { "each BPE merge must contain two symbols" }
                mergeRanks[SymbolPair(merge[0].requiredString("merge left"), merge[1].requiredString("merge right"))] = rank
            }

            val addedTokens = root.arrayOrEmpty("added_tokens").map { rawToken ->
                val token = rawToken.requiredObject("added token")
                AddedToken(
                    id = token.requiredInt("id"),
                    content = token.requiredString("content"),
                    singleWord = token.booleanOrDefault("single_word", false),
                    leftStrip = token.booleanOrDefault("lstrip", false),
                    rightStrip = token.booleanOrDefault("rstrip", false),
                    special = token.booleanOrDefault("special", false),
                )
            }
            val bos = addedTokens.firstOrNull { it.content == START_OF_TEXT }
                ?: error("tokenizer has no $START_OF_TEXT token")

            val preTokenizer = root.requiredObject("pre_tokenizer")
            val preTokenizers = if (preTokenizer["type"] == "Sequence") {
                preTokenizer.requiredArray("pretokenizers")
            } else {
                listOf(preTokenizer)
            }
            val split = preTokenizers
                .map { it.requiredObject("pre-tokenizer") }
                .firstOrNull { it["type"] == "Split" }
                ?: error("tokenizer has no Split pre-tokenizer")
            val regex = split.requiredObject("pattern").requiredString("Regex")
            val byteLevel = preTokenizers
                .map { it.requiredObject("pre-tokenizer") }
                .firstOrNull { it["type"] == "ByteLevel" }
                ?: error("tokenizer has no ByteLevel pre-tokenizer")
            require(!byteLevel.booleanOrDefault("add_prefix_space", false))
            require(!byteLevel.booleanOrDefault("use_regex", false))

            return LfmByteLevelBpeTokenizer(
                vocabulary = vocabulary,
                mergeRanks = mergeRanks,
                addedTokens = addedTokens,
                preTokenizerPattern = regex,
                bosTokenId = bos.id,
            )
        }

        private const val START_OF_TEXT = "<|startoftext|>"

        private val BYTE_ENCODER: Array<String>
        private val BYTE_DECODER: Map<Int, Int>

        init {
            val visibleBytes = ArrayList<Int>(256)
            visibleBytes += (33..126)
            visibleBytes += (161..172)
            visibleBytes += (174..255)
            val codePoints = ArrayList(visibleBytes)
            var extraCodePoint = 256
            for (byte in 0..255) {
                if (byte !in visibleBytes) {
                    visibleBytes += byte
                    codePoints += extraCodePoint
                    extraCodePoint += 1
                }
            }
            val encoder = arrayOfNulls<String>(256)
            val decoder = HashMap<Int, Int>(256)
            visibleBytes.indices.forEach { index ->
                val byte = visibleBytes[index]
                val codePoint = codePoints[index]
                encoder[byte] = String(Character.toChars(codePoint))
                decoder[codePoint] = byte
            }
            @Suppress("UNCHECKED_CAST")
            BYTE_ENCODER = encoder as Array<String>
            BYTE_DECODER = decoder
        }
    }
}

private fun Map<String, Any?>.requiredObject(name: String): Map<String, Any?> =
    this[name].requiredObject(name)

private fun Any?.requiredObject(name: String): Map<String, Any?> {
    @Suppress("UNCHECKED_CAST")
    return this as? Map<String, Any?> ?: error("$name must be a JSON object")
}

private fun Map<String, Any?>.requiredArray(name: String): List<Any?> =
    this[name].requiredArray(name)

private fun Any?.requiredArray(name: String): List<Any?> =
    this as? List<Any?> ?: error("$name must be a JSON array")

private fun Map<String, Any?>.arrayOrEmpty(name: String): List<Any?> =
    if (containsKey(name)) this[name].requiredArray(name) else emptyList()

private fun Map<String, Any?>.requiredString(name: String): String =
    this[name].requiredString(name)

private fun Any?.requiredString(name: String): String =
    this as? String ?: error("$name must be a JSON string")

private fun Map<String, Any?>.requiredInt(name: String): Int =
    this[name].requiredInt(name)

private fun Any?.requiredInt(name: String): Int = when (this) {
    is Long -> toInt().also { require(it.toLong() == this) { "$name is out of range" } }
    is Double -> toInt().also { require(it.toDouble() == this) { "$name must be an integer" } }
    else -> error("$name must be a JSON number")
}

private fun Map<String, Any?>.booleanOrDefault(name: String, default: Boolean): Boolean =
    when (val value = this[name]) {
        null -> default
        is Boolean -> value
        else -> error("$name must be a JSON boolean")
    }

/** Small dependency-free JSON reader sufficient for tokenizer.json. */
private class JsonParser(private val source: String) {
    private var cursor = 0

    fun parseObject(): Map<String, Any?> {
        val value = parseValue().requiredObject("tokenizer root")
        skipWhitespace()
        require(cursor == source.length) { "unexpected JSON content at offset $cursor" }
        return value
    }

    private fun parseValue(): Any? {
        skipWhitespace()
        require(cursor < source.length) { "unexpected end of JSON" }
        return when (source[cursor]) {
            '{' -> parseObjectValue()
            '[' -> parseArrayValue()
            '"' -> parseString()
            't' -> parseLiteral("true", true)
            'f' -> parseLiteral("false", false)
            'n' -> parseLiteral("null", null)
            else -> parseNumber()
        }
    }

    private fun parseObjectValue(): Map<String, Any?> {
        expect('{')
        skipWhitespace()
        val result = LinkedHashMap<String, Any?>()
        if (consume('}')) return result
        while (true) {
            skipWhitespace()
            val key = parseString()
            skipWhitespace()
            expect(':')
            result[key] = parseValue()
            skipWhitespace()
            if (consume('}')) return result
            expect(',')
        }
    }

    private fun parseArrayValue(): List<Any?> {
        expect('[')
        skipWhitespace()
        val result = ArrayList<Any?>()
        if (consume(']')) return result
        while (true) {
            result += parseValue()
            skipWhitespace()
            if (consume(']')) return result
            expect(',')
        }
    }

    private fun parseString(): String {
        expect('"')
        val result = StringBuilder()
        while (cursor < source.length) {
            val character = source[cursor++]
            when (character) {
                '"' -> return result.toString()
                '\\' -> {
                    require(cursor < source.length) { "unterminated JSON escape" }
                    when (val escaped = source[cursor++]) {
                        '"', '\\', '/' -> result.append(escaped)
                        'b' -> result.append('\b')
                        'f' -> result.append('\u000c')
                        'n' -> result.append('\n')
                        'r' -> result.append('\r')
                        't' -> result.append('\t')
                        'u' -> {
                            require(cursor + 4 <= source.length) { "short Unicode escape" }
                            val code = source.substring(cursor, cursor + 4).toInt(16)
                            result.append(code.toChar())
                            cursor += 4
                        }
                        else -> error("invalid JSON escape \\$escaped")
                    }
                }
                else -> result.append(character)
            }
        }
        error("unterminated JSON string")
    }

    private fun parseNumber(): Number {
        val start = cursor
        if (source[cursor] == '-') cursor += 1
        while (cursor < source.length && source[cursor].isDigit()) cursor += 1
        var floatingPoint = false
        if (cursor < source.length && source[cursor] == '.') {
            floatingPoint = true
            cursor += 1
            while (cursor < source.length && source[cursor].isDigit()) cursor += 1
        }
        if (cursor < source.length && source[cursor] in "eE") {
            floatingPoint = true
            cursor += 1
            if (cursor < source.length && source[cursor] in "+-") cursor += 1
            while (cursor < source.length && source[cursor].isDigit()) cursor += 1
        }
        require(cursor > start) { "invalid JSON value at offset $cursor" }
        val token = source.substring(start, cursor)
        return if (floatingPoint) token.toDouble() else token.toLong()
    }

    private fun parseLiteral(literal: String, value: Any?): Any? {
        require(source.startsWith(literal, cursor)) { "invalid JSON literal at offset $cursor" }
        cursor += literal.length
        return value
    }

    private fun skipWhitespace() {
        while (cursor < source.length && source[cursor] in " \t\r\n") cursor += 1
    }

    private fun expect(character: Char) {
        require(cursor < source.length && source[cursor] == character) {
            "expected '$character' at JSON offset $cursor"
        }
        cursor += 1
    }

    private fun consume(character: Char): Boolean {
        if (cursor >= source.length || source[cursor] != character) return false
        cursor += 1
        return true
    }
}
