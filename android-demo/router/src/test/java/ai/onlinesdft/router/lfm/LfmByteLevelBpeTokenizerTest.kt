package ai.onlinesdft.router.lfm

import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class LfmByteLevelBpeTokenizerTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `hermetic byte level BPE recognizes merges Unicode and added tokens`() {
        val tokenizer = miniatureTokenizer()
        val text = "hello world!<special>é"

        assertArrayEquals(
            intArrayOf(24, 29, 18, 31, 30),
            tokenizer.encode(text, addSpecialTokens = false),
        )
        assertArrayEquals(
            intArrayOf(1, 24, 29, 18, 31, 30),
            tokenizer.encode(text, addSpecialTokens = true),
        )
        assertEquals(text, tokenizer.decode(intArrayOf(24, 29, 18, 31, 30)))
        assertEquals(
            "hello world!é",
            tokenizer.decode(
                intArrayOf(24, 29, 18, 31, 30),
                skipSpecialTokens = true,
            ),
        )
    }

    @Test
    fun `decode skips padded model vocabulary ids like Hugging Face`() {
        val tokenizer = miniatureTokenizer()

        assertEquals(
            "hello world",
            tokenizer.decode(intArrayOf(24, 65_000, 29)),
        )
    }

    @Test
    fun `compact history rendering preserves factual callback wording`() {
        val codec = LfmCompactPromptCodec(miniatureTokenizer())
        val rendered = codec.renderStudentPrompt(
            context = "new notification",
            completedInteractions = listOf(
                LfmCompletedInteraction(
                    context = "past notification",
                    executedAction = "LATER",
                    outcome = "OPENED_DIGEST",
                    observedUserSelection = "LATER",
                    delayMinutes = 120,
                ),
            ),
        )

        assertTrue(rendered.startsWith("<|startoftext|><|im_start|>system\n"))
        assertTrue(
            rendered.contains(
                "1. past notification The router placed the notification in a later digest. " +
                    "The user opened it from the digest 120 minutes later. " +
                    "This behavior revealed LATER as the observed user selection on the executed surface. " +
                    "Its observed route was B for LATER.",
            ),
        )
        assertTrue(rendered.contains("UNKNOWN is unlabeled.\n\nNotification: new notification\nRoute:"))
        assertTrue(rendered.endsWith("<|im_start|>assistant\n"))
    }

    @Test
    fun `pinned tokenizer matches Unicode and special token golden`() {
        val tokenizer = pinnedTokenizer()
        val text = "<|startoftext|>Café ☕ <|im_end|><think>μ</think> Mathias python"
        val expected = intArrayOf(
            1, 544, 2305, 860, 40961, 753, 730, 7, 64400, 16883, 64401,
            730, 64011, 730, 64014,
        )

        assertArrayEquals(expected, tokenizer.encode(text, addSpecialTokens = false))
        assertEquals(text, tokenizer.decode(expected, skipSpecialTokens = false))
        assertEquals(
            "Café ☕ <think>μ</think> Mathias python",
            tokenizer.decode(expected, skipSpecialTokens = true),
        )
        assertEquals(542, tokenizer.tokenId("A"))
        assertEquals(543, tokenizer.tokenId("B"))
        assertEquals(544, tokenizer.tokenId("C"))
        assertEquals(64402, tokenizer.vocabularySize)
    }

    @Test
    fun `compact student prompt and token ids match pinned Hugging Face golden`() {
        val codec = LfmCompactPromptCodec(pinnedTokenizer())
        val encoded = codec.encodeStudentPrompt(STUDENT_CONTEXT)

        assertEquals(EXPECTED_STUDENT_PROMPT, encoded.text)
        assertArrayEquals(EXPECTED_STUDENT_IDS, encoded.tokenIds)
        assertEquals(encoded.text, pinnedTokenizer().decode(encoded.tokenIds))
    }

    @Test
    fun `factual hindsight teacher prompt and token ids match pinned Hugging Face golden`() {
        val codec = LfmCompactPromptCodec(pinnedTokenizer())
        val callback = standardTeacherCallback()
        assertEquals(TEACHER_EVIDENCE, codec.renderFactualEvidence(callback))
        val encoded = codec.encodeTeacherPrompt(
            context = TEACHER_CONTEXT,
            callback = callback,
            assessment = TEACHER_ASSESSMENT,
        )

        assertEquals(EXPECTED_TEACHER_PROMPT, encoded.text)
        assertArrayEquals(EXPECTED_TEACHER_IDS, encoded.tokenIds)
        assertEquals(encoded.text, pinnedTokenizer().decode(encoded.tokenIds))
    }

    @Test
    fun `teacher assessment prompt and token ids match pinned Hugging Face golden`() {
        val tokenizer = pinnedTokenizer()
        val codec = LfmCompactPromptCodec(tokenizer)
        val encoded = codec.encodeTeacherAssessmentPrompt(
            context = TEACHER_CONTEXT,
            callback = standardTeacherCallback(),
        )

        assertEquals(EXPECTED_TEACHER_ASSESSMENT_PROMPT, encoded.text)
        assertArrayEquals(EXPECTED_TEACHER_ASSESSMENT_IDS, encoded.tokenIds)
        assertEquals(encoded.text, tokenizer.decode(encoded.tokenIds))
    }

    @Test
    fun `explicit correction evidence names the user correction without invented timing`() {
        val codec = LfmCompactPromptCodec(miniatureTokenizer())
        val rendered = codec.renderFactualEvidence(
            LfmFactualCallback(
                actionTaken = "LATER",
                outcome = "EXPLICIT_USER_CORRECTION",
                observedUserSelection = "ARCHIVE",
                delayMinutes = 0,
            ),
        )

        assertEquals(
            "The router placed the notification in a later digest. " +
                "The user explicitly corrected the route. " +
                "This behavior revealed ARCHIVE as the observed user selection on the executed surface.",
            rendered,
        )
    }

    private fun standardTeacherCallback() = LfmFactualCallback(
        actionTaken = "INTERRUPT",
        outcome = "OPENED_IMMEDIATELY",
        observedUserSelection = "INTERRUPT",
        delayMinutes = 1,
    )

    private fun miniatureTokenizer(): LfmByteLevelBpeTokenizer {
        val file = temporaryFolder.newFile("tokenizer-${System.nanoTime()}.json")
        file.writeText(MINIATURE_TOKENIZER_JSON)
        return LfmByteLevelBpeTokenizer.fromFile(file)
    }

    private fun pinnedTokenizer(): LfmByteLevelBpeTokenizer {
        val file = findPinnedTokenizer()
        assumeTrue(
            "Set -Dlfm.tokenizer.json=/path/to/tokenizer.json to run pinned LFM goldens",
            file?.isFile == true,
        )
        val resolved = requireNotNull(file)
        assertEquals(PINNED_TOKENIZER_SHA256, sha256(resolved))
        return synchronized(PINNED_LOCK) {
            pinnedInstance ?: LfmByteLevelBpeTokenizer.fromFile(resolved).also {
                pinnedInstance = it
            }
        }
    }

    private fun findPinnedTokenizer(): File? {
        System.getProperty("lfm.tokenizer.json")?.let { explicit ->
            return File(explicit)
        }
        val cache = File(
            System.getProperty("user.home"),
            ".cache/huggingface/hub/models--LiquidAI--LFM2.5-230M/" +
                "snapshots/$PINNED_REVISION/tokenizer.json",
        )
        return cache.takeIf(File::isFile)
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    companion object {
        private val PINNED_LOCK = Any()

        @Volatile
        private var pinnedInstance: LfmByteLevelBpeTokenizer? = null

        private const val PINNED_REVISION = "13a53837c4906b4f7405932532ba85d182bb013b"
        private const val PINNED_TOKENIZER_SHA256 =
            "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd"

        private const val STUDENT_CONTEXT =
            "This is a calendar notification titled “Café ☕ review” from Zoë. " +
                "Body: 会议 starts at 14:30 — résumé attached. Metadata: " +
                "category=calendar; importance=0.91; local_hour=14.5; regime=weekday."

        private const val EXPECTED_STUDENT_PROMPT =
            "<|startoftext|><|im_start|>system\n" +
                "You are an on-device notification router.\n" +
                "Assess the partial evidence, then choose exactly one route:\n" +
                "A = INTERRUPT now\n" +
                "B = LATER in a digest\n" +
                "C = ARCHIVE without a notification\n" +
                "Use the current notification and any past completed interactions. Do not add explanation." +
                "<|im_end|>\n<|im_start|>user\n" +
                "Notification: $STUDENT_CONTEXT\nRoute:" +
                "<|im_end|>\n<|im_start|>assistant\n"

        private val EXPECTED_STUDENT_IDS = intArrayOf(
            1, 6, 24131, 708, 4083, 938, 902, 884, 5730, 10272, 39004, 35424,
            819, 9886, 894, 779, 15298, 4589, 521, 1757, 7056, 8544, 1235, 9153,
            1334, 542, 1051, 53975, 559, 13112, 561, 2166, 708, 543, 1051, 903,
            55349, 797, 768, 11813, 708, 544, 1051, 10248, 5750, 21219, 2630, 768,
            39004, 708, 14875, 779, 2835, 39004, 810, 1523, 3650, 7850, 11489,
            523, 4112, 1014, 1614, 14317, 523, 7, 708, 6, 6423, 708, 37796, 535,
            1470, 856, 768, 21041, 39004, 22310, 1371, 544, 2305, 860, 40961, 753,
            4202, 1238, 988, 25778, 7286, 523, 15632, 535, 730, 38543, 11353, 963,
            730, 1536, 535, 1664, 2180, 7490, 51521, 12930, 523, 5735, 11532, 535,
            11319, 538, 2459, 11983, 536, 7083, 538, 525, 523, 10006, 536, 2666,
            11440, 926, 538, 1536, 523, 530, 536, 15817, 538, 18378, 2367, 819,
            33239, 535, 7, 708, 6, 64015, 708,
        )

        private const val TEACHER_CONTEXT =
            "This is a monitoring notification titled “数据库 🚨”. Metadata: " +
                "category=monitoring; regime=on-call."
        private const val TEACHER_EVIDENCE =
            "The router delivered the notification as an immediate interruption. " +
                "The user opened it one minute later. This behavior revealed INTERRUPT " +
                "as the observed user selection on the executed surface."
        private const val TEACHER_ASSESSMENT =
            "Urgent production evidence; alternatives remain possible."

        private const val EXPECTED_TEACHER_PROMPT =
            "<|startoftext|><|im_start|>system\n" +
                "Choose a route for a similar future notification:\n" +
                "A = INTERRUPT now\n" +
                "B = LATER in a digest\n" +
                "C = ARCHIVE silently\n" +
                "Use the notification and observed callback. No hidden label or unchosen outcome is available. " +
                "A digest open after LATER leaves INTERRUPT versus LATER unresolved. " +
                "UNKNOWN supports no route. Keep alternatives possible." +
                "<|im_end|>\n<|im_start|>user\n" +
                "Notification:\n$TEACHER_CONTEXT\n\n" +
                "Observed callback:\n$TEACHER_EVIDENCE\n\n" +
                "Teacher evidence assessment:\n$TEACHER_ASSESSMENT\n\nRoute:" +
                "<|im_end|>\n<|im_start|>assistant\n"

        private val EXPECTED_TEACHER_IDS = intArrayOf(
            1, 6, 24131, 708, 52739, 768, 9153, 875, 768, 3456, 3866, 39004,
            1334, 542, 1051, 53975, 559, 13112, 561, 2166, 708, 543, 1051, 903,
            55349, 797, 768, 11813, 708, 544, 1051, 10248, 5750, 21219, 4598,
            2652, 708, 14875, 779, 39004, 810, 6023, 17873, 523, 3253, 14139,
            7841, 933, 946, 824, 7013, 12143, 856, 3161, 523, 835, 11813, 2714,
            1720, 903, 55349, 6977, 53975, 559, 13112, 561, 13236, 903, 55349,
            55186, 8808, 523, 7753, 49680, 35945, 12457, 1295, 9153, 523, 16823,
            22095, 3023, 523, 7, 708, 6, 6423, 708, 37796, 1334, 2443, 856, 768,
            11078, 39004, 22310, 1371, 20615, 37233, 23805, 758, 611, 10538, 5735,
            11532, 535, 11319, 538, 63884, 800, 536, 15817, 538, 772, 2981, 986,
            523, 509, 20663, 20886, 17873, 1334, 1098, 35424, 14465, 779, 39004,
            906, 902, 12907, 1251, 13662, 523, 941, 5196, 8414, 936, 1235,
            12577, 2919, 523, 1470, 5240, 9494, 53975, 559, 13112, 561, 906,
            779, 6023, 5196, 8579, 884, 779, 20271, 4256, 523, 509, 3840,
            11173, 4589, 8857, 1334, 11522, 7228, 3893, 4589, 536, 22095, 4860,
            3023, 523, 509, 33239, 535, 7, 708, 6, 64015, 708,
        )

        private const val EXPECTED_TEACHER_ASSESSMENT_PROMPT =
            "<|startoftext|><|im_start|>system\n" +
                "In one short paragraph, assess what the notification and observed callback imply " +
                "for a similar future case.\n" +
                "The executed surface reveals only its observed behavior. Do not invent a hidden label " +
                "or unchosen outcome. Explain uncertainty without choosing a route or giving a route code." +
                "<|im_end|>\n<|im_start|>user\n" +
                "Notification:\n$TEACHER_CONTEXT\n\n" +
                "Observed callback:\n$TEACHER_EVIDENCE\n\n" +
                "Assessment:" +
                "<|im_end|>\n<|im_start|>assistant\n"

        private val EXPECTED_TEACHER_ASSESSMENT_IDS = intArrayOf(
            1, 6, 24131, 708, 1286, 1235, 3290, 18569, 521, 4597, 1620, 779,
            39004, 810, 6023, 17873, 32465, 875, 768, 3456, 3866, 2533, 819,
            1098, 20271, 4256, 17987, 1550, 1352, 6023, 5240, 523, 4112, 1014,
            9331, 768, 14139, 7841, 933, 946, 824, 7013, 12143, 523, 43348,
            19902, 2630, 18836, 768, 9153, 933, 7852, 768, 9153, 5214, 523, 7,
            708, 6, 6423, 708, 37796, 1334, 2443, 856, 768, 11078, 39004, 22310,
            1371, 20615, 37233, 23805, 758, 611, 10538, 5735, 11532, 535, 11319,
            538, 63884, 800, 536, 15817, 538, 772, 2981, 986, 523, 509, 20663,
            20886, 17873, 1334, 1098, 35424, 14465, 779, 39004, 906, 902, 12907,
            1251, 13662, 523, 941, 5196, 8414, 936, 1235, 12577, 2919, 523, 1470,
            5240, 9494, 53975, 559, 13112, 561, 906, 779, 6023, 5196, 8579, 884,
            779, 20271, 4256, 523, 509, 9886, 17297, 535, 7, 708, 6, 64015, 708,
        )

        private const val MINIATURE_TOKENIZER_JSON = """
            {
              "version":"1.0",
              "truncation":null,
              "padding":null,
              "added_tokens":[
                {"id":0,"content":"<|pad|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                {"id":1,"content":"<|startoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                {"id":31,"content":"<special>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
              ],
              "normalizer":null,
              "pre_tokenizer":{
                "type":"Sequence",
                "pretokenizers":[
                  {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
                  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
                ]
              },
              "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
              "model":{
                "type":"BPE","dropout":null,"unk_token":null,
                "continuing_subword_prefix":null,"end_of_word_suffix":null,
                "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
                "vocab":{
                  "<|pad|>":0,"<|startoftext|>":1,
                  "h":10,"e":11,"l":12,"o":13,"Ġ":14,"w":15,"r":16,"d":17,"!":18,"Ã":19,"©":20,
                  "he":21,"hel":22,"hell":23,"hello":24,
                  "Ġw":25,"Ġwo":26,"Ġwor":27,"Ġworl":28,"Ġworld":29,"Ã©":30
                },
                "merges":[
                  ["h","e"],["he","l"],["hel","l"],["hell","o"],
                  ["Ġ","w"],["Ġw","o"],["Ġwo","r"],["Ġwor","l"],["Ġworl","d"],
                  ["Ã","©"]
                ]
              }
            }
        """
    }
}
