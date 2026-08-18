package ai.onlinesdft.router.lfm

data class LfmEncodedPrompt(
    val text: String,
    val tokenIds: IntArray,
)

data class LfmCompletedInteraction(
    val context: String,
    val executedAction: String,
    val outcome: String,
    val observedUserSelection: String,
    val delayMinutes: Int,
)

data class LfmTeacherObservation(
    val context: String,
    val evidence: String,
)

data class LfmPromptInsertion(
    val prefix: String,
    val suffix: String,
)

data class LfmFactualCallback(
    val actionTaken: String,
    val outcome: String,
    val observedUserSelection: String,
    val delayMinutes: Int,
)

/** Exact `compact` student and teacher prompts from online_sdft.methods. */
class LfmCompactPromptCodec(
    private val tokenizer: LfmByteLevelBpeTokenizer? = null,
) {
    fun renderStudentPrompt(
        context: String,
        completedInteractions: List<LfmCompletedInteraction> = emptyList(),
    ): String {
        val history = if (completedInteractions.isEmpty()) {
            ""
        } else {
            val blocks = completedInteractions.mapIndexed { index, interaction ->
                structuredInteraction(index + 1, interaction)
            }
            "Past completed interactions:\n" +
                blocks.joinToString("\n") +
                "\nUNKNOWN is unlabeled.\n\n"
        }
        return renderSystemAndUser(
            system = STUDENT_SYSTEM_PROMPT,
            user = history + "Notification: $context\nRoute:",
        )
    }

    fun encodeStudentPrompt(
        context: String,
        completedInteractions: List<LfmCompletedInteraction> = emptyList(),
    ): LfmEncodedPrompt = encode(
        renderStudentPrompt(context, completedInteractions),
    )

    fun renderTeacherPrompt(
        observation: LfmTeacherObservation,
        assessment: String? = null,
    ): String {
        val assessmentBlock = if (assessment != null && assessment.isNotEmpty()) {
            "\n\nTeacher evidence assessment:\n${assessment.trim()}"
        } else {
            ""
        }
        val user = "Notification:\n${observation.context}\n\n" +
            "Observed callback:\n${observation.evidence}" +
            assessmentBlock +
            "\n\nRoute:"
        return renderSystemAndUser(TEACHER_SYSTEM_PROMPT, user)
    }

    fun encodeTeacherPrompt(
        observation: LfmTeacherObservation,
        assessment: String? = null,
    ): LfmEncodedPrompt = encode(renderTeacherPrompt(observation, assessment))

    fun renderTeacherPrompt(
        context: String,
        callback: LfmFactualCallback,
        assessment: String? = null,
    ): String = renderTeacherPrompt(
        observation = LfmTeacherObservation(
            context = context,
            evidence = renderFactualEvidence(callback),
        ),
        assessment = assessment,
    )

    fun encodeTeacherPrompt(
        context: String,
        callback: LfmFactualCallback,
        assessment: String? = null,
    ): LfmEncodedPrompt = encode(renderTeacherPrompt(context, callback, assessment))

    /** Prefix/suffix surrounding the generated 40-token evidence assessment. */
    fun teacherScoringInsertion(
        context: String,
        callback: LfmFactualCallback,
    ): LfmPromptInsertion {
        val rendered = renderTeacherPrompt(
            context = context,
            callback = callback,
            assessment = ASSESSMENT_INSERTION_MARKER,
        )
        val markerIndex = rendered.indexOf(ASSESSMENT_INSERTION_MARKER)
        check(markerIndex >= 0) { "teacher assessment insertion marker is missing" }
        check(rendered.indexOf(ASSESSMENT_INSERTION_MARKER, markerIndex + 1) < 0) {
            "teacher assessment insertion marker is ambiguous"
        }
        return LfmPromptInsertion(
            prefix = rendered.substring(0, markerIndex),
            suffix = rendered.substring(markerIndex + ASSESSMENT_INSERTION_MARKER.length),
        )
    }

    fun renderTeacherAssessmentPrompt(
        observation: LfmTeacherObservation,
    ): String = renderSystemAndUser(
        system = TEACHER_REASONING_SYSTEM_PROMPT,
        user = "Notification:\n${observation.context}\n\n" +
            "Observed callback:\n${observation.evidence}\n\n" +
            "Assessment:",
    )

    fun encodeTeacherAssessmentPrompt(
        observation: LfmTeacherObservation,
    ): LfmEncodedPrompt = encode(renderTeacherAssessmentPrompt(observation))

    fun renderTeacherAssessmentPrompt(
        context: String,
        callback: LfmFactualCallback,
    ): String = renderTeacherAssessmentPrompt(
        LfmTeacherObservation(
            context = context,
            evidence = renderFactualEvidence(callback),
        ),
    )

    fun encodeTeacherAssessmentPrompt(
        context: String,
        callback: LfmFactualCallback,
    ): LfmEncodedPrompt = encode(renderTeacherAssessmentPrompt(context, callback))

    fun renderFactualEvidence(callback: LfmFactualCallback): String =
        narrativeEvidence(
            executedAction = callback.actionTaken,
            outcomeName = callback.outcome,
            observedSelection = callback.observedUserSelection,
            delayMinutes = callback.delayMinutes,
        )

    private fun encode(prompt: String): LfmEncodedPrompt = LfmEncodedPrompt(
        text = prompt,
        tokenIds = requireNotNull(tokenizer) {
            "A tokenizer is required for encode*; render* is dependency-free"
        }.encode(prompt, addSpecialTokens = false),
    )

    private fun renderSystemAndUser(system: String, user: String): String =
        START_OF_TEXT +
            IM_START + "system\n" + system + IM_END + "\n" +
            IM_START + "user\n" + user + IM_END + "\n" +
            IM_START + "assistant\n"

    private fun structuredInteraction(
        index: Int,
        interaction: LfmCompletedInteraction,
    ): String {
        val selection = interaction.observedUserSelection
        val label = ACTION_CODES[selection]?.let { code ->
            "Its observed route was $code for $selection."
        } ?: "This interaction is unlabeled."
        return "$index. ${interaction.context} " +
            narrativeEvidence(interaction) +
            " $label"
    }

    private fun narrativeEvidence(interaction: LfmCompletedInteraction): String {
        return narrativeEvidence(
            executedAction = interaction.executedAction,
            outcomeName = interaction.outcome,
            observedSelection = interaction.observedUserSelection,
            delayMinutes = interaction.delayMinutes,
        )
    }

    private fun narrativeEvidence(
        executedAction: String,
        outcomeName: String,
        observedSelection: String,
        delayMinutes: Int,
    ): String {
        val route = requireNotNull(ROUTE_NARRATIVES[executedAction]) {
            "unknown executed action $executedAction"
        }
        val outcome = requireNotNull(OUTCOME_NARRATIVES[outcomeName]) {
            "unknown observed outcome $outcomeName"
        }
        val sentences = ArrayList<String>(3)
        sentences += "The router $route."
        sentences += when {
            outcomeName == "EXPLICIT_USER_CORRECTION" -> "$outcome."
            outcomeName == "NO_OBSERVABLE_SELECTION" ||
                outcomeName == "TIMED_OUT_UNTOUCHED" ->
                "$outcome during the $delayMinutes minute observation window."
            delayMinutes == 1 -> "$outcome one minute later."
            else -> "$outcome $delayMinutes minutes later."
        }
        sentences += if (observedSelection == "UNKNOWN") {
            "The user's preferred route remains unknown because the executed surface revealed no selection."
        } else {
            "This behavior revealed $observedSelection as the observed user selection on the executed surface."
        }
        return sentences.joinToString(" ")
    }

    companion object {
        const val STUDENT_SYSTEM_PROMPT = """You are an on-device notification router.
Assess the partial evidence, then choose exactly one route:
A = INTERRUPT now
B = LATER in a digest
C = ARCHIVE without a notification
Use the current notification and any past completed interactions. Do not add explanation."""

        const val TEACHER_SYSTEM_PROMPT = """Choose a route for a similar future notification:
A = INTERRUPT now
B = LATER in a digest
C = ARCHIVE silently
Use the notification and observed callback. No hidden label or unchosen outcome is available. A digest open after LATER leaves INTERRUPT versus LATER unresolved. UNKNOWN supports no route. Keep alternatives possible."""

        const val TEACHER_REASONING_SYSTEM_PROMPT = """In one short paragraph, assess what the notification and observed callback imply for a similar future case.
The executed surface reveals only its observed behavior. Do not invent a hidden label or unchosen outcome. Explain uncertainty without choosing a route or giving a route code."""

        private const val START_OF_TEXT = "<|startoftext|>"
        private const val IM_START = "<|im_start|>"
        private const val IM_END = "<|im_end|>"
        private const val ASSESSMENT_INSERTION_MARKER =
            "__ONLINE_SDFT_GENERATED_ASSESSMENT_4D58250A__"

        private val ACTION_CODES = mapOf(
            "INTERRUPT" to "A",
            "LATER" to "B",
            "ARCHIVE" to "C",
        )

        private val ROUTE_NARRATIVES = mapOf(
            "INTERRUPT" to "delivered the notification as an immediate interruption",
            "LATER" to "placed the notification in a later digest",
            "ARCHIVE" to "archived the item without delivering a notification",
        )

        private val OUTCOME_NARRATIVES = mapOf(
            "OPENED_IMMEDIATELY" to "The user opened it",
            "OPENED_AFTER_DELAY" to "The user opened it",
            "DELETED_NOTIFICATION" to "The user deleted the immediate notification",
            "OPENED_DIGEST" to "The user opened it from the digest",
            "DELETED_FROM_DIGEST" to "The user deleted it from the digest",
            "NO_OBSERVABLE_SELECTION" to
                "No delivered notification surface revealed a user choice",
            "EXPLICIT_USER_CORRECTION" to "The user explicitly corrected the route",
            "TIMED_OUT_UNTOUCHED" to
                "The notification expired without a user gesture",
        )
    }
}
