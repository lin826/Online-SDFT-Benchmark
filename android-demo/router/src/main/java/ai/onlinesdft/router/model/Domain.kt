package ai.onlinesdft.router.model

enum class Route {
    INTERRUPT,
    LATER,
    ARCHIVE;

    companion object {
        fun fromWire(value: String?): Route? = entries.firstOrNull {
            it.name.equals(value, ignoreCase = true)
        }
    }
}

enum class Regime {
    WEEKDAY,
    ON_CALL,
    OFF_HOURS;

    companion object {
        fun fromWire(value: String?): Regime = when (
            value?.trim()?.lowercase()?.replace('-', '_')
        ) {
            "on_call", "oncall" -> ON_CALL
            "off_hours", "offhours" -> OFF_HOURS
            else -> WEEKDAY
        }
    }
}

enum class Outcome {
    OPENED_IMMEDIATELY,
    OPENED_AFTER_DELAY,
    DELETED_NOTIFICATION,
    OPENED_DIGEST,
    DELETED_FROM_DIGEST,
    NO_OBSERVABLE_SELECTION,
    EXPLICIT_USER_CORRECTION,
    /** Android expired an INTERRUPT notification without any user gesture. */
    TIMED_OUT_UNTOUCHED,
}

enum class FeedbackSource {
    ANDROID_CALLBACK,
    DIGEST_CALLBACK,
    EXPLICIT_USER_CORRECTION,
    SYNTHETIC_LAB,
}

enum class ExecutionConstraint(val wireName: String) {
    NONE("none"),
    ONGOING("ongoing"),
    FOREGROUND_SERVICE("foreground_service"),
    CALL("call"),
    MEDIA("media"),
    GROUP_SUMMARY("group_summary"),
    NON_CLEARABLE("non_clearable"),
    DIGEST_UNAVAILABLE("digest_unavailable"),
    MODEL_UNAVAILABLE("model_unavailable"),
}

data class NotificationContext(
    val eventId: String,
    val packageName: String,
    val title: String,
    val body: String,
    val category: String,
    val importance: Float,
    val regime: Regime,
    val hourOfDay: Float,
    val postedAtMillis: Long,
    val caseId: String? = null,
    val isClearable: Boolean = true,
    val isOngoing: Boolean = false,
    val isForegroundService: Boolean = false,
    val isCall: Boolean = false,
    val isMedia: Boolean = false,
    val isGroupSummary: Boolean = false,
    val isNoClear: Boolean = false,
    val canPublishDigest: Boolean = true,
    /** Trusted demo-only wall-clock expiry supplied by a separate publisher. */
    val demoTimeoutMillis: Long? = null,
    /** Editorial preference horizon represented by the accelerated demo expiry. */
    val semanticDelayMinutes: Int? = null,
) {
    init {
        require(eventId.isNotBlank() && eventId.length <= MAX_EVENT_ID_CHARS) {
            "eventId must contain 1..$MAX_EVENT_ID_CHARS characters"
        }
    }

    companion object {
        const val MAX_EVENT_ID_CHARS = 200
    }
}

data class DecisionSnapshot(
    val context: NotificationContext,
    /** Exact compact-chat prompt sealed before the action and later replayed. */
    val studentPrompt: String = "",
    /** Visible notification traits retained for UI/safety only; never optimized. */
    val studentFeatures: FloatArray,
    val probabilities: FloatArray,
    val baseProbabilities: FloatArray,
    /**
     * Frozen-foundation distribution sealed in FP64 before the action. This is
     * the only base distribution accepted by replay; [baseProbabilities] is a
     * lossy display/transport copy.
     */
    val foundationProbabilitiesFp64: DoubleArray = doubleArrayOf(),
    /** True when base probabilities came from the immutable frozen foundation. */
    val baselineAvailable: Boolean = false,
    val chosenRoute: Route,
    val recommendedRoute: Route = chosenRoute,
    val executionConstraint: ExecutionConstraint = ExecutionConstraint.NONE,
    val baseRoute: Route,
    val baseRecommendedRoute: Route = baseRoute,
    val checkpointIndex: Long,
    val adapterChecksum: String,
    val decidedAtMillis: Long,
    val inferenceLatencyMillis: Double,
    val foundationModelId: String = "unknown",
    val foundationPrecision: String = "unknown",
    val foundationRoute: Route = baseRecommendedRoute,
    val foundationAvailable: Boolean = true,
    val foundationInferenceLatencyMillis: Double? = null,
    val foundationPromptTokens: Long? = null,
    val foundationCompletionTokens: Long? = null,
    val foundationTokensPerSecond: Float? = null,
    val runEpoch: Long = 0L,
    /** Raw frozen-foundation A/B/C logits for audit; never optimized. */
    val foundationActionLogits: FloatArray = floatArrayOf(),
    /** Student minus fixed-base A/B/C logits; this is an audit delta, not a trainable head. */
    val adapterLogitDelta: FloatArray = floatArrayOf(),
    /** FP64 LoRA-student distribution sealed for target fusion. */
    val adaptiveDecisionProbabilities: DoubleArray = doubleArrayOf(),
) {
    val foundationProbabilities: FloatArray get() = baseProbabilities
    val loraUpdateIndex: Long get() = checkpointIndex
    val loraCheckpointChecksum: String get() = adapterChecksum
}

data class FactualFeedback(
    val eventId: String,
    val executedRoute: Route,
    val outcome: Outcome,
    val observedSelection: Route?,
    val delayMinutes: Int,
    val source: FeedbackSource,
    val explicitPreference: Route? = null,
    val observedAtMillis: Long,
)

/** Machine-auditable proof for one independently sampled ORT AdamW step. */
data class OptimizerStepProof(
    /** One-based position within this callback update. */
    val stepIndex: Int,
    /** Monotonic LoRA update index after applying this step. */
    val updateIndex: Long,
    /** Exact ordered replay event ids; the triggering/newest event is first. */
    val sampledEventIds: List<String>,
    val batchSize: Int,
    val lossBefore: Double,
    val lossAfter: Double,
    val gradientNorm: Double,
    val unclippedUpdateNorm: Double,
    val appliedUpdateNorm: Double,
) {
    init {
        require(stepIndex >= 1)
        require(updateIndex >= 1L)
        require(sampledEventIds.size == batchSize && batchSize >= 1)
        require(sampledEventIds.all {
            it.isNotBlank() && it.length <= NotificationContext.MAX_EVENT_ID_CHARS
        })
        require(sampledEventIds.distinct().size == sampledEventIds.size)
        require(lossBefore.isFinite() && lossBefore >= 0.0)
        require(lossAfter.isFinite() && lossAfter >= 0.0)
        require(gradientNorm.isFinite() && gradientNorm >= 0.0)
        require(unclippedUpdateNorm.isFinite() && unclippedUpdateNorm >= 0.0)
        require(appliedUpdateNorm.isFinite() && appliedUpdateNorm >= 0.0)
    }
}

data class TrainingMetrics(
    val updateIndex: Long,
    val eventId: String,
    val teacherProbabilities: FloatArray,
    val sealedDecisionProbabilities: FloatArray,
    val behaviorSupport: FloatArray,
    val fusedTarget: FloatArray,
    val lossBefore: Double,
    val lossAfter: Double,
    val targetKlBefore: Double,
    val targetKlAfter: Double,
    /** ORT 1.19 Java does not expose gradient norm; null for device updates. */
    val gradientNorm: Double?,
    val adapterNorm: Double,
    /** LoRA tensor L2 norm reported by the training graph. */
    val deltaNorm: Double,
    /** Absolute change in the reported LoRA tensor norm for this callback. */
    val callbackUpdateNorm: Double,
    /** LoRA tensor norm before this callback. */
    val adapterDeltaNormBefore: Double = 0.0,
    /** LoRA tensor norm after this callback. */
    val adapterDeltaNormAfter: Double = deltaNorm,
    val checksumBefore: String,
    val checksumAfter: String,
    val batchSize: Int,
    val replaySize: Int,
    val optimizerSteps: Int,
    val optimizerStepLosses: DoubleArray = doubleArrayOf(),
    /** Complete proof for every independent optimizer step. */
    val optimizerStepProofs: List<OptimizerStepProof> = emptyList(),
    val trainingExamples: Int = batchSize,
    val teacherForwardLatencyMillis: Double,
    val durationMillis: Double,
) {
    val loraNorm: Double get() = adapterNorm
    val loraNormChange: Double get() = callbackUpdateNorm
    val loraNormBefore: Double get() = adapterDeltaNormBefore
    val loraNormAfter: Double get() = adapterDeltaNormAfter
    val loraCheckpointChecksumBefore: String get() = checksumBefore
    val loraCheckpointChecksumAfter: String get() = checksumAfter
    val updateApplied: Boolean get() = optimizerSteps > 0

    init {
        require(optimizerSteps >= 0)
        require(optimizerStepProofs.size == optimizerSteps)
        require(optimizerStepLosses.size == optimizerSteps)
        optimizerStepProofs.forEachIndexed { index, proof ->
            require(proof.stepIndex == index + 1)
            require(proof.updateIndex == updateIndex - optimizerSteps + index + 1L)
            require(proof.sampledEventIds.first() == eventId)
            require(optimizerStepLosses[index] == proof.lossAfter)
        }
    }
}

data class EvaluationTruth(
    val goldRoute: Route,
    val utilities: FloatArray,
) {
    init {
        require(utilities.size == Route.entries.size)
    }
}

data class EvaluationMetrics(
    val decisions: Int,
    val correct: Int,
    val onlineAccuracy: Double,
    val cumulativeRegret: Double,
    val baseCorrect: Int,
    val baseAccuracy: Double,
    val baseCumulativeRegret: Double,
    val lastStepRegret: Double,
)
