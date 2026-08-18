package ai.onlinesdft.router.model

import ai.onlinesdft.router.lfm.LfmCompactPromptCodec
import ai.onlinesdft.router.lfm.LfmFactualCallback
import kotlin.math.abs

/**
 * Online-SDFT over a real rank-4 LoRA adapter inside LFM2.5-230M.
 *
 * The immutable zero-adapter graph supplies the base and hindsight teacher.
 * Student decisions and optimizer updates use the live ORT training session.
 * There is no residual policy head or hand-engineered feature learner here.
 */
class OnlineSdftLearner(
    private val foundationRuntime: FrozenFoundationRuntime,
    private val replayStore: LoraReplayStore = LoraReplayStore(),
    private val promptCodec: LfmCompactPromptCodec = LfmCompactPromptCodec(),
    private val clockMillis: () -> Long = System::currentTimeMillis,
) {
    private val replay = ArrayDeque<LoraTrainingExample>()
    private var replayRng = NumpyReplayRng.canonicalSeed57()
    private var replayGeneration = 0L
    private var replayRestoreAttempted = false
    private var checkpointRestoreError: String? = null
    private var lastLessonBuffered = false
    private val processedFeedbackFingerprints = linkedSetOf<String>()
    private val baseCache = object : LinkedHashMap<String, FrozenFoundationEvaluation>(
        BASE_CACHE_CAPACITY,
        0.75f,
        true,
    ) {
        override fun removeEldestEntry(
            eldest: MutableMap.MutableEntry<String, FrozenFoundationEvaluation>?,
        ): Boolean = size > BASE_CACHE_CAPACITY
    }

    internal var lastStepSampledEventIds: List<List<String>> = emptyList()
        private set

    init {
        foundationRuntime.preload()
    }

    @Synchronized
    fun prepareFrozenBaseline(contexts: List<NotificationContext>): Int {
        val distinct = contexts.filter { it.caseId != null }.distinctBy { it.caseId }
        distinct.forEach { frozenEvaluation(studentPrompt(it)) }
        return distinct.size
    }

    @Synchronized
    fun decide(context: NotificationContext): DecisionSnapshot {
        ensureReplayRestored()
        val visibleFeatures = FeatureEncoder.student(context)
        val canonicalContext = FeatureEncoder.canonicalStudentContext(context)
        val prompt = promptCodec.renderStudentPrompt(canonicalContext)
        val started = System.nanoTime()
        val base = runCatching { frozenEvaluation(prompt) }.getOrNull()
        val student = runCatching { foundationRuntime.evaluateStudent(prompt) }.getOrNull()
        val runtimeStatus = foundationRuntime.runtimeStatus()
        val adapter = foundationRuntime.adapterStatus()
        if (base == null || student == null) {
            val safe = floatArrayOf(1f, 0f, 0f)
            val execution = NotificationExecutionPolicy.plan(context, Route.INTERRUPT)
            return DecisionSnapshot(
                context = context,
                studentPrompt = prompt,
                studentFeatures = visibleFeatures,
                probabilities = safe,
                baseProbabilities = safe.copyOf(),
                foundationProbabilitiesFp64 = doubleArrayOf(1.0, 0.0, 0.0),
                baselineAvailable = false,
                chosenRoute = execution.effectiveRoute,
                recommendedRoute = Route.INTERRUPT,
                executionConstraint = ExecutionConstraint.MODEL_UNAVAILABLE,
                baseRoute = execution.effectiveRoute,
                baseRecommendedRoute = Route.INTERRUPT,
                checkpointIndex = adapter.updateIndex,
                adapterChecksum = adapter.checksum,
                decidedAtMillis = clockMillis(),
                inferenceLatencyMillis = elapsedMillis(started),
                foundationModelId = runtimeStatus.modelId,
                foundationPrecision = runtimeStatus.precision,
                foundationRoute = Route.INTERRUPT,
                foundationAvailable = false,
                adaptiveDecisionProbabilities = doubleArrayOf(1.0, 0.0, 0.0),
            )
        }

        val baseProbabilities = base.actionProbabilitiesFp64.normalized()
        val studentProbabilities = student.actionProbabilitiesFp64.normalized()
        val recommended = studentProbabilities.argmaxRoute()
        val baseRecommended = baseProbabilities.argmaxRoute()
        val execution = NotificationExecutionPolicy.plan(context, recommended)
        val baseExecution = NotificationExecutionPolicy.plan(context, baseRecommended)
        val logitDelta = FloatArray(Route.entries.size) { index ->
            student.actionLogits[index] - base.actionLogits[index]
        }
        return DecisionSnapshot(
            context = context,
            studentPrompt = prompt,
            studentFeatures = visibleFeatures,
            probabilities = studentProbabilities.toFloatArrayCopy(),
            baseProbabilities = baseProbabilities.toFloatArrayCopy(),
            foundationProbabilitiesFp64 = baseProbabilities.copyOf(),
            baselineAvailable = true,
            chosenRoute = execution.effectiveRoute,
            recommendedRoute = recommended,
            executionConstraint = execution.constraint,
            baseRoute = baseExecution.effectiveRoute,
            baseRecommendedRoute = baseRecommended,
            checkpointIndex = adapter.updateIndex,
            adapterChecksum = adapter.checksum,
            decidedAtMillis = clockMillis(),
            inferenceLatencyMillis = elapsedMillis(started),
            foundationModelId = runtimeStatus.modelId,
            foundationPrecision = runtimeStatus.precision,
            foundationRoute = baseRecommended,
            foundationAvailable = true,
            foundationInferenceLatencyMillis = base.latencyMillis,
            foundationPromptTokens = student.promptTokens,
            foundationCompletionTokens = 0L,
            foundationTokensPerSecond = null,
            foundationActionLogits = base.actionLogits.copyOf(),
            adapterLogitDelta = logitDelta,
            adaptiveDecisionProbabilities = studentProbabilities.copyOf(),
        )
    }

    /** Current LoRA score for a sealed prompt; measurement only. */
    @Synchronized
    fun reevaluateDistribution(decision: DecisionSnapshot): FloatArray? = runCatching {
        foundationRuntime.evaluateStudent(decision.studentPrompt)
            .actionProbabilitiesFp64
            .normalized()
            .toFloatArrayCopy()
    }.getOrNull()

    @Synchronized
    fun learn(decision: DecisionSnapshot, feedback: FactualFeedback): TrainingMetrics? {
        ensureReplayRestored()
        require(decision.context.eventId == feedback.eventId)
        require(decision.chosenRoute == feedback.executedRoute)
        val fingerprint = feedbackFingerprint(feedback)
        if (fingerprint in processedFeedbackFingerprints) return null
        lastLessonBuffered = false
        if (!decision.foundationAvailable || decision.studentPrompt.isBlank()) return null
        if (decision.adaptiveDecisionProbabilities.size != Route.entries.size) return null
        if (feedback.explicitPreference == null && feedback.observedSelection == null) return null
        if (SdftTargetBuilder.support(feedback) == null) return null

        val callback = LfmFactualCallback(
            actionTaken = feedback.executedRoute.name,
            outcome = feedback.outcome.name,
            observedUserSelection = (
                feedback.explicitPreference ?: feedback.observedSelection
            )?.name ?: "UNKNOWN",
            delayMinutes = feedback.delayMinutes,
        )
        val teacherPrompt = promptCodec.renderTeacherPrompt(
            context = FeatureEncoder.canonicalStudentContext(decision.context),
            callback = callback,
            assessment = null,
        )
        val teacher = runCatching { foundationRuntime.evaluate(teacherPrompt) }.getOrNull()
            ?: return null
        val target = SdftTargetBuilder.build(
            teacher = teacher.actionProbabilitiesFp64.normalized(),
            sealedDecision = decision.adaptiveDecisionProbabilities.normalized(),
            feedback = feedback,
        ) ?: return null
        val row = LoraTrainingExample(
            eventId = decision.context.eventId,
            prompt = decision.studentPrompt,
            target = target.probabilitiesFp64.copyOf(),
            replayLabel = target.replayLabel,
        )

        replay.addLast(row)
        while (replay.size > REPLAY_CAPACITY) replay.removeFirst()
        rememberProcessedFeedback(fingerprint)
        if (replay.size < WARMUP_EXAMPLES) {
            saveReplay()
            lastStepSampledEventIds = emptyList()
            lastLessonBuffered = true
            return null
        }

        val trainingStarted = System.nanoTime()
        val batch = selectionBalancedReplayBatch()
        lastStepSampledEventIds = listOf(batch.map { it.eventId })
        val step = foundationRuntime.trainStep(batch)
        saveReplay()
        val appliedNorm = abs(step.adapterNormAfter - step.adapterNormBefore)
        val proof = OptimizerStepProof(
            stepIndex = 1,
            updateIndex = step.updateIndex,
            sampledEventIds = batch.map { it.eventId },
            batchSize = batch.size,
            lossBefore = step.lossBefore,
            lossAfter = step.lossAfter,
            gradientNorm = 0.0,
            unclippedUpdateNorm = appliedNorm,
            appliedUpdateNorm = appliedNorm,
        )
        return TrainingMetrics(
            updateIndex = step.updateIndex,
            eventId = feedback.eventId,
            teacherProbabilities = teacher.actionProbabilitiesFp64.toFloatArrayCopy(),
            sealedDecisionProbabilities = decision.adaptiveDecisionProbabilities.toFloatArrayCopy(),
            behaviorSupport = target.behaviorSupport.copyOf(),
            fusedTarget = target.probabilities.copyOf(),
            lossBefore = step.lossBefore,
            lossAfter = step.lossAfter,
            targetKlBefore = 0.0,
            targetKlAfter = 0.0,
            gradientNorm = null,
            adapterNorm = step.adapterNormAfter,
            deltaNorm = step.adapterNormAfter,
            callbackUpdateNorm = appliedNorm,
            adapterDeltaNormBefore = step.adapterNormBefore,
            adapterDeltaNormAfter = step.adapterNormAfter,
            checksumBefore = step.checksumBefore,
            checksumAfter = step.checksumAfter,
            batchSize = batch.size,
            replaySize = replay.size,
            optimizerSteps = 1,
            optimizerStepLosses = doubleArrayOf(step.lossAfter),
            optimizerStepProofs = listOf(proof),
            trainingExamples = batch.size,
            teacherForwardLatencyMillis = teacher.latencyMillis,
            durationMillis = elapsedMillis(trainingStarted),
        )
    }

    @Synchronized
    fun status(): ModelStatus {
        ensureReplayRestored()
        val adapter = foundationRuntime.adapterStatus()
        return ModelStatus(
            updateIndex = adapter.updateIndex,
            checksum = adapter.checksum,
            adapterNorm = adapter.l2Norm,
            rawAdapterNorm = adapter.l2Norm,
            replaySize = replay.size,
            trainableParameters = adapter.trainableParameters,
            trainableTensors = adapter.trainableTensors,
            foundationStatus = foundationRuntime.runtimeStatus(),
            learnerError = checkpointRestoreError,
            lastLessonBuffered = lastLessonBuffered,
            warmupRemaining = (WARMUP_EXAMPLES - replay.size).coerceAtLeast(0),
        )
    }

    @Synchronized
    fun reset() {
        replayStore.clear()
        foundationRuntime.resetAdapter()
        replay.clear()
        replayRng = NumpyReplayRng.canonicalSeed57()
        processedFeedbackFingerprints.clear()
        replayGeneration = 0L
        replayRestoreAttempted = true
        checkpointRestoreError = null
        lastLessonBuffered = false
        baseCache.clear()
        lastStepSampledEventIds = emptyList()
    }

    private fun selectionBalancedReplayBatch(): List<LoraTrainingExample> {
        val all = replay.toList()
        val newest = all.last()
        if (all.size == 1 || MAX_BATCH_SIZE == 1) return listOf(newest)
        val prior = all.dropLast(1)
        val sampleSize = minOf(MAX_BATCH_SIZE - 1, prior.size)
        val groupCounts = prior.groupingBy { it.replayLabel }.eachCount()
        val weights = DoubleArray(prior.size) { index ->
            val row = prior[index]
            val groupWeight = if (row.replayLabel == AMBIGUOUS_LABEL) {
                AMBIGUOUS_REPLAY_GROUP_WEIGHT
            } else {
                1.0
            }
            groupWeight / requireNotNull(groupCounts[row.replayLabel])
        }
        val sampled = replayRng.weightedChoiceWithoutReplacement(weights, sampleSize)
        return buildList(sampleSize + 1) {
            add(newest)
            sampled.forEach { add(prior[it]) }
        }
    }

    private fun saveReplay() {
        val adapter = foundationRuntime.adapterStatus()
        val receipt = replayStore.save(
            LoraReplayCheckpoint(
                generation = replayGeneration,
                modelId = foundationRuntime.runtimeStatus().modelId,
                adapterUpdateIndex = adapter.updateIndex,
                adapterChecksum = adapter.checksum,
                replay = replay.map(LoraTrainingExample::deepCopy),
                replayRng = replayRng.snapshot(),
                processedFeedbackFingerprints = processedFeedbackFingerprints.toList(),
            ),
        )
        replayGeneration = receipt.generation
    }

    private fun restoreReplay() {
        val checkpoint = runCatching { replayStore.load() }.getOrElse { error ->
            checkpointRestoreError = "LoRA replay checkpoint read failed: ${error.message}"
            return
        } ?: return
        val adapter = foundationRuntime.adapterStatus()
        if (
            checkpoint.modelId != foundationRuntime.runtimeStatus().modelId ||
            checkpoint.adapterUpdateIndex != adapter.updateIndex ||
            checkpoint.adapterChecksum != adapter.checksum
        ) {
            checkpointRestoreError = "Replay checkpoint does not match the committed LoRA checkpoint"
            return
        }
        replay.clear()
        checkpoint.replay.forEach { replay.addLast(it.deepCopy()) }
        replayRng = NumpyReplayRng.fromSnapshot(checkpoint.replayRng)
        processedFeedbackFingerprints.clear()
        processedFeedbackFingerprints.addAll(checkpoint.processedFeedbackFingerprints)
        replayGeneration = checkpoint.generation
    }

    private fun ensureReplayRestored() {
        if (replayRestoreAttempted) return
        val adapter = foundationRuntime.adapterStatus()
        if (adapter.trainableParameters == 0) return
        replayRestoreAttempted = true
        restoreReplay()
    }

    private fun frozenEvaluation(prompt: String): FrozenFoundationEvaluation {
        baseCache[prompt]?.let { return it.deepCopy() }
        val evaluated = foundationRuntime.evaluate(prompt)
        evaluated.actionProbabilitiesFp64.normalized()
        baseCache[prompt] = evaluated.deepCopy()
        return evaluated.deepCopy()
    }

    private fun studentPrompt(context: NotificationContext): String =
        promptCodec.renderStudentPrompt(FeatureEncoder.canonicalStudentContext(context))

    private fun FrozenFoundationEvaluation.deepCopy() = copy(
        actionLogits = actionLogits.copyOf(),
        actionProbabilities = actionProbabilities.copyOf(),
        actionProbabilitiesFp64 = actionProbabilitiesFp64.copyOf(),
    )

    private fun DoubleArray.normalized(): DoubleArray {
        require(size == Route.entries.size)
        val result = DoubleArray(size) { index -> this[index].coerceAtLeast(PROBABILITY_FLOOR) }
        require(result.all { it.isFinite() })
        val total = result.sum()
        require(total.isFinite() && total > 0.0)
        result.indices.forEach { result[it] /= total }
        return result
    }

    private fun DoubleArray.argmaxRoute(): Route {
        var best = 0
        for (index in 1 until size) if (this[index] > this[best]) best = index
        return Route.entries[best]
    }

    private fun DoubleArray.toFloatArrayCopy() = FloatArray(size) { this[it].toFloat() }

    private fun elapsedMillis(startedNanos: Long) =
        (System.nanoTime() - startedNanos) / 1_000_000.0

    private fun feedbackFingerprint(feedback: FactualFeedback): String = listOf(
        feedback.eventId,
        feedback.executedRoute.name,
        feedback.outcome.name,
        feedback.source.name,
        feedback.explicitPreference?.name.orEmpty(),
        feedback.observedSelection?.name.orEmpty(),
    ).joinToString("\u001f")

    private fun rememberProcessedFeedback(fingerprint: String) {
        processedFeedbackFingerprints.remove(fingerprint)
        processedFeedbackFingerprints.add(fingerprint)
        while (processedFeedbackFingerprints.size > MAX_PROCESSED_FEEDBACK) {
            processedFeedbackFingerprints.remove(processedFeedbackFingerprints.first())
        }
    }

    companion object {
        const val REPLAY_CAPACITY = 32
        /** Kept small because the training graph pads prompts and runs on phone CPU/RAM. */
        // ORT 1.19.2 on Android ARM can produce non-finite LFM activations for
        // long, dynamically padded multi-row batches. A single newest-row
        // update remains a real LoRA forward/backward/AdamW step and also
        // keeps peak phone memory lower.
        const val MAX_BATCH_SIZE = 1
        const val OPTIMIZER_STEPS = 1
        const val WARMUP_EXAMPLES = 4
        const val AMBIGUOUS_REPLAY_GROUP_WEIGHT = 0.05
        internal const val ASSESSMENT_FALLBACK =
            "The factual callback is incomplete or ambiguous; preserve uncertainty across plausible routes."
        private const val AMBIGUOUS_LABEL = "AMBIGUOUS"
        private const val BASE_CACHE_CAPACITY = 96
        private const val MAX_PROCESSED_FEEDBACK = 512
        private const val PROBABILITY_FLOOR = 1e-8
    }
}

data class ModelStatus(
    val updateIndex: Long,
    val checksum: String,
    val adapterNorm: Double,
    val rawAdapterNorm: Double = adapterNorm,
    val replaySize: Int,
    val trainableParameters: Int,
    val trainableTensors: Int = LoraAdapterStatus.EXPECTED_TENSORS,
    val foundationStatus: FoundationModelStatus,
    val learnerError: String? = null,
    val lastLessonBuffered: Boolean = false,
    val warmupRemaining: Int = 0,
) {
    val loraCheckpointChecksum: String get() = checksum
    val loraNorm: Double get() = adapterNorm
    val rawLoraNorm: Double get() = rawAdapterNorm
}

/** Retained only for validation of legacy reasoning-enabled artifacts. */
internal object TeacherAssessmentBoundary {
    private val whitespace = Regex("\\s+")
    private val forbidden = listOf(
        Regex("\\b(?:busy|busyness|deadline|urgency|affinity)\\b", RegexOption.IGNORE_CASE),
        Regex("\\binterruption(?:[\\s_-]+)filter\\b", RegexOption.IGNORE_CASE),
        Regex("\\b(?:output only|copy exactly|correct route|correct answer|gold action|" +
            "scalar reward|counterfactual|oracle utility)\\b", RegexOption.IGNORE_CASE),
        Regex("^(?:[ABC]|INTERRUPT|LATER|ARCHIVE)[.!]?$", RegexOption.IGNORE_CASE),
        Regex("\\b(?:callback|evidence|assessment)\\s+(?:strongly\\s+)?" +
            "(?:favors|favours|indicates|recommends|points\\s+to)\\s+" +
            "(?:INTERRUPT|LATER|ARCHIVE|[ABC])\\b", RegexOption.IGNORE_CASE),
    )

    fun sanitize(raw: String): String {
        val normalized = raw.trim().replace(whitespace, " ")
        return if (normalized.isEmpty() || forbidden.any { it.containsMatchIn(normalized) }) {
            OnlineSdftLearner.ASSESSMENT_FALLBACK
        } else {
            normalized
        }
    }
}
