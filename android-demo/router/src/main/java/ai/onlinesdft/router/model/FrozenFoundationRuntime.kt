package ai.onlinesdft.router.model

/** One A/B/C next-token evaluation from either the base or adapted LFM. */
data class FrozenFoundationEvaluation(
    val actionLogits: FloatArray,
    /** Display/ORT transport copy. Target construction uses [actionProbabilitiesFp64]. */
    val actionProbabilities: FloatArray,
    /** FP64 softmax computed directly from the frozen FP32 logits, before any Float cast. */
    val actionProbabilitiesFp64: DoubleArray,
    val promptTokens: Long,
    val latencyMillis: Double,
) {
    init {
        require(actionLogits.size == Route.entries.size)
        require(actionProbabilities.size == Route.entries.size)
        require(actionProbabilitiesFp64.size == Route.entries.size)
        actionProbabilitiesFp64.forEach { require(it.isFinite() && it >= 0.0) }
        require(kotlin.math.abs(actionProbabilitiesFp64.sum() - 1.0) <= 1e-12)
    }
}

/**
 * Blocking on-device LoRA boundary used by Online-SDFT.
 *
 * [evaluate] always uses the immutable zero-adapter checkpoint and is the
 * fixed base/teacher. [evaluateStudent] and [trainStep] share the live LoRA
 * checkpoint. Implementations must make optimizer step + checkpoint commit
 * atomic from this interface's point of view.
 */
interface FrozenFoundationRuntime : AutoCloseable {
    fun preload()

    /** Fixed base model (LoRA A/B tensors at their exported initial values). */
    fun evaluate(fullCompactPrompt: String): FrozenFoundationEvaluation

    /** Current student model with the live LoRA checkpoint enabled. */
    fun evaluateStudent(fullCompactPrompt: String): FrozenFoundationEvaluation =
        evaluate(fullCompactPrompt)

    /** Apply one real forward/backward/AdamW update to LoRA tensors only. */
    fun trainStep(batch: List<LoraTrainingExample>): LoraStepMetrics {
        throw UnsupportedOperationException("This runtime has no LoRA training graph")
    }

    fun adapterStatus(): LoraAdapterStatus = LoraAdapterStatus.unavailable()

    /** Drop learned LoRA/optimizer state and reopen the exported initial checkpoint. */
    fun resetAdapter() = Unit

    fun runtimeStatus(): FoundationModelStatus

    override fun close() = Unit
}

data class LoraTrainingExample(
    val eventId: String,
    val prompt: String,
    val target: DoubleArray,
    val replayLabel: String,
) {
    init {
        require(eventId.isNotBlank())
        require(prompt.isNotBlank())
        require(target.size == Route.entries.size)
        require(target.all { it.isFinite() && it >= 0.0 })
        require(kotlin.math.abs(target.sum() - 1.0) <= 1e-9)
        require(replayLabel.isNotBlank())
    }

    fun deepCopy(): LoraTrainingExample = copy(target = target.copyOf())
}

data class LoraAdapterStatus(
    val updateIndex: Long,
    val checksum: String,
    val l2Norm: Double,
    val trainableParameters: Int,
    val trainableTensors: Int,
    val checkpointGeneration: Long,
) {
    init {
        require(updateIndex >= 0L)
        require(l2Norm.isFinite() && l2Norm >= 0.0)
        require(trainableParameters >= 0)
        require(trainableTensors >= 0)
        require(checkpointGeneration >= 0L)
    }

    companion object {
        const val EXPECTED_PARAMETERS = 172_032
        const val EXPECTED_TENSORS = 48

        fun unavailable() = LoraAdapterStatus(
            updateIndex = 0L,
            checksum = "unavailable",
            l2Norm = 0.0,
            trainableParameters = 0,
            trainableTensors = 0,
            checkpointGeneration = 0L,
        )
    }
}

data class LoraStepMetrics(
    val updateIndex: Long,
    val lossBefore: Double,
    val lossAfter: Double,
    val adapterNormBefore: Double,
    val adapterNormAfter: Double,
    val checksumBefore: String,
    val checksumAfter: String,
    val batchSize: Int,
    val durationMillis: Double,
) {
    init {
        require(updateIndex > 0L)
        require(lossBefore.isFinite() && lossAfter.isFinite())
        require(adapterNormBefore.isFinite() && adapterNormAfter.isFinite())
        require(batchSize > 0)
        require(durationMillis.isFinite() && durationMillis >= 0.0)
        require(checksumBefore != checksumAfter) {
            "A reported optimizer step must change the persisted LoRA checkpoint"
        }
    }
}
