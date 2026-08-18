package ai.onlinesdft.router.model

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/** Fast test double for the production OrtTrainingSession boundary. */
internal class DeterministicFrozenFoundationRuntime(
    studentProbabilities: FloatArray = floatArrayOf(0.2f, 0.5f, 0.3f),
    teacherProbabilities: FloatArray = floatArrayOf(0.6f, 0.3f, 0.1f),
    private val available: Boolean = true,
    private val modelId: String = LiquidOrtFoundationModel.MODEL_ID,
) : FrozenFoundationRuntime {
    private val studentBacking = studentProbabilities.copyOf()
    private val teacherBacking = teacherProbabilities.copyOf()
    private val promptBias = mutableMapOf<String, DoubleArray>()
    val evaluatedPrompts = mutableListOf<String>()
    var preloadCalls = 0
        private set
    var closed = false
        private set
    private var updates = 0L

    val studentEvaluationCount: Int
        get() = evaluatedPrompts.count { it.startsWith("student:") }
    val teacherEvaluationCount: Int
        get() = evaluatedPrompts.count { it.startsWith("base:") && "Observed callback:" in it }

    override fun preload() { preloadCalls += 1 }

    override fun evaluate(fullCompactPrompt: String): FrozenFoundationEvaluation {
        check(available)
        evaluatedPrompts += "base:$fullCompactPrompt"
        val probabilities = if ("Observed callback:" in fullCompactPrompt) teacherBacking else studentBacking
        return result(probabilities, fullCompactPrompt)
    }

    override fun evaluateStudent(fullCompactPrompt: String): FrozenFoundationEvaluation {
        check(available)
        evaluatedPrompts += "student:$fullCompactPrompt"
        val base = DoubleArray(3) { studentBacking[it].toDouble() }
        val bias = promptBias.getOrPut(key(fullCompactPrompt)) { DoubleArray(3) }
        val logits = DoubleArray(3) { ln(base[it]) + bias[it] }
        val max = logits.maxOrNull()!!
        val exp = DoubleArray(3) { exp(logits[it] - max) }
        val total = exp.sum()
        return result(FloatArray(3) { (exp[it] / total).toFloat() }, fullCompactPrompt)
    }

    override fun trainStep(batch: List<LoraTrainingExample>): LoraStepMetrics {
        val beforeStatus = adapterStatus()
        val beforeLoss = meanLoss(batch)
        batch.forEach { row ->
            val prediction = evaluateStudent(row.prompt).actionProbabilitiesFp64
            val bias = promptBias.getOrPut(key(row.prompt)) { DoubleArray(3) }
            repeat(3) { index -> bias[index] += 0.8 * (row.target[index] - prediction[index]) }
        }
        updates += 1L
        val afterLoss = meanLoss(batch)
        val afterStatus = adapterStatus()
        return LoraStepMetrics(
            updates,
            beforeLoss,
            afterLoss,
            beforeStatus.l2Norm,
            afterStatus.l2Norm,
            beforeStatus.checksum,
            afterStatus.checksum,
            batch.size,
            1.0,
        )
    }

    override fun adapterStatus(): LoraAdapterStatus = LoraAdapterStatus(
        updateIndex = updates,
        checksum = "fake-lora-$updates-${promptBias.values.sumOf { it.sum() }.toBits()}",
        l2Norm = sqrt(promptBias.values.sumOf { row -> row.sumOf { it * it } }),
        trainableParameters = LoraAdapterStatus.EXPECTED_PARAMETERS,
        trainableTensors = LoraAdapterStatus.EXPECTED_TENSORS,
        checkpointGeneration = updates,
    )

    override fun resetAdapter() {
        promptBias.clear()
        updates = 0L
    }

    override fun runtimeStatus() = FoundationModelStatus(
        modelId = modelId,
        precision = "fp32-test",
        phase = if (available) FoundationModelPhase.READY else FoundationModelPhase.ERROR,
        lastError = if (available) null else "foundation unavailable",
    )

    override fun close() { closed = true }

    fun overwriteStudentBacking(values: FloatArray) {
        require(values.size == studentBacking.size)
        values.copyInto(studentBacking)
    }

    private fun meanLoss(batch: List<LoraTrainingExample>): Double = batch.sumOf { row ->
        val predicted = evaluateStudent(row.prompt).actionProbabilitiesFp64
        -row.target.indices.sumOf { row.target[it] * ln(predicted[it].coerceAtLeast(1e-8)) }
    } / batch.size

    private fun result(probabilities: FloatArray, prompt: String): FrozenFoundationEvaluation {
        val logits = FloatArray(3) { ln(probabilities[it].toDouble()).toFloat() }
        return FrozenFoundationEvaluation(
            actionLogits = logits,
            actionProbabilities = probabilities.copyOf(),
            actionProbabilitiesFp64 = frozenActionSoftmaxFp64(logits),
            promptTokens = prompt.length.toLong(),
            latencyMillis = 2.0,
        )
    }

    private fun key(prompt: String): String = Regex("This is a ([\\w-]+) notification")
        .find(prompt)?.groupValues?.get(1) ?: prompt
}
