package ai.onlinesdft.router.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtTrainingSession
import ai.onlinesdft.router.lfm.LfmByteLevelBpeTokenizer
import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.abs
import kotlin.math.exp
import org.json.JSONObject

internal fun frozenActionSoftmaxFp64(logits: FloatArray): DoubleArray {
    require(logits.size == 3)
    logits.forEach { require(it.isFinite()) }
    val maximum = logits.maxOrNull()!!.toDouble()
    val exponentials = DoubleArray(logits.size) { exp(logits[it].toDouble() - maximum) }
    val sum = exponentials.sum()
    require(sum.isFinite() && sum > 0.0)
    return DoubleArray(logits.size) { exponentials[it] / sum }
}

internal fun frozenActionSoftmax(logits: FloatArray): FloatArray =
    frozenActionSoftmaxFp64(logits).let { values ->
        FloatArray(values.size) { index -> values[index].toFloat() }
    }

class LiquidOrtFoundationException(message: String, cause: Throwable? = null) :
    IllegalStateException(message, cause)

/**
 * Preserve the beginning and end of the rendered interaction when the mobile
 * graph's fixed sequence budget is exceeded. The prefix contains the system
 * instruction and notification title; the suffix contains semantic metadata,
 * callback evidence (for teacher prompts), and the final Route cue.
 */
internal fun fitPromptTokenBudget(
    encoded: IntArray,
    marker: IntArray,
    maxSequenceLength: Int,
): LongArray {
    require(maxSequenceLength > 2)
    require(encoded.isNotEmpty())
    if (encoded.size <= maxSequenceLength) {
        return LongArray(encoded.size) { encoded[it].toLong() }
    }
    require(marker.isNotEmpty() && marker.size <= maxSequenceLength - 2)
    val contentBudget = maxSequenceLength - marker.size
    val prefixSize = (contentBudget * 3 / 5).coerceIn(1, contentBudget - 1)
    val suffixSize = contentBudget - prefixSize
    return LongArray(maxSequenceLength).also { fitted ->
        for (index in 0 until prefixSize) fitted[index] = encoded[index].toLong()
        for (index in marker.indices) fitted[prefixSize + index] = marker[index].toLong()
        val suffixOffset = encoded.size - suffixSize
        for (index in 0 until suffixSize) {
            fitted[prefixSize + marker.size + index] = encoded[suffixOffset + index].toLong()
        }
    }
}

/**
 * Real ONNX Runtime Training implementation for rank-4 LoRA on LFM2.5-230M.
 *
 * The live [OrtTrainingSession] owns only the LoRA tensors and AdamW state as
 * mutable parameters. A separate zero-adapter inference graph is opened only
 * while a fixed-base or hindsight-teacher evaluation is running.
 */
class LiquidOrtFoundationModel(
    context: Context,
    private val artifactDirectory: File = provisioningDirectory(context),
    private val checkpointRoot: File = File(context.filesDir, "model/lora-ort-v1"),
    private val onStatus: (FoundationModelStatus) -> Unit = {},
) : FrozenFoundationRuntime {
    private val environment = OrtEnvironment.getEnvironment()
    private val operationLock = ReentrantLock()
    private val preloadScheduled = AtomicBoolean(false)
    private val preloadExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "lfm-lora-preload").apply { isDaemon = true }
    }
    private val currentStatus = AtomicReference(
        FoundationModelStatus(modelId = MODEL_ID, precision = PRECISION),
    )

    @Volatile private var closed = false
    private var trainingSession: OrtTrainingSession? = null
    private var tokenizer: LfmByteLevelBpeTokenizer? = null
    private var manifest: LoraManifest? = null
    private var adapter = LoraAdapterStatus.unavailable()

    override fun preload() {
        if (closed || runtimeStatus().isReady) return
        if (!preloadScheduled.compareAndSet(false, true)) return
        runCatching {
            preloadExecutor.execute {
                try {
                    operationLock.withLock { loadLocked() }
                } catch (error: Throwable) {
                    recordFailure("preload LoRA training bundle", error)
                    Log.e(TAG, "LoRA preload failed", error)
                } finally {
                    preloadScheduled.set(false)
                }
            }
        }.onFailure { error ->
            preloadScheduled.set(false)
            recordFailure("schedule LoRA preload", error)
        }
    }

    /** Fixed base/teacher evaluation with the exported zero adapter. */
    override fun evaluate(fullCompactPrompt: String): FrozenFoundationEvaluation =
        operationLock.withLock {
            try {
                loadLocked()
                val loadedManifest = requireNotNull(manifest)
                val tokenIds = tokenize(fullCompactPrompt)
                val started = System.nanoTime()
                val options = sessionOptions()
                try {
                    environment.createSession(resolve(loadedManifest.baseModelPath).path, options).use { session ->
                        val output = runBase(session, loadedManifest, tokenIds)
                        return@withLock evaluation(output, tokenIds.size, started)
                    }
                } finally {
                    options.close()
                }
            } catch (error: Throwable) {
                throw recordFailure("evaluate fixed-base Liquid prompt", error)
            }
        }

    /** Current LoRA student evaluation through the training session's eval graph. */
    override fun evaluateStudent(fullCompactPrompt: String): FrozenFoundationEvaluation =
        operationLock.withLock {
            try {
                loadLocked()
                val tokenIds = tokenize(fullCompactPrompt)
                val started = System.nanoTime()
                val output = evalTrainingSession(listOf(tokenIds), listOf(UNIFORM_TARGET))
                adapter = adapter.copy(l2Norm = output.adapterNorm)
                evaluation(output, tokenIds.size, started)
            } catch (error: Throwable) {
                throw recordFailure("evaluate LoRA student prompt", error)
            }
        }

    override fun trainStep(batch: List<LoraTrainingExample>): LoraStepMetrics =
        operationLock.withLock {
            require(batch.isNotEmpty() && batch.size <= OnlineSdftLearner.MAX_BATCH_SIZE)
            try {
                loadLocked()
                val tokenRows = batch.map { tokenize(it.prompt) }
                val targets = batch.map { row -> FloatArray(3) { row.target[it].toFloat() } }
                val started = System.nanoTime()
                val before = evalTrainingSession(tokenRows, targets)
                val checksumBefore = adapter.checksum
                val session = requireNotNull(trainingSession)
                createFeeds(tokenRows, targets, requireNotNull(manifest).maxSequenceLength).use { feeds ->
                    session.trainStep(feeds.values).use { outputs ->
                        scalar(outputs.required(LOSS_OUTPUT), LOSS_OUTPUT)
                    }
                }
                session.optimizerStep()
                session.lazyResetGrad()
                val after = evalTrainingSession(tokenRows, targets)
                val newIndex = Math.addExact(adapter.updateIndex, 1L)
                val receipt = commitCheckpointLocked(newIndex)
                adapter = LoraAdapterStatus(
                    updateIndex = newIndex,
                    checksum = receipt.checksum,
                    l2Norm = after.adapterNorm,
                    trainableParameters = requireNotNull(manifest).trainableParameters,
                    trainableTensors = requireNotNull(manifest).trainableTensors,
                    checkpointGeneration = receipt.generation,
                )
                require(checksumBefore != adapter.checksum) {
                    "ORT optimizer step did not change the persisted LoRA checkpoint"
                }
                LoraStepMetrics(
                    updateIndex = newIndex,
                    lossBefore = before.loss,
                    lossAfter = after.loss,
                    adapterNormBefore = before.adapterNorm,
                    adapterNormAfter = after.adapterNorm,
                    checksumBefore = checksumBefore,
                    checksumAfter = adapter.checksum,
                    batchSize = batch.size,
                    durationMillis = elapsedMillis(started),
                )
            } catch (error: Throwable) {
                throw recordFailure("run LoRA forward/backward/AdamW step", error)
            }
        }

    override fun adapterStatus(): LoraAdapterStatus = adapter

    override fun resetAdapter() = operationLock.withLock {
        trainingSession?.close()
        trainingSession = null
        checkpointSlots().forEach { slot ->
            if (slot.exists()) slot.delete()
            checkpointMeta(slot).let { if (it.exists()) it.delete() }
        }
        if (checkpointRoot.exists() && checkpointRoot.listFiles().isNullOrEmpty()) checkpointRoot.delete()
        adapter = LoraAdapterStatus.unavailable()
        loadLocked()
    }

    override fun runtimeStatus(): FoundationModelStatus = currentStatus.get()

    override fun close() = operationLock.withLock {
        if (closed) return@withLock
        closed = true
        trainingSession?.close()
        trainingSession = null
        preloadExecutor.shutdownNow()
    }

    private fun loadLocked() {
        check(!closed)
        if (trainingSession != null && tokenizer != null && manifest != null) return
        publishStatus(runtimeStatus().copy(phase = FoundationModelPhase.LOADING, lastError = null))
        require(environment.isTrainingEnabled) {
            "The app was not packaged with onnxruntime-training-android"
        }
        val loadedManifest = LoraManifest.read(File(artifactDirectory, MANIFEST_FILE))
        val bytes = verifyBundle(loadedManifest)
        val loadedTokenizer = LfmByteLevelBpeTokenizer.fromFile(
            resolve(loadedManifest.tokenizerPath),
        )
        validateTokenizer(loadedTokenizer, loadedManifest)
        val restored = latestCheckpoint()
        val checkpoint = restored?.directory ?: resolve(loadedManifest.initialCheckpointPath)
        val options = sessionOptions()
        val session = try {
            environment.createTrainingSession(
                checkpoint.path,
                resolve(loadedManifest.trainingModelPath).path,
                resolve(loadedManifest.evalModelPath).path,
                resolve(loadedManifest.optimizerModelPath).path,
                options,
            )
        } finally {
            options.close()
        }
        try {
            validateTrainingSession(session, loadedManifest)
            session.setLearningRate(loadedManifest.learningRate)
            OrtTrainingSession.setSeed(TRAINING_SEED)
            val probe = loadedManifest.probeTokenIds
            val output = evalTrainingSession(session, loadedManifest, listOf(probe), listOf(UNIFORM_TARGET))
            val generation = restored?.generation ?: 0L
            val updateIndex = restored?.updateIndex ?: 0L
            val checksum = restored?.checksum ?: sha256(checkpoint)
            adapter = LoraAdapterStatus(
                updateIndex,
                checksum,
                output.adapterNorm,
                loadedManifest.trainableParameters,
                loadedManifest.trainableTensors,
                generation,
            )
        } catch (error: Throwable) {
            session.close()
            throw error
        }
        manifest = loadedManifest
        tokenizer = loadedTokenizer
        trainingSession = session
        publishStatus(
            runtimeStatus().copy(
                modelId = loadedManifest.modelId,
                precision = loadedManifest.precision,
                phase = FoundationModelPhase.READY,
                downloadedBytes = bytes,
                totalBytes = bytes,
                lastError = null,
            ),
        )
    }

    private fun evalTrainingSession(
        tokenRows: List<LongArray>,
        targets: List<FloatArray>,
    ): EvalOutput = evalTrainingSession(
        requireNotNull(trainingSession),
        requireNotNull(manifest),
        tokenRows,
        targets,
    )

    private fun evalTrainingSession(
        session: OrtTrainingSession,
        manifest: LoraManifest,
        tokenRows: List<LongArray>,
        targets: List<FloatArray>,
    ): EvalOutput = createFeeds(tokenRows, targets, manifest.maxSequenceLength).use { feeds ->
        session.evalStep(feeds.values).use { outputs ->
            val logits = floatRows(outputs.required(manifest.logitsOutputName), manifest.logitsOutputName)
            val probabilities = floatRows(
                outputs.required(manifest.probabilitiesOutputName),
                manifest.probabilitiesOutputName,
            )
            require(logits.size == tokenRows.size && probabilities.size == tokenRows.size)
            EvalOutput(
                loss = scalar(outputs.required(LOSS_OUTPUT), LOSS_OUTPUT).toDouble(),
                logits = logits,
                probabilities = probabilities,
                adapterNorm = scalar(outputs.required(ADAPTER_NORM_OUTPUT), ADAPTER_NORM_OUTPUT).toDouble(),
            )
        }
    }

    private fun runBase(
        session: OrtSession,
        manifest: LoraManifest,
        tokenIds: LongArray,
    ): EvalOutput {
        val mask = LongArray(tokenIds.size) { 1L }
        val feeds = linkedMapOf(
            INPUT_IDS to int64Tensor(tokenIds, longArrayOf(1, tokenIds.size.toLong())),
            ATTENTION_MASK to int64Tensor(mask, longArrayOf(1, mask.size.toLong())),
        )
        try {
            session.run(feeds).use { outputs ->
                val logits = floatRows(outputs.required(manifest.logitsOutputName), manifest.logitsOutputName)
                val probabilities = floatRows(
                    outputs.required(manifest.probabilitiesOutputName),
                    manifest.probabilitiesOutputName,
                )
                return EvalOutput(0.0, logits, probabilities, 0.0)
            }
        } finally {
            feeds.values.forEach { it.close() }
        }
    }

    private fun evaluation(output: EvalOutput, promptTokens: Int, started: Long): FrozenFoundationEvaluation {
        val logits = output.logits.single()
        val probabilities = output.probabilities.single()
        val expected = frozenActionSoftmax(logits)
        require(probabilities.indices.all { abs(probabilities[it] - expected[it]) <= 2e-5f })
        val latency = elapsedMillis(started)
        publishStatus(
            runtimeStatus().copy(
                phase = FoundationModelPhase.READY,
                lastError = null,
                lastInferenceMillis = latency,
                lastPromptTokens = promptTokens.toLong(),
                lastCompletionTokens = 0L,
            ),
        )
        return FrozenFoundationEvaluation(
            actionLogits = logits,
            actionProbabilities = probabilities,
            actionProbabilitiesFp64 = frozenActionSoftmaxFp64(logits),
            promptTokens = promptTokens.toLong(),
            latencyMillis = latency,
        )
    }

    private fun createFeeds(
        tokenRows: List<LongArray>,
        targets: List<FloatArray>,
        maxSequenceLength: Int,
    ): TensorFeeds {
        require(tokenRows.isNotEmpty() && tokenRows.size == targets.size)
        val maxLength = tokenRows.maxOf { it.size }
        require(maxLength <= maxSequenceLength)
        val batch = tokenRows.size
        val ids = LongArray(batch * maxLength) { PAD_TOKEN_ID }
        val mask = LongArray(batch * maxLength)
        tokenRows.forEachIndexed { row, tokens ->
            val offset = row * maxLength + (maxLength - tokens.size)
            tokens.copyInto(ids, offset)
            repeat(tokens.size) { mask[offset + it] = 1L }
        }
        val targetValues = FloatArray(batch * 3)
        targets.forEachIndexed { row, values ->
            require(values.size == 3 && values.all { it.isFinite() && it >= 0f })
            values.copyInto(targetValues, row * 3)
        }
        return TensorFeeds(
            linkedMapOf(
                INPUT_IDS to int64Tensor(ids, longArrayOf(batch.toLong(), maxLength.toLong())),
                ATTENTION_MASK to int64Tensor(mask, longArrayOf(batch.toLong(), maxLength.toLong())),
                TARGETS_INPUT to floatTensor(targetValues, longArrayOf(batch.toLong(), 3L)),
            ),
        )
    }

    private fun tokenize(prompt: String): LongArray {
        val loaded = requireNotNull(tokenizer)
        val contract = requireNotNull(manifest)
        val encoded = loaded.encode(prompt, addSpecialTokens = false)
        require(encoded.isNotEmpty() && encoded.first().toLong() == contract.bosTokenId)
        if (encoded.size <= contract.maxSequenceLength) {
            return LongArray(encoded.size) { encoded[it].toLong() }
        }
        val marker = loaded.encode(TRUNCATION_MARKER, addSpecialTokens = false)
        val fitted = fitPromptTokenBudget(encoded, marker, contract.maxSequenceLength)
        Log.i(
            TAG,
            "PROMPT_TRUNCATED original_tokens=${encoded.size} " +
                "retained_tokens=${fitted.size} max_tokens=${contract.maxSequenceLength}",
        )
        return fitted
    }

    private fun validateTokenizer(tokenizer: LfmByteLevelBpeTokenizer, manifest: LoraManifest) {
        require(tokenizer.vocabularySize == manifest.tokenizerSize)
        require(tokenizer.bosTokenId.toLong() == manifest.bosTokenId)
        ACTION_CODES.forEachIndexed { index, code ->
            require(
                tokenizer.encode(code, addSpecialTokens = false)
                    .contentEquals(intArrayOf(manifest.actionTokenIds[index].toInt())),
            )
        }
    }

    private fun validateTrainingSession(session: OrtTrainingSession, manifest: LoraManifest) {
        val expectedInputs = setOf(INPUT_IDS, ATTENTION_MASK, TARGETS_INPUT)
        val expectedOutputs = setOf(
            LOSS_OUTPUT,
            manifest.logitsOutputName,
            manifest.probabilitiesOutputName,
            ADAPTER_NORM_OUTPUT,
        )
        require(session.trainInputNames == expectedInputs)
        require(session.evalInputNames == expectedInputs)
        require(session.trainOutputNames == expectedOutputs)
        require(session.evalOutputNames == expectedOutputs)
    }

    private fun commitCheckpointLocked(updateIndex: Long): CheckpointReceipt {
        val session = requireNotNull(trainingSession)
        val generation = Math.addExact(adapter.checkpointGeneration, 1L)
        require(checkpointRoot.isDirectory || checkpointRoot.mkdirs())
        val destination = checkpointSlots()[(generation and 1L).toInt()]
        val pending = File(checkpointRoot, ".pending-$generation")
        val pendingMeta = checkpointMeta(pending)
        if (pending.exists()) pending.delete()
        if (pendingMeta.exists()) pendingMeta.delete()
        // The opposite slot still contains the newest committed checkpoint.
        // Reclaim this stale destination before writing the pending checkpoint
        // so a 919 MB LFM state never requires three simultaneous copies.
        val destinationMeta = checkpointMeta(destination)
        if (destination.exists()) destination.delete()
        if (destinationMeta.exists()) destinationMeta.delete()
        try {
            session.addProperty(UPDATE_PROPERTY, updateIndex.toInt())
            session.addProperty(GENERATION_PROPERTY, generation.toInt())
            session.saveCheckpoint(pending.toPath(), true)
            val checksum = sha256(pending)
            pendingMeta.writeText(
                JSONObject()
                    .put("generation", generation)
                    .put("update_index", updateIndex)
                    .put("checkpoint_sha256", checksum)
                    .toString(),
            )
            try {
                Files.move(pending.toPath(), destination.toPath(), StandardCopyOption.ATOMIC_MOVE)
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(pending.toPath(), destination.toPath())
            }
            try {
                Files.move(pendingMeta.toPath(), destinationMeta.toPath(), StandardCopyOption.ATOMIC_MOVE)
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(pendingMeta.toPath(), destinationMeta.toPath())
            }
            require(readCheckpoint(destination)?.checksum == checksum)
            return CheckpointReceipt(generation, updateIndex, checksum)
        } finally {
            if (pending.exists()) pending.delete()
            if (pendingMeta.exists()) pendingMeta.delete()
        }
    }

    private fun latestCheckpoint(): CheckpointReceipt? =
        checkpointSlots().mapNotNull(::readCheckpoint).maxByOrNull { it.generation }

    private fun readCheckpoint(directory: File): CheckpointReceipt? = runCatching {
        if (!directory.isFile) return@runCatching null
        val meta = JSONObject(checkpointMeta(directory).readText())
        val receipt = CheckpointReceipt(
            meta.getLong("generation"),
            meta.getLong("update_index"),
            meta.getString("checkpoint_sha256"),
            directory,
        )
        require(receipt.checksum.matches(Regex("[0-9a-f]{64}")))
        require(sha256(directory) == receipt.checksum)
        receipt
    }.getOrNull()

    private fun checkpointSlots() = listOf(
        File(checkpointRoot, "checkpoint.0"),
        File(checkpointRoot, "checkpoint.1"),
    )

    private fun checkpointMeta(checkpoint: File) = File("${checkpoint.path}.$META_FILE")

    private fun verifyBundle(manifest: LoraManifest): Long {
        var total = 0L
        manifest.artifacts.forEach { record ->
            val file = resolve(record.path)
            require(file.isFile)
            require(file.length() == record.bytes)
            require(sha256(file) == record.sha256)
            total = Math.addExact(total, record.bytes)
        }
        return total
    }

    private fun resolve(relative: String): File {
        require(relative.isNotBlank() && !File(relative).isAbsolute)
        require(relative.split('/', '\\').none { it.isEmpty() || it == ".." })
        val root = artifactDirectory.canonicalFile
        val resolved = File(root, relative).canonicalFile
        require(resolved.toPath().startsWith(root.toPath()))
        return resolved
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { stream ->
            val buffer = ByteArray(8 * 1024 * 1024)
            while (true) {
                val read = stream.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(Locale.US, it) }
    }

    private fun sessionOptions() = OrtSession.SessionOptions().apply {
        setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        setInterOpNumThreads(1)
        setIntraOpNumThreads(minOf(Runtime.getRuntime().availableProcessors(), 4))
    }

    private fun int64Tensor(values: LongArray, shape: LongArray): OnnxTensor {
        val buffer = ByteBuffer.allocateDirect(values.size * java.lang.Long.BYTES)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        buffer.put(values).rewind()
        return OnnxTensor.createTensor(environment, buffer, shape)
    }

    private fun floatTensor(values: FloatArray, shape: LongArray): OnnxTensor {
        val buffer = ByteBuffer.allocateDirect(values.size * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        buffer.put(values).rewind()
        return OnnxTensor.createTensor(environment, buffer, shape)
    }

    private fun floatRows(value: OnnxValue, name: String): List<FloatArray> {
        val outer = value.value
        require(outer is Array<*>) { "$name must have shape [batch,3]" }
        return outer.map { row ->
            require(row is FloatArray && row.size == 3)
            row.copyOf()
        }
    }

    private fun scalar(value: OnnxValue, name: String): Float = when (val raw = value.value) {
        is Number -> raw.toFloat()
        is FloatArray -> raw.single()
        else -> error("$name must be a float scalar")
    }.also { require(it.isFinite()) }

    private fun OrtSession.Result.required(name: String): OnnxValue =
        get(name).orElseThrow { LiquidOrtFoundationException("Graph omitted $name") }

    private fun publishStatus(status: FoundationModelStatus) {
        currentStatus.set(status)
        onStatus(status)
    }

    private fun recordFailure(operation: String, error: Throwable): LiquidOrtFoundationException {
        val wrapped = if (error is LiquidOrtFoundationException) error else {
            LiquidOrtFoundationException(
                "Failed to $operation: ${error.message ?: error.javaClass.simpleName}",
                error,
            )
        }
        if (!closed) {
            publishStatus(
                runtimeStatus().copy(
                    phase = FoundationModelPhase.ERROR,
                    lastError = wrapped.message?.take(500),
                ),
            )
        }
        return wrapped
    }

    private fun elapsedMillis(started: Long) = (System.nanoTime() - started) / 1_000_000.0

    private data class EvalOutput(
        val loss: Double,
        val logits: List<FloatArray>,
        val probabilities: List<FloatArray>,
        val adapterNorm: Double,
    )

    private class TensorFeeds(val values: LinkedHashMap<String, OnnxTensor>) : AutoCloseable {
        override fun close() = values.values.forEach { runCatching { it.close() } }
    }

    private data class ArtifactRecord(val path: String, val bytes: Long, val sha256: String)

    private data class CheckpointReceipt(
        val generation: Long,
        val updateIndex: Long,
        val checksum: String,
        val directory: File = File("."),
    )

    private data class LoraManifest(
        val modelId: String,
        val precision: String,
        val tokenizerPath: String,
        val baseModelPath: String,
        val trainingModelPath: String,
        val evalModelPath: String,
        val optimizerModelPath: String,
        val initialCheckpointPath: String,
        val tokenizerSize: Int,
        val bosTokenId: Long,
        val maxSequenceLength: Int,
        val actionTokenIds: LongArray,
        val logitsOutputName: String,
        val probabilitiesOutputName: String,
        val trainableParameters: Int,
        val trainableTensors: Int,
        val learningRate: Float,
        val probeTokenIds: LongArray,
        val artifacts: List<ArtifactRecord>,
    ) {
        companion object {
            fun read(file: File): LoraManifest {
                val root = JSONObject(file.readText())
                require(root.getString("schema") == MANIFEST_SCHEMA)
                require(root.getInt("schema_version") == MANIFEST_VERSION)
                val model = root.getJSONObject("model")
                require(model.getString("id") == MODEL_ID)
                require(model.getString("revision") == MODEL_REVISION)
                require(model.getString("precision") == PRECISION)
                require(model.getInt("trainable_parameters") == LoraAdapterStatus.EXPECTED_PARAMETERS)
                require(model.getInt("trainable_tensors") == LoraAdapterStatus.EXPECTED_TENSORS)
                require(model.getString("optimizer") == "AdamW")
                val lora = model.getJSONObject("lora")
                require(lora.getInt("rank") == 4 && lora.getInt("alpha") == 8)
                require(lora.getDouble("dropout") == 0.0)
                val graph = root.getJSONObject("graph_contract")
                require(graph.getInt("max_batch_size") == OnlineSdftLearner.MAX_BATCH_SIZE)
                require(graph.getBoolean("training_graph"))
                require(graph.getBoolean("soft_target_cross_entropy"))
                require(!graph.getBoolean("full_vocabulary_output"))
                val inputs = graph.getJSONObject("inputs")
                require(inputs.getString("input_ids") == INPUT_IDS)
                require(inputs.getString("attention_mask") == ATTENTION_MASK)
                require(inputs.getString("target_probabilities") == TARGETS_INPUT)
                val tokenizer = root.getJSONObject("tokenizer")
                val deploy = root.getJSONObject("deploy_contract")
                val outputs = graph.getJSONObject("outputs")
                require(outputs.getString("loss") == LOSS_OUTPUT)
                require(outputs.getString("adapter_l2_norm") == ADAPTER_NORM_OUTPUT)
                val runtime = root.getJSONObject("runtime")
                require(runtime.getString("onnxruntime_version") == ORT_VERSION)
                require(
                    runtime.getString("android_maven_artifact") ==
                        "com.microsoft.onnxruntime:onnxruntime-training-android:$ORT_VERSION",
                )
                val actionIdsJson = graph.getJSONArray("action_token_ids")
                val actionIds = LongArray(actionIdsJson.length()) { actionIdsJson.getLong(it) }
                require(actionIds.contentEquals(ACTION_TOKEN_IDS))
                val probeJson = root.getJSONObject("parity").getJSONArray("probe_token_ids")
                val probe = LongArray(probeJson.length()) { probeJson.getLong(it) }
                val records = root.getJSONArray("artifacts")
                val artifacts = List(records.length()) { index ->
                    val row = records.getJSONObject(index)
                    require(row.getBoolean("deploy") && row.getString("kind") == "file")
                    val record = ArtifactRecord(
                        row.getString("path"),
                        row.getLong("bytes"),
                        row.getString("sha256"),
                    )
                    require(record.bytes > 0L && record.sha256.matches(Regex("[0-9a-f]{64}")))
                    record
                }
                require(artifacts.map { it.path }.toSet().size == artifacts.size)
                return LoraManifest(
                    model.getString("id"),
                    model.getString("precision"),
                    deploy.getString("tokenizer_path"),
                    deploy.getString("base_model_path"),
                    deploy.getString("training_model_path"),
                    deploy.getString("eval_model_path"),
                    deploy.getString("optimizer_model_path"),
                    deploy.getString("initial_checkpoint_path"),
                    tokenizer.getInt("tokenizer_size"),
                    tokenizer.getLong("bos_token_id"),
                    graph.getInt("max_sequence_length"),
                    actionIds,
                    outputs.getString("action_logits"),
                    outputs.getString("action_probabilities"),
                    model.getInt("trainable_parameters"),
                    model.getInt("trainable_tensors"),
                    model.getDouble("learning_rate").toFloat(),
                    probe,
                    artifacts,
                )
            }
        }
    }

    companion object {
        const val MODEL_ID = "LiquidAI/LFM2.5-230M"
        const val MODEL_REVISION = "13a53837c4906b4f7405932532ba85d182bb013b"
        const val PRECISION = "fp32"
        const val MANIFEST_SCHEMA = "ai.onlinesdft.lfm_lora_ort_bundle"
        const val MANIFEST_VERSION = 1
        const val ORT_VERSION = "1.19.2"
        const val PAD_TOKEN_ID = 0L
        const val BOS_TOKEN_ID = 1L
        val ACTION_TOKEN_IDS = longArrayOf(542, 543, 544)
        private val ACTION_CODES = arrayOf("A", "B", "C")
        private val UNIFORM_TARGET = floatArrayOf(1f / 3f, 1f / 3f, 1f / 3f)
        private const val INPUT_IDS = "input_ids"
        private const val ATTENTION_MASK = "attention_mask"
        private const val TARGETS_INPUT = "target_probabilities"
        private const val LOSS_OUTPUT = "loss"
        private const val ADAPTER_NORM_OUTPUT = "adapter_l2_norm"
        private const val UPDATE_PROPERTY = "sdft_update_index"
        private const val GENERATION_PROPERTY = "sdft_checkpoint_generation"
        private const val META_FILE = "sdft_checkpoint.json"
        private const val MANIFEST_FILE = "manifest.json"
        private const val TRUNCATION_MARKER = "\n[notification content truncated]\n"
        private const val DIRECTORY_NAME = "lfm_lora"
        private const val TAG = "LiquidOrtLoRA"
        private const val TRAINING_SEED = 57L

        fun provisioningDirectory(context: Context): File =
            File(context.filesDir, DIRECTORY_NAME)
    }
}
