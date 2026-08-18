package ai.onlinesdft.router.telemetry

import ai.onlinesdft.router.demo.PersonaScenarioCatalog
import ai.onlinesdft.router.model.DecisionSnapshot
import ai.onlinesdft.router.model.ExecutionConstraint
import ai.onlinesdft.router.model.FactualFeedback
import ai.onlinesdft.router.model.Route
import ai.onlinesdft.router.model.TrainingMetrics
import android.os.Debug
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.time.Instant
import java.util.Properties
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.ln

enum class BackendState { LOCAL_ONLY, CONNECTING, CONNECTED, QUEUED_OFFLINE }

/** Strict JSON payloads shared by transport and local JVM contract tests. */
object TelemetryPayloads {
    fun runCreate(idempotencyKey: String, startedAtMillis: Long): String = jsonObject(
        "idempotency_key" to idempotencyKey,
        "started_at" to Instant.ofEpochMilli(startedAtMillis).toString(),
        "device_id_hash" to "android-emulator-demo-v1",
        "app_version" to "0.5.0-lora-debug",
        "model_id" to "LiquidAI/LFM2.5-230M@13a53837+lora-r4-v1",
        "base_adapter_version" to "lora-r4-zero-v1",
        "preference_profile_id" to "alex-v1",
        "evaluation_suite_id" to "alex-workday-v1",
        "mode" to "hybrid",
        "config" to RawJson(
            jsonObject(
                "foundation_model" to "LiquidAI/LFM2.5-230M",
                "foundation_revision" to
                    "13a53837c4906b4f7405932532ba85d182bb013b",
                "foundation_precision" to "FP32",
                "foundation_quantization" to "none",
                "foundation_safetensors_sha256" to
                    "f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd",
                "foundation_weights_trainable" to false,
                "foundation_output" to "next-token-logits-A-B-C",
                "student_update_mode" to "onnxruntime-training-lora",
                "lora_rank" to 4,
                "lora_alpha" to 8,
                "lora_target_modules" to RawJson("[\"q_proj\",\"k_proj\",\"v_proj\",\"self_attn.out_proj\"]"),
                "lora_layers" to RawJson("[2,4,6,8,10,12]"),
                "trainable_tensors" to 48,
                "trainable_parameters" to 172_032,
                "initialization" to "PEFT default: A random, B zero",
                "reliable_teacher_weight" to 0.05,
                "reliable_decision_weight" to 0.05,
                "reliable_behavior_weight" to 0.90,
                "ambiguous_teacher_weight" to 0.0,
                "ambiguous_decision_weight" to 1.0,
                "ambiguous_behavior_weight" to 0.0,
                "ambiguous_projection" to "causal_support",
                "teacher_state" to "fixed_initial",
                "teacher_reasoning_tokens" to 0,
                "replay_capacity" to 32,
                "warmup_examples" to 4,
                "max_batch_size" to 1,
                "optimizer_steps" to 1,
                "optimizer" to "ORT-AdamW",
                "learning_rate" to 0.001,
                "l2" to 0.0,
                "runtime" to "onnxruntime-training-android-1.19.2",
                "replay_sampler" to "newest-only-android-arm-safe",
                "ambiguous_replay_group_weight" to 0.05,
                "checkpoint_storage" to
                    "ORT-checkpoint+AdamW two-slot; replay+PCG64 companion two-slot",
                "legacy_backend_adapter_field_semantics" to
                    "adapter_before/adapter_after=SHA256(ORT checkpoint);" +
                    "adapter_norm=training-graph LoRA tensor L2",
                "lab_stream_version" to PersonaScenarioCatalog.STREAM_VERSION,
            ),
        ),
    )

    fun decision(
        decision: DecisionSnapshot,
        sequence: Int,
        labMode: Boolean,
        routeExecution: String? = null,
    ): String =
        jsonObject(
            "event_id" to decision.context.eventId,
            "sequence" to sequence,
            "committed_at" to Instant.ofEpochMilli(decision.decidedAtMillis).toString(),
            // The three demo publishers are intentionally non-private provenance:
            // showing their package names proves the source is a separate installed
            // app. Every other package remains hashed exactly as before.
            "source_package" to telemetrySourcePackage(decision.context.packageName),
            "category" to decision.context.category,
            "model_version" to
                "LFM2.5-230M-FP32+lora-${decision.adapterChecksum}",
            "student_probs" to routeProbabilities(decision.probabilities),
            "behavior_probs" to if (decision.executionConstraint != ExecutionConstraint.NONE) {
                routeProbabilities(
                    FloatArray(3) { index ->
                        if (index == decision.chosenRoute.ordinal) 1f else 0f
                    },
                )
            } else {
                null
            },
            "chosen_route" to decision.chosenRoute.name,
            "inference_ms" to decision.inferenceLatencyMillis,
            "evaluation_case_id" to if (labMode) decision.context.caseId else null,
            "context" to RawJson(
                jsonObject(
                    "package_hash" to redactedHash(decision.context.packageName),
                    "title_hash" to redactedHash(decision.context.title),
                    "title_length" to decision.context.title.length,
                    "body_hash" to redactedHash(decision.context.body),
                    "body_length" to decision.context.body.length,
                    "importance" to decision.context.importance,
                    "regime" to decision.context.regime.name.lowercase(),
                    "lora_checkpoint" to decision.checkpointIndex,
                    "run_epoch" to decision.runEpoch,
                    "base_route" to decision.baseRoute.name,
                    "base_recommended_route" to decision.baseRecommendedRoute.name,
                    "model_recommended_route" to decision.recommendedRoute.name,
                    "foundation_model_id" to decision.foundationModelId,
                    "foundation_precision" to decision.foundationPrecision,
                    "foundation_quantization" to "none",
                    "foundation_route" to decision.foundationRoute.name,
                    "foundation_available" to decision.foundationAvailable,
                    "foundation_inference_ms" to decision.foundationInferenceLatencyMillis,
                    "foundation_prompt_tokens" to decision.foundationPromptTokens,
                    "foundation_completion_tokens" to decision.foundationCompletionTokens,
                    "foundation_tokens_per_second" to decision.foundationTokensPerSecond,
                    "foundation_probs" to if (decision.foundationAvailable) {
                        routeProbabilities(decision.foundationProbabilities)
                    } else {
                        null
                    },
                    "foundation_raw_probabilities_exposed" to decision.foundationAvailable,
                    "baseline_available" to decision.baselineAvailable,
                    "execution_constraint" to decision.executionConstraint.wireName,
                    "route_execution" to (routeExecution ?: when {
                        decision.executionConstraint == ExecutionConstraint.MODEL_UNAVAILABLE ->
                            "model_unavailable_pass_through"
                        decision.executionConstraint != ExecutionConstraint.NONE ->
                            "platform_protected_pass_through"
                        decision.chosenRoute == Route.INTERRUPT -> "left_visible"
                        else -> "cancellation_requested"
                    }),
                    "clearable" to decision.context.isClearable,
                    "ongoing" to decision.context.isOngoing,
                    "foreground_service" to decision.context.isForegroundService,
                    "call" to decision.context.isCall,
                    "media" to decision.context.isMedia,
                    "group_summary" to decision.context.isGroupSummary,
                    "no_clear" to decision.context.isNoClear,
                    "digest_available" to decision.context.canPublishDigest,
                ),
            ),
        )

    fun feedback(feedback: FactualFeedback): String = jsonObject(
        "event_id" to feedback.eventId,
        "observed_at" to Instant.ofEpochMilli(feedback.observedAtMillis).toString(),
        "action_taken" to feedback.executedRoute.name,
        "outcome" to if (feedback.source.name == "EXPLICIT_USER_CORRECTION") {
            "EXPLICIT_USER_CORRECTION"
        } else {
            feedback.outcome.name
        },
        "observed_user_selection" to (feedback.observedSelection?.name ?: "UNKNOWN"),
        "delay_ms" to feedback.delayMinutes.toLong() * 60_000L,
        "feedback_source" to if (feedback.source.name == "EXPLICIT_USER_CORRECTION") {
            "explicit_user_correction"
        } else {
            "notification_surface"
        },
    )

    fun trainingUpdate(metrics: TrainingMetrics, endedAtMillis: Long): String {
        val heapMegabytes = runCatching { Debug.getPss() / 1024.0 }.getOrElse {
            (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) /
                (1024.0 * 1024.0)
        }
        return jsonObject(
            "update_id" to "android-${metrics.updateIndex}-${metrics.eventId}",
            "update_index" to metrics.updateIndex,
            "trigger_event_ids" to RawJson("[\"${escapeJson(metrics.eventId)}\"]"),
            // The strict backend retains these historical field names. Values
            // are SHA-256 checksums of committed ORT LoRA checkpoints.
            "adapter_before" to metrics.checksumBefore,
            "adapter_after" to metrics.checksumAfter,
            "started_at" to Instant.ofEpochMilli(
                endedAtMillis - metrics.durationMillis.toLong().coerceAtLeast(0),
            ).toString(),
            "ended_at" to Instant.ofEpochMilli(endedAtMillis).toString(),
            "status" to "succeeded",
            "metrics" to RawJson(
                jsonObject(
                    "loss_pre" to metrics.lossBefore,
                    "loss_post" to metrics.lossAfter,
                    "learning_rate" to 0.001,
                    "batch_size" to metrics.batchSize,
                    "replay_size" to metrics.replaySize,
                    "optimizer_steps" to metrics.optimizerSteps,
                    "lessons_in_batch" to metrics.trainingExamples,
                    "grad_norm" to metrics.gradientNorm,
                    "student_entropy" to entropy(metrics.sealedDecisionProbabilities),
                    "teacher_entropy" to entropy(metrics.teacherProbabilities),
                    "target_entropy" to entropy(metrics.fusedTarget),
                    "kl_target_student" to metrics.targetKlBefore,
                    "kl_target_student_post" to metrics.targetKlAfter,
                    // Legacy strict-backend slots; both now carry LoRA L2 norms.
                    "adapter_norm" to metrics.adapterNorm,
                    "adapter_delta_norm" to metrics.deltaNorm,
                    "callback_update_norm" to metrics.callbackUpdateNorm,
                    "optimizer_step_losses" to RawJson(
                        metrics.optimizerStepLosses.joinToString(
                            prefix = "[",
                            postfix = "]",
                        ),
                    ),
                    "optimizer_step_proofs" to RawJson(
                        optimizerStepProofs(metrics.optimizerStepProofs),
                    ),
                    "teacher_forward_ms" to metrics.teacherForwardLatencyMillis,
                    "backprop_ms" to metrics.durationMillis,
                    "peak_memory_mb" to heapMegabytes,
                    "battery_delta_pct" to null,
                    "thermal_status" to null,
                ),
            ),
            "error_message" to null,
        )
    }

    fun optimizerStepProofs(
        proofs: List<ai.onlinesdft.router.model.OptimizerStepProof>,
    ): String = proofs.joinToString(prefix = "[", postfix = "]", separator = ",") { proof ->
        jsonObject(
            "step_index" to proof.stepIndex,
            "update_index" to proof.updateIndex,
            "sampled_event_ids" to RawJson(
                proof.sampledEventIds.joinToString(
                    prefix = "[",
                    postfix = "]",
                    separator = ",",
                ) { eventId -> "\"${escapeJson(eventId)}\"" },
            ),
            "batch_size" to proof.batchSize,
            "loss_pre" to proof.lossBefore,
            "loss_post" to proof.lossAfter,
            "grad_norm" to proof.gradientNorm,
            "unclipped_update_norm" to proof.unclippedUpdateNorm,
            "applied_update_norm" to proof.appliedUpdateNorm,
        )
    }

    private fun entropy(probabilities: FloatArray): Double = probabilities.sumOf { value ->
        val probability = value.coerceAtLeast(1e-8f).toDouble()
        -probability * ln(probability)
    }

    private fun telemetrySourcePackage(packageName: String): String =
        if (packageName in DEMO_PUBLISHER_PACKAGES) {
            packageName
        } else {
            "sha256:${redactedHash(packageName)}"
        }

    private val DEMO_PUBLISHER_PACKAGES = setOf(
        "ai.onlinesdft.publisher.chat",
        "ai.onlinesdft.publisher.calendar",
        "ai.onlinesdft.publisher.mail",
    )
}

/**
 * App-private JSONL audit log with optional, best-effort remote export.
 *
 * Export is deliberately disabled by default: routing, feedback, learning,
 * evaluation, checkpointing, and reset never need a network peer. When export
 * is explicitly enabled, strict run-scoped records remain queued until remote
 * run creation succeeds. Reset archives the prior local session instead of
 * replaying it under the new identity.
 */
class TelemetryClient(
    filesDir: File,
    private val baseUrl: String = "http://10.0.2.2:8787",
    private val exportEnabled: Boolean = false,
    private val onState: (BackendState) -> Unit = {},
) {
    private val telemetryDir = File(filesDir, "telemetry")
    private val auditLog = File(telemetryDir, "audit-v1.jsonl")
    private val outbox = File(telemetryDir, "outbox-v3.jsonl")
    private val metadata = File(telemetryDir, "run-v3.properties")
    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "sdft-telemetry").apply { isDaemon = true }
    }
    private val lock = Any()
    private val meta = loadMetadata()
    @Volatile
    var runId: String = meta.getProperty("idempotency_key") ?: UUID.randomUUID().toString()
        private set
    @Volatile
    private var startedAtMillis = meta.getProperty("started_at_ms")?.toLongOrNull()
        ?: System.currentTimeMillis()
    private var serverRunId: String? = meta.getProperty("server_run_id")
    private val sequence = AtomicInteger(meta.getProperty("sequence")?.toIntOrNull() ?: 0)

    init {
        try {
            persistMetadata()
            synchronized(lock) {
                ensureAuditHeaderLocked()
            }
        } catch (_: Exception) {
            // Audit/export is deliberately non-critical. A later task retries
            // directory creation and header persistence.
        }
        if (exportEnabled) {
            publishState(BackendState.CONNECTING)
            executeSafely(::flushBlocking)
        } else {
            publishState(BackendState.LOCAL_ONLY)
        }
    }

    fun decision(
        decision: DecisionSnapshot,
        labMode: Boolean,
        routeExecution: String? = null,
    ) {
        executeSafely {
            val next = sequence.incrementAndGet()
            persistMetadata()
            enqueueBlocking(
                "decisions",
                TelemetryPayloads.decision(decision, next, labMode, routeExecution),
            )
        }
    }

    fun feedback(feedback: FactualFeedback) {
        executeSafely {
            enqueueBlocking("feedback", TelemetryPayloads.feedback(feedback))
        }
    }

    fun trainingUpdate(metrics: TrainingMetrics) {
        val endedAtMillis = System.currentTimeMillis()
        executeSafely {
            enqueueBlocking(
                "training-updates",
                TelemetryPayloads.trainingUpdate(metrics, endedAtMillis),
            )
        }
    }

    fun flush() {
        if (exportEnabled) {
            executeSafely(::flushBlocking)
        } else {
            publishState(BackendState.LOCAL_ONLY)
        }
    }

    /**
     * Starts a new local capture session without waiting for disk or network.
     *
     * Rotation is queued on the same FIFO executor as audit writes, so rows
     * accepted before this call stay in the old archive and rows accepted after
     * it use the new session. Optional export work can be slow or unavailable;
     * neither condition can block model reset.
     */
    fun startNewRun(): String {
        val newRunId = UUID.randomUUID().toString()
        val newStartedAtMillis = System.currentTimeMillis()
        if (!executeSafely { rotateRunBlocking(newRunId, newStartedAtMillis) }) {
            publishState(
                if (exportEnabled) BackendState.QUEUED_OFFLINE else BackendState.LOCAL_ONLY,
            )
        }
        return newRunId
    }

    private fun rotateRunBlocking(newRunId: String, newStartedAtMillis: Long) {
        synchronized(lock) {
            requireTelemetryDirectory()
            val archiveSuffix = "${System.currentTimeMillis()}-${runId.take(8)}"
            archiveForNewSession(auditLog, "audit-archived-$archiveSuffix.jsonl")
            archiveForNewSession(outbox, "outbox-archived-$archiveSuffix.jsonl")
            runId = newRunId
            startedAtMillis = newStartedAtMillis
            serverRunId = null
            sequence.set(0)
            meta.clear()
            persistMetadata()
            ensureAuditHeaderLocked()
        }
        publishState(if (exportEnabled) BackendState.CONNECTING else BackendState.LOCAL_ONLY)
    }

    private fun archiveForNewSession(active: File, archiveName: String) {
        requireTelemetryDirectory()
        if (!active.isFile || active.length() == 0L) {
            active.writeText("")
            return
        }
        val archived = File(telemetryDir, archiveName)
        if (!active.renameTo(archived)) {
            archived.writeText(active.readText())
            active.writeText("")
        }
    }

    private fun enqueueBlocking(kind: String, payload: String) {
        synchronized(lock) {
            requireTelemetryDirectory()
            ensureAuditHeaderLocked()
            val row = "$kind\t$payload\n"
            auditLog.appendText(row)
            if (exportEnabled) outbox.appendText(row)
        }
        if (exportEnabled) {
            publishState(BackendState.CONNECTING)
            flushBlocking()
        } else {
            publishState(BackendState.LOCAL_ONLY)
        }
    }

    private fun flushBlocking() {
        if (!exportEnabled) {
            publishState(BackendState.LOCAL_ONLY)
            return
        }
        val activeRun = serverRunId ?: createServerRun()
        if (activeRun == null) {
            publishState(BackendState.QUEUED_OFFLINE)
            return
        }
        val queued = synchronized(lock) {
            if (!outbox.isFile) emptyList() else outbox.readLines().filter { it.isNotBlank() }
        }
        if (queued.isEmpty()) {
            publishState(BackendState.CONNECTED)
            return
        }
        var sent = 0
        for (line in queued) {
            val separator = line.indexOf('\t')
            if (separator <= 0) {
                sent += 1
                continue
            }
            val kind = line.substring(0, separator)
            val payload = line.substring(separator + 1)
            val response = post("/api/v1/runs/$activeRun/$kind", payload)
            if (response == null) break
            sent += 1
        }
        synchronized(lock) {
            requireTelemetryDirectory()
            val current = if (outbox.isFile) outbox.readLines() else emptyList()
            val remaining = current.drop(sent)
            val pending = File(outbox.parentFile, "${outbox.name}.pending")
            pending.writeText(
                if (remaining.isEmpty()) "" else remaining.joinToString("\n", postfix = "\n"),
            )
            if (!pending.renameTo(outbox)) {
                outbox.writeText(pending.readText())
                pending.delete()
            }
        }
        publishState(
            if (sent == queued.size) BackendState.CONNECTED else BackendState.QUEUED_OFFLINE,
        )
    }

    private fun createServerRun(): String? {
        val response = post(
            "/api/v1/runs",
            TelemetryPayloads.runCreate(runId, startedAtMillis),
        ) ?: return null
        val parsed = RUN_ID_PATTERN.find(response)?.groupValues?.get(1) ?: return null
        serverRunId = parsed
        persistMetadata()
        return parsed
    }

    private fun post(endpoint: String, payload: String): String? = try {
        val connection = URL(baseUrl.trimEnd('/') + endpoint).openConnection() as HttpURLConnection
        connection.requestMethod = "POST"
        connection.connectTimeout = 1_500
        connection.readTimeout = 2_000
        connection.doOutput = true
        connection.setRequestProperty("Content-Type", "application/json")
        connection.outputStream.use { it.write(payload.toByteArray(Charsets.UTF_8)) }
        val responseCode = connection.responseCode
        val response = if (responseCode in 200..299) {
            connection.inputStream.bufferedReader().use { it.readText() }
        } else {
            null
        }
        connection.disconnect()
        response
    } catch (_: Exception) {
        null
    }

    private fun loadMetadata(): Properties = Properties().also { properties ->
        try {
            requireTelemetryDirectory()
            if (metadata.isFile) metadata.inputStream().use(properties::load)
        } catch (_: Exception) {
            // Missing/corrupt audit state starts a new local run. Telemetry must
            // never prevent the router runtime from being constructed.
        }
    }

    private fun persistMetadata() {
        synchronized(lock) {
            requireTelemetryDirectory()
            meta.setProperty("idempotency_key", runId)
            meta.setProperty("started_at_ms", startedAtMillis.toString())
            meta.setProperty("sequence", sequence.get().toString())
            serverRunId?.let { meta.setProperty("server_run_id", it) }
            val pending = File(metadata.parentFile, "${metadata.name}.pending")
            pending.outputStream().use { meta.store(it, "Online SDFT telemetry run") }
            if (!pending.renameTo(metadata)) {
                metadata.outputStream().use { meta.store(it, "Online SDFT telemetry run") }
                pending.delete()
            }
        }
    }

    /** Must be called while [lock] is held. */
    private fun ensureAuditHeaderLocked() {
        requireTelemetryDirectory()
        if (!auditLog.isFile || auditLog.length() == 0L) {
            auditLog.appendText(
                "run\t${TelemetryPayloads.runCreate(runId, startedAtMillis)}\n",
            )
        }
    }

    private fun requireTelemetryDirectory() {
        if (telemetryDir.isDirectory) return
        if (!telemetryDir.mkdirs() && !telemetryDir.isDirectory) {
            throw IOException("Unable to create app-private telemetry directory")
        }
    }

    /**
     * Contains all executor exceptions inside optional telemetry. This keeps a
     * removed/unwritable audit directory from reaching Android's uncaught-
     * exception handler and terminating the app process.
     */
    private fun executeSafely(block: () -> Unit): Boolean = try {
        executor.execute {
            try {
                block()
            } catch (_: Exception) {
                publishState(
                    if (exportEnabled) BackendState.QUEUED_OFFLINE else BackendState.LOCAL_ONLY,
                )
            }
        }
        true
    } catch (_: Exception) {
        publishState(
            if (exportEnabled) BackendState.QUEUED_OFFLINE else BackendState.LOCAL_ONLY,
        )
        false
    }

    private fun publishState(state: BackendState) {
        try {
            onState(state)
        } catch (_: Exception) {
            // UI observation is optional just like export.
        }
    }

    private companion object {
        val RUN_ID_PATTERN = Regex("\\\"run_id\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"")
    }
}
