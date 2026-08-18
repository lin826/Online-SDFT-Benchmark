package ai.onlinesdft.router.telemetry

import ai.onlinesdft.router.model.DecisionSnapshot
import ai.onlinesdft.router.model.FactualFeedback
import ai.onlinesdft.router.model.FeedbackSource
import ai.onlinesdft.router.model.ExecutionConstraint
import ai.onlinesdft.router.model.FeatureEncoder
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.OptimizerStepProof
import ai.onlinesdft.router.model.Outcome
import ai.onlinesdft.router.model.Regime
import ai.onlinesdft.router.model.Route
import ai.onlinesdft.router.model.TrainingMetrics
import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.net.InetAddress
import java.net.ServerSocket
import java.net.SocketTimeoutException
import java.util.Collections
import java.util.Properties
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class TelemetryPayloadsTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `run payload contains strict backend fields and aware timestamp`() {
        val payload = TelemetryPayloads.runCreate("idempotency-1234", 1_700_000_000_000L)
        requiredKeys(
            payload,
            "idempotency_key",
            "started_at",
            "device_id_hash",
            "app_version",
            "model_id",
            "base_adapter_version",
            "preference_profile_id",
            "evaluation_suite_id",
            "mode",
            "config",
        )
        assertTrue(payload.contains("2023-11-14T22:13:20Z"))
    }

    @Test
    fun `decision payload is redacted run child schema with normalized route object`() {
        val payload = TelemetryPayloads.decision(decision(), sequence = 3, labMode = true)
        requiredKeys(
            payload,
            "event_id",
            "sequence",
            "committed_at",
            "source_package",
            "category",
            "model_version",
            "student_probs",
            "chosen_route",
            "inference_ms",
            "evaluation_case_id",
            "context",
        )
        assertTrue(payload.contains("\"INTERRUPT\":0.2"))
        assertTrue(payload.contains("\"LATER\":0.3"))
        assertTrue(payload.contains("\"ARCHIVE\":0.5"))
        assertFalse(payload.contains("secret title"))
        assertFalse(payload.contains("private body"))
        assertFalse(payload.contains("gold_route"))
        assertFalse(payload.contains("utilities"))
        assertFalse(payload.contains("ai.example.private"))
        assertTrue(payload.contains("\"source_package\":\"sha256:"))
        assertTrue(payload.contains("\"foundation_model_id\":\"LiquidAI/LFM2.5-230M\""))
        assertTrue(payload.contains("\"foundation_precision\":\"FP32\""))
        assertTrue(payload.contains("\"foundation_quantization\":\"none\""))
        assertTrue(payload.contains("\"foundation_route\":\"ARCHIVE\""))
        assertTrue(payload.contains("\"foundation_available\":true"))
        assertTrue(
            payload.contains(
                "\"foundation_probs\":{\"INTERRUPT\":0.1,\"LATER\":0.2,\"ARCHIVE\":0.7}",
            ),
        )
        assertTrue(payload.contains("\"foundation_raw_probabilities_exposed\":true"))
    }

    @Test
    fun `separate demo publisher package remains visible as capture provenance`() {
        val payload = TelemetryPayloads.decision(
            decision().copy(
                context = decision().context.copy(
                    packageName = "ai.onlinesdft.publisher.calendar",
                    eventId = "real-calendar-001",
                ),
            ),
            sequence = 1,
            labMode = false,
        )

        assertTrue(
            payload.contains(
                "\"source_package\":\"ai.onlinesdft.publisher.calendar\"",
            ),
        )
    }

    @Test
    fun `protected decision separates model recommendation from applied pass through`() {
        val original = decision()
        val payload = TelemetryPayloads.decision(
            original.copy(
                context = original.context.copy(isClearable = false, isCall = true),
                recommendedRoute = Route.LATER,
                chosenRoute = Route.INTERRUPT,
                executionConstraint = ExecutionConstraint.CALL,
            ),
            sequence = 4,
            labMode = false,
        )

        assertTrue(payload.contains("\"model_recommended_route\":\"LATER\""))
        assertTrue(payload.contains("\"chosen_route\":\"INTERRUPT\""))
        assertTrue(payload.contains("\"execution_constraint\":\"call\""))
        assertTrue(payload.contains("\"route_execution\":\"platform_protected_pass_through\""))
        assertTrue(payload.contains("\"behavior_probs\":{\"INTERRUPT\":1.0"))
        assertTrue(payload.contains("\"call\":true"))
        assertFalse(payload.contains("secret title"))
    }

    @Test
    fun `protected interrupt records the executed pass through behavior`() {
        val original = decision()
        val payload = TelemetryPayloads.decision(
            original.copy(
                context = original.context.copy(isOngoing = true),
                recommendedRoute = Route.INTERRUPT,
                chosenRoute = Route.INTERRUPT,
                executionConstraint = ExecutionConstraint.ONGOING,
            ),
            sequence = 5,
            labMode = false,
        )

        assertTrue(payload.contains("\"execution_constraint\":\"ongoing\""))
        assertTrue(payload.contains("\"route_execution\":\"platform_protected_pass_through\""))
        assertTrue(payload.contains("\"behavior_probs\":{\"INTERRUPT\":1.0"))
    }

    @Test
    fun `listener race records truthful non applied execution state`() {
        val payload = TelemetryPayloads.decision(
            decision(),
            sequence = 6,
            labMode = false,
            routeExecution = "not_applied_listener_disconnected",
        )

        assertTrue(
            payload.contains(
                "\"route_execution\":\"not_applied_listener_disconnected\"",
            ),
        )
        assertFalse(payload.contains("\"route_execution\":\"cancellation_requested\""))
    }

    @Test
    fun `explicit correction uses dedicated strict source and outcome`() {
        val payload = TelemetryPayloads.feedback(
            FactualFeedback(
                eventId = "event-1",
                executedRoute = Route.LATER,
                outcome = Outcome.EXPLICIT_USER_CORRECTION,
                observedSelection = Route.INTERRUPT,
                delayMinutes = 0,
                source = FeedbackSource.EXPLICIT_USER_CORRECTION,
                explicitPreference = Route.INTERRUPT,
                observedAtMillis = 1_700_000_000_000L,
            ),
        )
        assertTrue(payload.contains("\"feedback_source\":\"explicit_user_correction\""))
        assertTrue(payload.contains("\"outcome\":\"EXPLICIT_USER_CORRECTION\""))
        assertTrue(payload.contains("\"action_taken\":\"LATER\""))
        assertTrue(payload.contains("\"observed_user_selection\":\"INTERRUPT\""))
    }

    @Test
    fun `training payload exposes strict proof for both independent optimizer steps`() {
        val proofs = listOf(
            OptimizerStepProof(
                stepIndex = 1,
                updateIndex = 7,
                sampledEventIds = listOf("newest", "prior-1"),
                batchSize = 2,
                lossBefore = 0.9,
                lossAfter = 0.7,
                gradientNorm = 0.4,
                unclippedUpdateNorm = 0.6,
                appliedUpdateNorm = 0.25,
            ),
            OptimizerStepProof(
                stepIndex = 2,
                updateIndex = 8,
                sampledEventIds = listOf("newest", "prior-\"2"),
                batchSize = 2,
                lossBefore = 0.7,
                lossAfter = 0.55,
                gradientNorm = 0.2,
                unclippedUpdateNorm = 0.3,
                appliedUpdateNorm = 0.25,
            ),
        )
        val payload = TelemetryPayloads.trainingUpdate(
            TrainingMetrics(
                updateIndex = 8,
                eventId = "newest",
                teacherProbabilities = floatArrayOf(0.6f, 0.3f, 0.1f),
                sealedDecisionProbabilities = floatArrayOf(0.2f, 0.5f, 0.3f),
                behaviorSupport = floatArrayOf(1f, 0f, 0f),
                fusedTarget = floatArrayOf(0.9f, 0.08f, 0.02f),
                lossBefore = 0.9,
                lossAfter = 0.55,
                targetKlBefore = 0.5,
                targetKlAfter = 0.2,
                gradientNorm = 0.2,
                adapterNorm = 0.31,
                deltaNorm = 0.31,
                callbackUpdateNorm = 0.17,
                adapterDeltaNormBefore = 0.2,
                adapterDeltaNormAfter = 0.31,
                checksumBefore = "before",
                checksumAfter = "after",
                batchSize = 2,
                replaySize = 9,
                optimizerSteps = 2,
                optimizerStepLosses = doubleArrayOf(0.7, 0.55),
                optimizerStepProofs = proofs,
                trainingExamples = 4,
                teacherForwardLatencyMillis = 12.0,
                durationMillis = 5.0,
            ),
            endedAtMillis = 1_700_000_000_000L,
        )

        assertTrue(payload.contains("\"optimizer_steps\":2"))
        assertTrue(payload.contains("\"adapter_delta_norm\":0.31"))
        assertTrue(payload.contains("\"callback_update_norm\":0.17"))
        assertTrue(payload.contains("\"optimizer_step_proofs\":[{"))
        assertTrue(payload.contains("\"step_index\":1,\"update_index\":7"))
        assertTrue(payload.contains("\"sampled_event_ids\":[\"newest\",\"prior-1\"]"))
        assertTrue(payload.contains("\"unclipped_update_norm\":0.6"))
        assertTrue(payload.contains("\"applied_update_norm\":0.25"))
        assertTrue(payload.contains("\"prior-\\\"2\""))
        assertFalse(payload.contains("\"optimizer_step_proofs\":\""))
    }

    @Test
    fun `transport creates server run before flushing run scoped child records`() {
        val paths = Collections.synchronizedList(mutableListOf<String>())
        val bodies = Collections.synchronizedList(mutableListOf<String>())
        val server = ServerSocket(0, 4, InetAddress.getByName("127.0.0.1"))
        val serverThread = Thread {
            repeat(4) {
                server.accept().use { socket ->
                    val reader = socket.getInputStream().bufferedReader()
                    val requestLine = reader.readLine()
                    val path = requestLine.split(' ')[1]
                    var contentLength = 0
                    while (true) {
                        val header = reader.readLine()
                        if (header.isNullOrEmpty()) break
                        if (header.startsWith("Content-Length:", ignoreCase = true)) {
                            contentLength = header.substringAfter(':').trim().toInt()
                        }
                    }
                    val body = CharArray(contentLength)
                    var offset = 0
                    while (offset < contentLength) {
                        val count = reader.read(body, offset, contentLength - offset)
                        if (count < 0) break
                        offset += count
                    }
                    paths += path
                    bodies += String(body, 0, offset)
                    val response = if (path == "/api/v1/runs") {
                        val ordinal = paths.count { it == "/api/v1/runs" }
                        "{\"run_id\":\"server-run-${if (ordinal == 1) "123" else "456"}\",\"created\":true}"
                    } else {
                        "{\"created\":true}"
                    }
                    val bytes = response.toByteArray()
                    socket.getOutputStream().bufferedWriter().use { writer ->
                        writer.write("HTTP/1.1 200 OK\r\n")
                        writer.write("Content-Type: application/json\r\n")
                        writer.write("Content-Length: ${bytes.size}\r\n")
                        writer.write("Connection: close\r\n\r\n")
                        writer.write(response)
                    }
                }
            }
        }.apply {
            isDaemon = true
            start()
        }
        try {
            val client = TelemetryClient(
                filesDir = temporaryFolder.root,
                baseUrl = "http://127.0.0.1:${server.localPort}",
                exportEnabled = true,
            )
            client.decision(decision(), labMode = true)
            val deadline = System.currentTimeMillis() + 4_000
            while (paths.size < 2 && System.currentTimeMillis() < deadline) Thread.sleep(20)

            val firstLocalRun = client.runId
            val secondLocalRun = client.startNewRun()
            client.decision(decision().copy(context = decision().context.copy(eventId = "event-2")), labMode = true)
            val secondDeadline = System.currentTimeMillis() + 4_000
            while (paths.size < 4 && System.currentTimeMillis() < secondDeadline) Thread.sleep(20)

            assertEquals("/api/v1/runs", paths[0])
            assertEquals("/api/v1/runs/server-run-123/decisions", paths[1])
            assertEquals("/api/v1/runs", paths[2])
            assertEquals("/api/v1/runs/server-run-456/decisions", paths[3])
            assertNotEquals(firstLocalRun, secondLocalRun)
            assertTrue(bodies[0].contains("\"idempotency_key\""))
            assertTrue(bodies[1].contains("\"sequence\":1"))
            assertTrue(bodies[3].contains("\"sequence\":1"))
            val retainedAudit = File(temporaryFolder.root, "telemetry")
                .listFiles()
                .orEmpty()
                .filter { it.name.startsWith("audit-") }
                .joinToString("\n") { it.readText() }
            assertTrue(retainedAudit.contains("event-1"))
            assertTrue(retainedAudit.contains("event-2"))
        } finally {
            server.close()
            serverThread.join(1_000)
        }
    }

    @Test
    fun `local audit recovers after telemetry directory is externally deleted`() {
        val client = TelemetryClient(filesDir = temporaryFolder.root)
        val telemetryDir = File(temporaryFolder.root, "telemetry")
        assertTrue(telemetryDir.isDirectory)
        assertTrue(telemetryDir.deleteRecursively())
        assertFalse(telemetryDir.exists())

        val preResetDecision = runCatching {
            client.decision(
                decision().copy(
                    context = decision().context.copy(eventId = "deleted-dir-before-reset"),
                ),
                labMode = true,
            )
        }
        val newSession = runCatching { client.startNewRun() }
        val postResetDecision = runCatching {
            client.decision(
                decision().copy(
                    context = decision().context.copy(eventId = "deleted-dir-after-reset"),
                ),
                labMode = true,
            )
        }

        assertNull(preResetDecision.exceptionOrNull())
        assertNull(newSession.exceptionOrNull())
        assertNull(postResetDecision.exceptionOrNull())
        val newRunId = newSession.getOrThrow()
        val audit = File(telemetryDir, "audit-v1.jsonl")
        val metadata = File(telemetryDir, "run-v3.properties")
        val deadline = System.currentTimeMillis() + 2_000L
        while (
            (!audit.isFile || !metadata.isFile ||
                !audit.readText().contains("deleted-dir-after-reset")) &&
            System.currentTimeMillis() < deadline
        ) {
            Thread.sleep(10)
        }

        assertTrue(telemetryDir.isDirectory)
        assertTrue(audit.isFile)
        assertTrue(audit.readText().contains("run\t"))
        assertTrue(audit.readText().contains("decisions\t"))
        assertTrue(audit.readText().contains("deleted-dir-after-reset"))
        val session = Properties().apply {
            metadata.inputStream().use(::load)
        }
        assertEquals(newRunId, session.getProperty("idempotency_key"))
        assertEquals("1", session.getProperty("sequence"))
        val retainedAudit = telemetryDir.listFiles().orEmpty()
            .filter { it.name.startsWith("audit-") }
            .joinToString("\n") { it.readText() }
        assertTrue(retainedAudit.contains("deleted-dir-before-reset"))
    }

    @Test
    fun `default mode writes local audit without opening a socket`() {
        val server = ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))
        val states = Collections.synchronizedList(mutableListOf<BackendState>())
        try {
            val client = TelemetryClient(
                filesDir = temporaryFolder.root,
                baseUrl = "http://127.0.0.1:${server.localPort}",
                onState = states::add,
            )
            client.decision(decision(), labMode = true)
            client.flush()

            val audit = File(temporaryFolder.root, "telemetry/audit-v1.jsonl")
            val deadline = System.currentTimeMillis() + 2_000L
            while ((!audit.isFile || !audit.readText().contains("event-1")) &&
                System.currentTimeMillis() < deadline
            ) {
                Thread.sleep(10)
            }

            assertTrue(audit.readText().contains("run\t"))
            assertTrue(audit.readText().contains("decisions\t"))
            assertTrue(audit.readText().contains("event-1"))
            assertTrue(states.contains(BackendState.LOCAL_ONLY))
            val exportQueue = File(temporaryFolder.root, "telemetry/outbox-v3.jsonl")
            assertTrue(!exportQueue.exists() || exportQueue.length() == 0L)

            server.soTimeout = 250
            val openedSocket = try {
                server.accept().use { }
                true
            } catch (_: SocketTimeoutException) {
                false
            }
            assertFalse("local-only telemetry must not attempt export", openedSocket)
        } finally {
            server.close()
        }
    }

    @Test
    fun `new local session does not wait for stalled optional export`() {
        val server = ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))
        val requestAccepted = CountDownLatch(1)
        val releaseRequest = CountDownLatch(1)
        val serverThread = Thread {
            runCatching {
                server.accept().use {
                    requestAccepted.countDown()
                    releaseRequest.await(3, TimeUnit.SECONDS)
                }
            }
        }.apply {
            isDaemon = true
            start()
        }
        try {
            val client = TelemetryClient(
                filesDir = temporaryFolder.root,
                baseUrl = "http://127.0.0.1:${server.localPort}",
                exportEnabled = true,
            )
            assertTrue(requestAccepted.await(2, TimeUnit.SECONDS))

            val oldRun = client.runId
            val startedNanos = System.nanoTime()
            val newRun = client.startNewRun()
            val elapsedMillis = (System.nanoTime() - startedNanos) / 1_000_000L

            assertNotEquals(oldRun, newRun)
            assertTrue(
                "session rotation blocked for ${elapsedMillis}ms",
                elapsedMillis < 1_000L,
            )
        } finally {
            releaseRequest.countDown()
            server.close()
            serverThread.join(1_000)
        }
    }

    private fun requiredKeys(payload: String, vararg keys: String) {
        keys.forEach { key -> assertTrue("missing $key", payload.contains("\"$key\":")) }
    }

    private fun decision() = DecisionSnapshot(
        context = NotificationContext(
            eventId = "event-1",
            packageName = "ai.example.private",
            title = "secret title",
            body = "private body",
            category = "calendar",
            importance = 0.8f,
            regime = Regime.WEEKDAY,
            hourOfDay = 10f,
            postedAtMillis = 1L,
            caseId = "calendar-design-review",
        ),
        studentFeatures = FloatArray(FeatureEncoder.FEATURE_DIM),
        probabilities = floatArrayOf(0.2f, 0.3f, 0.5f),
        baseProbabilities = floatArrayOf(0.1f, 0.2f, 0.7f),
        chosenRoute = Route.ARCHIVE,
        baseRoute = Route.ARCHIVE,
        checkpointIndex = 2,
        adapterChecksum = "0123456789abcdef",
        decidedAtMillis = 1_700_000_000_000L,
        inferenceLatencyMillis = 0.25,
        foundationModelId = "LiquidAI/LFM2.5-230M",
        foundationPrecision = "FP32",
        foundationRoute = Route.ARCHIVE,
        foundationAvailable = true,
        foundationInferenceLatencyMillis = 150.0,
        foundationPromptTokens = 32,
        foundationCompletionTokens = 1,
        foundationTokensPerSecond = 6.67f,
    )
}
