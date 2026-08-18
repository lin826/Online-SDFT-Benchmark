package ai.onlinesdft.router.model

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class OnlineSdftLearnerTest {
    @Test
    fun `decision comes from live LoRA student while base stays zero adapter`() {
        val runtime = DeterministicFrozenFoundationRuntime()
        val learner = learner(runtime)
        val before = learner.decide(context("before"))

        repeat(4) { index ->
            val decision = learner.decide(context("train-$index"))
            learner.learn(decision, correction(decision, Route.ARCHIVE, 100L + index))
        }
        val after = learner.decide(context("after"))

        assertArrayEquals(before.baseProbabilities, after.baseProbabilities, 0f)
        assertTrue(after.probabilities[Route.ARCHIVE.ordinal] > before.probabilities[Route.ARCHIVE.ordinal])
        assertEquals(1L, after.checkpointIndex)
        assertNotEquals(before.adapterChecksum, after.adapterChecksum)
        assertEquals(172_032, learner.status().trainableParameters)
        assertEquals(48, learner.status().trainableTensors)
    }

    @Test
    fun `fourth accepted callback performs a real runtime optimizer step`() {
        val runtime = DeterministicFrozenFoundationRuntime()
        val learner = learner(runtime)

        repeat(3) { index ->
            val decision = learner.decide(context("warmup-$index"))
            assertNull(learner.learn(decision, correction(decision, Route.LATER, 200L + index)))
        }
        assertEquals(3, learner.status().replaySize)
        assertEquals(0L, learner.status().updateIndex)

        val decision = learner.decide(context("update"))
        val metrics = requireNotNull(
            learner.learn(decision, correction(decision, Route.LATER, 300L)),
        )

        assertEquals(1L, metrics.updateIndex)
        assertEquals(1, metrics.optimizerSteps)
        assertEquals(1, metrics.batchSize)
        assertNotEquals(metrics.checksumBefore, metrics.checksumAfter)
        assertTrue(metrics.adapterNorm > 0.0)
        assertEquals(1L, runtime.adapterStatus().updateIndex)
    }

    @Test
    fun `hindsight teacher always uses fixed base evaluation`() {
        val runtime = DeterministicFrozenFoundationRuntime(
            studentProbabilities = floatArrayOf(0.1f, 0.2f, 0.7f),
            teacherProbabilities = floatArrayOf(0.8f, 0.1f, 0.1f),
        )
        val learner = learner(runtime)
        repeat(4) { index ->
            val decision = learner.decide(context("teacher-$index"))
            learner.learn(decision, correction(decision, Route.INTERRUPT, 400L + index))
        }

        assertEquals(4, runtime.teacherEvaluationCount)
        assertTrue(runtime.evaluatedPrompts.any { it.startsWith("base:") && "Observed callback:" in it })
    }

    @Test
    fun `duplicate feedback cannot update LoRA twice`() {
        val learner = learner(DeterministicFrozenFoundationRuntime())
        repeat(3) { index ->
            val decision = learner.decide(context("seed-$index"))
            learner.learn(decision, correction(decision, Route.ARCHIVE, 500L + index))
        }
        val decision = learner.decide(context("duplicate"))
        val feedback = correction(decision, Route.ARCHIVE, 600L)
        requireNotNull(learner.learn(decision, feedback))
        val status = learner.status()

        assertNull(learner.learn(decision, feedback))
        assertEquals(status.updateIndex, learner.status().updateIndex)
        assertEquals(status.checksum, learner.status().checksum)
    }

    @Test
    fun `reset drops LoRA optimizer state and replay`() {
        val runtime = DeterministicFrozenFoundationRuntime()
        val learner = learner(runtime)
        repeat(4) { index ->
            val decision = learner.decide(context("reset-$index"))
            learner.learn(decision, correction(decision, Route.ARCHIVE, 700L + index))
        }
        assertEquals(1L, learner.status().updateIndex)

        learner.reset()

        assertEquals(0L, learner.status().updateIndex)
        assertEquals(0, learner.status().replaySize)
        assertEquals(0.0, learner.status().loraNorm, 0.0)
    }

    private fun learner(runtime: FrozenFoundationRuntime) = OnlineSdftLearner(
        foundationRuntime = runtime,
        replayStore = LoraReplayStore(),
        clockMillis = { 42L },
    )

    private fun correction(
        decision: DecisionSnapshot,
        preferred: Route,
        observedAtMillis: Long,
    ) = FactualFeedback(
        eventId = decision.context.eventId,
        executedRoute = decision.chosenRoute,
        outcome = Outcome.EXPLICIT_USER_CORRECTION,
        observedSelection = preferred,
        delayMinutes = 0,
        source = FeedbackSource.EXPLICIT_USER_CORRECTION,
        explicitPreference = preferred,
        observedAtMillis = observedAtMillis,
    )

    private fun context(eventId: String) = NotificationContext(
        eventId = eventId,
        packageName = "demo.source",
        title = "Routine project digest",
        body = "A realistic visible message for on-device adaptation.",
        category = "teammate",
        importance = 0.6f,
        regime = Regime.WEEKDAY,
        hourOfDay = 14f,
        postedAtMillis = 1L,
        caseId = "same-semantic-row",
    )
}
