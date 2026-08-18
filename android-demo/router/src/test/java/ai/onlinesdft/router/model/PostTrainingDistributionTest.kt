package ai.onlinesdft.router.model

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PostTrainingDistributionTest {
    @Test
    fun `each identical notification snapshot moves monotonically toward keep silent`() {
        val learner = OnlineSdftLearner(
            foundationRuntime = DeterministicFrozenFoundationRuntime(),
            replayStore = LoraReplayStore(),
            clockMillis = { 100L },
        )
        val keepSilentAfterUpdate = mutableListOf<Float>()
        val updateIndices = mutableListOf<Long>()

        repeat(7) { index ->
            val decision = learner.decide(context("iteration-${index + 1}"))
            learner.learn(
                decision,
                correction(decision, Route.ARCHIVE, 200L + index),
            )?.let { training ->
                updateIndices += training.updateIndex
                keepSilentAfterUpdate += requireNotNull(
                    learner.reevaluateDistribution(decision),
                )[Route.ARCHIVE.ordinal]
            }
        }

        assertEquals(listOf(1L, 2L, 3L, 4L), updateIndices)
        assertEquals(4, keepSilentAfterUpdate.size)
        keepSilentAfterUpdate.zipWithNext().forEach { (before, after) ->
            assertTrue("Keep silent must increase after every iteration", after > before)
        }
    }

    @Test
    fun `post training reevaluation measures the current head without changing sealed action`() {
        val foundation = DeterministicFrozenFoundationRuntime()
        val learner = OnlineSdftLearner(
            foundationRuntime = foundation,
            replayStore = LoraReplayStore(),
            clockMillis = { 100L },
        )
        val sealed = learner.decide(context("sealed-action"))
        val sealedProbabilities = sealed.probabilities.copyOf()
        val sealedResidualDistribution = sealed.adaptiveDecisionProbabilities.copyOf()
        val sealedFoundationDistribution = sealed.foundationProbabilitiesFp64.copyOf()
        val sealedRecommended = sealed.recommendedRoute
        val sealedChosen = sealed.chosenRoute
        val sealedCheckpoint = sealed.checkpointIndex

        var completedUpdate: TrainingMetrics? = null
        repeat(4) { index ->
            val trainingDecision = learner.decide(context("training-${index + 1}"))
            learner.learn(
                trainingDecision,
                correction(trainingDecision, Route.ARCHIVE, 200L + index),
            )?.let { completedUpdate = it }
        }
        assertNotNull(completedUpdate)
        val foundationCallsBeforeMeasurement = foundation.studentEvaluationCount
        val statusBeforeMeasurement = learner.status()

        val current = requireNotNull(learner.reevaluateDistribution(sealed))

        assertEquals(foundationCallsBeforeMeasurement + 1, foundation.studentEvaluationCount)
        assertEquals(statusBeforeMeasurement.updateIndex, learner.status().updateIndex)
        assertEquals(statusBeforeMeasurement.checksum, learner.status().checksum)
        assertEquals(1.0, current.sumOf(Float::toDouble), 1e-6)
        assertTrue(current.indices.any { index ->
            kotlin.math.abs(current[index] - sealedProbabilities[index]) > 1e-6f
        })

        // The measurement is deliberately separate from the action snapshot.
        assertEquals(sealedRecommended, sealed.recommendedRoute)
        assertEquals(sealedChosen, sealed.chosenRoute)
        assertEquals(sealedCheckpoint, sealed.checkpointIndex)
        assertArrayEquals(sealedProbabilities, sealed.probabilities, 0f)
        assertArrayEquals(
            sealedResidualDistribution,
            sealed.adaptiveDecisionProbabilities,
            0.0,
        )
        assertArrayEquals(
            sealedFoundationDistribution,
            sealed.foundationProbabilitiesFp64,
            0.0,
        )
    }

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
        body = "A realistic visible message for LoRA adaptation.",
        category = "teammate",
        importance = 0.6f,
        regime = Regime.WEEKDAY,
        hourOfDay = 14f,
        postedAtMillis = 1L,
        caseId = "same-semantic-row",
    )
}
