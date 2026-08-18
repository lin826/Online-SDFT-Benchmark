package ai.onlinesdft.router.ui

import ai.onlinesdft.router.model.EvaluationMetrics
import ai.onlinesdft.router.model.OptimizerStepProof
import ai.onlinesdft.router.model.TrainingMetrics
import ai.onlinesdft.router.state.DemoUiState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LocalScoresUiModelTest {
    @Test
    fun `empty state still exposes one pending ORT proof row`() {
        val scores = LocalScoresSnapshot.from(DemoUiState())

        assertEquals(1, scores.optimizerSteps.size)
        assertEquals(listOf(1), scores.optimizerSteps.map { it.stepNumber })
        assertTrue(scores.optimizerSteps.none { it.completed })
        assertNull(scores.latestLossBefore)
        assertNull(scores.latestLossAfter)
    }

    @Test
    fun `projection reads evaluation training and model metrics only from ui state`() {
        val training = trainingMetrics()
        val initial = DemoUiState()
        val state = initial.copy(
            evaluation = EvaluationMetrics(
                decisions = 10,
                correct = 8,
                onlineAccuracy = 0.8,
                cumulativeRegret = 2.0,
                baseCorrect = 5,
                baseAccuracy = 0.5,
                baseCumulativeRegret = 5.0,
                lastStepRegret = 0.1,
            ),
            lastTraining = training,
            modelStatus = initial.modelStatus.copy(
                updateIndex = 12,
                checksum = "residual-checksum-12",
                adapterNorm = 1.25,
                rawAdapterNorm = 1.25,
                replaySize = 9,
            ),
        )

        val scores = LocalScoresSnapshot.from(state)

        assertEquals(0.3, scores.accuracyDelta, 1e-12)
        assertEquals(3.0, scores.regretGap, 1e-12)
        assertEquals(0.6, scores.regretReduction, 1e-12)
        assertEquals(0.9, scores.latestLossBefore!!, 0.0)
        assertEquals(0.7, scores.latestLossAfter!!, 0.0)
        assertEquals(12L, scores.updates)
        assertEquals(9, scores.replaySize)
        assertEquals(1.25, scores.loraNorm, 0.0)

        val first = scores.optimizerSteps[0]
        assertTrue(first.completed)
        assertEquals(2, first.sampleCount)
        assertEquals(0.9, first.lossBefore!!, 0.0)
        assertEquals(0.8, first.lossAfter!!, 0.0)
        assertEquals(0.1, first.gradientNorm!!, 0.0)
        assertEquals(0.15, first.appliedUpdateNorm!!, 0.0)

        assertFalse(scores.optimizerSteps.any { it.sampleCount == null })
    }

    private fun trainingMetrics(): TrainingMetrics {
        val stepOne = OptimizerStepProof(
            stepIndex = 1,
            updateIndex = 12,
            sampledEventIds = listOf("event-new", "event-old"),
            batchSize = 2,
            lossBefore = 0.9,
            lossAfter = 0.8,
            gradientNorm = 0.1,
            unclippedUpdateNorm = 0.15,
            appliedUpdateNorm = 0.15,
        )
        return TrainingMetrics(
            updateIndex = 12,
            eventId = "event-new",
            teacherProbabilities = floatArrayOf(0.2f, 0.3f, 0.5f),
            sealedDecisionProbabilities = floatArrayOf(0.3f, 0.3f, 0.4f),
            behaviorSupport = floatArrayOf(0.0f, 1.0f, 0.0f),
            fusedTarget = floatArrayOf(0.02f, 0.93f, 0.05f),
            lossBefore = 0.9,
            lossAfter = 0.7,
            targetKlBefore = 0.5,
            targetKlAfter = 0.2,
            gradientNorm = 0.2,
            adapterNorm = 1.25,
            deltaNorm = 1.25,
            callbackUpdateNorm = 0.2,
            adapterDeltaNormBefore = 1.0,
            adapterDeltaNormAfter = 1.25,
            checksumBefore = "residual-checksum-10",
            checksumAfter = "residual-checksum-12",
            batchSize = 3,
            replaySize = 9,
            optimizerSteps = 1,
            optimizerStepLosses = doubleArrayOf(0.8),
            optimizerStepProofs = listOf(stepOne),
            trainingExamples = 3,
            teacherForwardLatencyMillis = 12.0,
            durationMillis = 4.0,
        )
    }
}
