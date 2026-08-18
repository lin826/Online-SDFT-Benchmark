package ai.onlinesdft.router.ui

import ai.onlinesdft.router.model.EvaluationMetrics
import ai.onlinesdft.router.model.FoundationModelPhase
import ai.onlinesdft.router.model.FoundationModelStatus
import ai.onlinesdft.router.model.ModelStatus
import ai.onlinesdft.router.state.DemoUiState
import ai.onlinesdft.router.state.ScoredDecisionPoint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LearningProgressUiModelTest {
    @Test
    fun `a fresh install reports no scores and no personalization`() {
        val progress = LearningProgress.from(DemoUiState())

        assertFalse(progress.hasScores)
        assertEquals(PersonalizationStage.NEW, progress.stage)
        assertEquals(0f, progress.stageProgress, 1e-6f)
        assertNull(progress.accuracy)
        assertNull(progress.advantagePoints)
        assertNull(progress.improvementPoints)
        assertTrue(progress.trend.isEmpty())
    }

    @Test
    fun `readiness follows the on-device model phase`() {
        val phases = mapOf(
            FoundationModelPhase.NOT_STARTED to AssistantReadiness.STARTING,
            FoundationModelPhase.LOADING to AssistantReadiness.WARMING,
            FoundationModelPhase.READY to AssistantReadiness.READY,
            FoundationModelPhase.ERROR to AssistantReadiness.PAUSED,
        )

        phases.forEach { (phase, expected) ->
            val progress = LearningProgress.from(state(phase = phase))
            assertEquals(expected, progress.readiness)
        }
    }

    @Test
    fun `personalization stage advances with applied lessons`() {
        assertEquals(PersonalizationStage.NEW, LearningProgress.from(state(lessons = 0)).stage)
        assertEquals(PersonalizationStage.LEARNING, LearningProgress.from(state(lessons = 1)).stage)
        assertEquals(PersonalizationStage.FAMILIAR, LearningProgress.from(state(lessons = 5)).stage)
        assertEquals(
            PersonalizationStage.TUNED,
            LearningProgress.from(state(lessons = LearningProgress.TUNED_AT.toLong())).stage,
        )
        assertEquals(
            1f,
            LearningProgress.from(state(lessons = 999)).stageProgress,
            1e-6f,
        )
    }

    @Test
    fun `advantage over the non-learning baseline is reported in points`() {
        val progress = LearningProgress.from(
            state(
                evaluation = metrics(decisions = 10, correct = 8, baseCorrect = 5),
            ),
        )

        assertEquals(0.8, progress.accuracy!!, 1e-9)
        assertEquals(0.5, progress.baselineAccuracy!!, 1e-9)
        assertEquals(30.0, progress.advantagePoints!!, 1e-9)
    }

    @Test
    fun `improvement compares the first and last windows once both are full`() {
        val window = LearningProgress.WINDOW
        // First window: 1 of 5 correct. Last window: 5 of 5 correct.
        val correctByStep = List(window) { 0 } + List(window) { 1 }
        var adaptive = 0
        val history = correctByStep.mapIndexed { index, hit ->
            adaptive += hit
            ScoredDecisionPoint(
                decisions = index + 1,
                adaptiveCorrect = adaptive,
                baseCorrect = 0,
            )
        }

        val partial = LearningProgress.from(state(history = history.dropLast(1)))
        assertNull(partial.improvementPoints)
        assertFalse(partial.hasImprovement)

        val full = LearningProgress.from(state(history = history))
        assertEquals(0.0, full.earlyAccuracy!!, 1e-9)
        assertEquals(1.0, full.recentAccuracy!!, 1e-9)
        assertEquals(100.0, full.improvementPoints!!, 1e-9)
        assertTrue(full.hasImprovement)
    }

    @Test
    fun `mistake cost reduction is null until the baseline has accrued cost`() {
        val noCost = LearningProgress.from(
            state(evaluation = metrics(decisions = 4, correct = 4, baseCorrect = 4)),
        )
        assertNull(noCost.mistakeCostReduction)

        val withCost = LearningProgress.from(
            state(
                evaluation = metrics(
                    decisions = 4,
                    correct = 3,
                    baseCorrect = 1,
                    regret = 0.25,
                    baseRegret = 1.0,
                ),
            ),
        )
        assertEquals(0.75, withCost.mistakeCostReduction!!, 1e-9)
    }

    @Test
    fun `the trend carries a running accuracy for both arms`() {
        val progress = LearningProgress.from(
            state(
                history = listOf(
                    ScoredDecisionPoint(decisions = 1, adaptiveCorrect = 0, baseCorrect = 1),
                    ScoredDecisionPoint(decisions = 2, adaptiveCorrect = 1, baseCorrect = 1),
                    ScoredDecisionPoint(decisions = 4, adaptiveCorrect = 3, baseCorrect = 1),
                ),
            ),
        )

        assertEquals(3, progress.trend.size)
        assertEquals(0.0, progress.trend[0].adaptiveAccuracy, 1e-9)
        assertEquals(1.0, progress.trend[0].baselineAccuracy, 1e-9)
        assertEquals(0.75, progress.trend[2].adaptiveAccuracy, 1e-9)
        assertEquals(0.25, progress.trend[2].baselineAccuracy, 1e-9)
    }

    private fun state(
        phase: FoundationModelPhase = FoundationModelPhase.READY,
        lessons: Long = 0L,
        evaluation: EvaluationMetrics = metrics(),
        history: List<ScoredDecisionPoint> = emptyList(),
    ): DemoUiState = DemoUiState(
        modelStatus = ModelStatus(
            updateIndex = lessons,
            checksum = "test",
            adapterNorm = 0.0,
            rawAdapterNorm = 0.0,
            replaySize = 0,
            trainableParameters = 1,
            trainableTensors = 1,
            foundationStatus = FoundationModelStatus(
                modelId = "test",
                precision = "FP32",
                phase = phase,
            ),
        ),
        evaluation = evaluation,
        scoredHistory = history,
    )

    private fun metrics(
        decisions: Int = 0,
        correct: Int = 0,
        baseCorrect: Int = 0,
        regret: Double = 0.0,
        baseRegret: Double = 0.0,
    ): EvaluationMetrics = EvaluationMetrics(
        decisions = decisions,
        correct = correct,
        onlineAccuracy = if (decisions == 0) 0.0 else correct.toDouble() / decisions,
        cumulativeRegret = regret,
        baseCorrect = baseCorrect,
        baseAccuracy = if (decisions == 0) 0.0 else baseCorrect.toDouble() / decisions,
        baseCumulativeRegret = baseRegret,
        lastStepRegret = 0.0,
    )
}
