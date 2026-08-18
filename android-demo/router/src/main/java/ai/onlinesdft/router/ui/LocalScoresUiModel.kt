package ai.onlinesdft.router.ui

import ai.onlinesdft.router.state.DemoUiState
import kotlin.math.abs

/** Display-only projection whose sole input is the immutable Compose state. */
internal data class LocalScoresSnapshot(
    val decisions: Int,
    val adaptiveCorrect: Int,
    val adaptiveAccuracy: Double,
    val frozenCorrect: Int,
    val frozenAccuracy: Double,
    val accuracyDelta: Double,
    val adaptiveCumulativeRegret: Double,
    val frozenCumulativeRegret: Double,
    val regretGap: Double,
    val regretReduction: Double,
    val latestLossBefore: Double?,
    val latestLossAfter: Double?,
    val optimizerSteps: List<LocalOptimizerStepSnapshot>,
    val updates: Long,
    val replaySize: Int,
    val loraNorm: Double,
) {
    init {
        require(optimizerSteps.size == DISPLAYED_OPTIMIZER_STEPS)
    }

    companion object {
        const val DISPLAYED_OPTIMIZER_STEPS = 1

        fun from(state: DemoUiState): LocalScoresSnapshot {
            val evaluation = state.evaluation
            val training = state.lastTraining
            val regretGap =
                evaluation.baseCumulativeRegret - evaluation.cumulativeRegret
            val regretReduction = if (abs(evaluation.baseCumulativeRegret) < 1e-12) {
                0.0
            } else {
                regretGap / evaluation.baseCumulativeRegret
            }
            return LocalScoresSnapshot(
                decisions = evaluation.decisions,
                adaptiveCorrect = evaluation.correct,
                adaptiveAccuracy = evaluation.onlineAccuracy,
                frozenCorrect = evaluation.baseCorrect,
                frozenAccuracy = evaluation.baseAccuracy,
                accuracyDelta = evaluation.onlineAccuracy - evaluation.baseAccuracy,
                adaptiveCumulativeRegret = evaluation.cumulativeRegret,
                frozenCumulativeRegret = evaluation.baseCumulativeRegret,
                regretGap = regretGap,
                regretReduction = regretReduction,
                latestLossBefore = training?.lossBefore,
                latestLossAfter = training?.lossAfter,
                optimizerSteps = List(DISPLAYED_OPTIMIZER_STEPS) { index ->
                    val stepNumber = index + 1
                    val proof = training?.optimizerStepProofs
                        ?.firstOrNull { it.stepIndex == stepNumber }
                    LocalOptimizerStepSnapshot(
                        stepNumber = stepNumber,
                        sampleCount = proof?.batchSize,
                        lossBefore = proof?.lossBefore,
                        lossAfter = proof?.lossAfter,
                        gradientNorm = proof?.gradientNorm,
                        appliedUpdateNorm = proof?.appliedUpdateNorm,
                    )
                },
                updates = state.modelStatus.updateIndex,
                replaySize = state.modelStatus.replaySize,
                loraNorm = state.modelStatus.loraNorm,
            )
        }
    }
}

internal data class LocalOptimizerStepSnapshot(
    val stepNumber: Int,
    val sampleCount: Int?,
    val lossBefore: Double?,
    val lossAfter: Double?,
    val gradientNorm: Double?,
    val appliedUpdateNorm: Double?,
) {
    val completed: Boolean
        get() = sampleCount != null &&
            lossBefore != null &&
            lossAfter != null &&
            gradientNorm != null &&
            appliedUpdateNorm != null
}
