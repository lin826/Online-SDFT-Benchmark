package ai.onlinesdft.router.ui

import ai.onlinesdft.router.model.FoundationModelPhase
import ai.onlinesdft.router.state.DemoUiState
import ai.onlinesdft.router.state.ScoredDecisionPoint

/** Plain-language readiness of the assistant, for a status pill. */
internal enum class AssistantReadiness(val label: String) {
    STARTING("Starting up"),
    WARMING("Getting ready"),
    READY("Ready"),
    PAUSED("Paused"),
}

/**
 * How well the assistant knows this person. Derived from the number of lessons
 * it has actually applied, so the stage never overstates what happened.
 */
internal enum class PersonalizationStage(val label: String, val blurb: String) {
    NEW("Just met you", "It starts with a general sense of what usually matters."),
    LEARNING("Learning you", "Each correction you make shapes what it does next."),
    FAMILIAR("Getting to know you", "It is starting to predict your calls before you make them."),
    TUNED("Tuned to you", "It has a solid picture of what you want interrupted for."),
}

internal data class TrendPoint(
    val decisions: Int,
    val adaptiveAccuracy: Double,
    val baselineAccuracy: Double,
)

/**
 * Everything the progress screen shows, derived only from the immutable UI
 * state. Values are null when there is not yet enough evidence to state them.
 */
internal data class LearningProgress(
    val readiness: AssistantReadiness,
    val lessons: Long,
    val stage: PersonalizationStage,
    /** 0..1 fill for the personalization meter. */
    val stageProgress: Float,
    val decisions: Int,
    /** Share of scored decisions that matched what the person wanted. */
    val accuracy: Double?,
    /** Same model, same notifications, no learning. */
    val baselineAccuracy: Double?,
    /** Accuracy advantage over the non-learning baseline, in points. */
    val advantagePoints: Double?,
    /** Accuracy over the earliest decisions, once there are enough of them. */
    val earlyAccuracy: Double?,
    /** Accuracy over the most recent decisions. */
    val recentAccuracy: Double?,
    /** Improvement from the early window to the recent window, in points. */
    val improvementPoints: Double?,
    /** Share of the baseline's cost of wrong calls that learning removed. */
    val mistakeCostReduction: Double?,
    val examplesRemembered: Int,
    val lastLessonErrorBefore: Double?,
    val lastLessonErrorAfter: Double?,
    val trend: List<TrendPoint>,
) {
    val hasScores: Boolean get() = decisions > 0

    /** Improvement is only meaningful once both windows are full. */
    val hasImprovement: Boolean get() = improvementPoints != null

    val lastLessonErrorDrop: Double?
        get() {
            val before = lastLessonErrorBefore ?: return null
            val after = lastLessonErrorAfter ?: return null
            if (before <= 0.0) return null
            return (before - after) / before
        }

    companion object {
        /** Decisions per comparison window for the early/recent split. */
        const val WINDOW = 5

        /** Lessons after which the assistant is described as tuned. */
        const val TUNED_AT = 20

        fun from(state: DemoUiState): LearningProgress {
            val evaluation = state.evaluation
            val model = state.modelStatus
            val lessons = model.updateIndex
            val history = state.scoredHistory
            val scored = evaluation.decisions > 0
            val costReduction = if (evaluation.baseCumulativeRegret <= 1e-12) {
                null
            } else {
                ((evaluation.baseCumulativeRegret - evaluation.cumulativeRegret) /
                    evaluation.baseCumulativeRegret).coerceIn(-1.0, 1.0)
            }
            val early = windowAccuracy(history, first = true)
            val recent = windowAccuracy(history, first = false)
            return LearningProgress(
                readiness = when (model.foundationStatus.phase) {
                    FoundationModelPhase.NOT_STARTED -> AssistantReadiness.STARTING
                    FoundationModelPhase.LOADING -> AssistantReadiness.WARMING
                    FoundationModelPhase.READY -> AssistantReadiness.READY
                    FoundationModelPhase.ERROR -> AssistantReadiness.PAUSED
                },
                lessons = lessons,
                stage = when {
                    lessons <= 0L -> PersonalizationStage.NEW
                    lessons < 5L -> PersonalizationStage.LEARNING
                    lessons < TUNED_AT -> PersonalizationStage.FAMILIAR
                    else -> PersonalizationStage.TUNED
                },
                stageProgress = (lessons.toFloat() / TUNED_AT).coerceIn(0f, 1f),
                decisions = evaluation.decisions,
                accuracy = evaluation.onlineAccuracy.takeIf { scored },
                baselineAccuracy = evaluation.baseAccuracy.takeIf { scored },
                advantagePoints = if (scored) {
                    (evaluation.onlineAccuracy - evaluation.baseAccuracy) * 100.0
                } else {
                    null
                },
                earlyAccuracy = early,
                recentAccuracy = recent,
                improvementPoints = if (early != null && recent != null) {
                    (recent - early) * 100.0
                } else {
                    null
                },
                mistakeCostReduction = costReduction,
                examplesRemembered = model.replaySize,
                lastLessonErrorBefore = state.lastTraining?.lossBefore,
                lastLessonErrorAfter = state.lastTraining?.lossAfter,
                trend = history.map {
                    TrendPoint(
                        decisions = it.decisions,
                        adaptiveAccuracy = ratio(it.adaptiveCorrect, it.decisions),
                        baselineAccuracy = ratio(it.baseCorrect, it.decisions),
                    )
                },
            )
        }

        /**
         * Accuracy across the first or last [WINDOW] scored decisions. Needs two
         * full, non-overlapping windows so the two numbers describe different
         * stretches of the stream.
         */
        private fun windowAccuracy(
            history: List<ScoredDecisionPoint>,
            first: Boolean,
        ): Double? {
            if (history.size < WINDOW * 2) return null
            return if (first) {
                val end = history[WINDOW - 1]
                ratio(end.adaptiveCorrect, end.decisions)
            } else {
                val start = history[history.size - WINDOW - 1]
                val end = history.last()
                ratio(
                    end.adaptiveCorrect - start.adaptiveCorrect,
                    end.decisions - start.decisions,
                )
            }
        }

        private fun ratio(correct: Int, total: Int): Double =
            if (total <= 0) 0.0 else correct.toDouble() / total
    }
}
