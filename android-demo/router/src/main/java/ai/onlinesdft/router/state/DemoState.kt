package ai.onlinesdft.router.state

import ai.onlinesdft.router.model.DecisionSnapshot
import ai.onlinesdft.router.model.EvaluationMetrics
import ai.onlinesdft.router.model.FactualFeedback
import ai.onlinesdft.router.model.FoundationModelStatus
import ai.onlinesdft.router.model.ModelStatus
import ai.onlinesdft.router.model.TrainingMetrics
import ai.onlinesdft.router.notification.DigestInboxItem
import ai.onlinesdft.router.notification.DigestInboxSummary
import ai.onlinesdft.router.telemetry.BackendState

data class RoutedEventUi(
    val decision: DecisionSnapshot,
    /** Distribution captured after this row's own latest accepted update. */
    val postTrainingProbabilities: FloatArray? = null,
    val distributionUpdateIndex: Long? = null,
    val feedback: FactualFeedback? = null,
    val training: TrainingMetrics? = null,
    val lessonStatus: String = "Waiting for a factual callback",
    val syntheticLab: Boolean = false,
)

enum class DemoPage {
    ROUTER,
    DIGEST,
    SCORES,
}

/**
 * One scored decision, recorded in order so the progress screen can show how
 * the assistant tracked against the non-learning baseline over time. Counters
 * are cumulative, exactly as the evaluator reports them.
 */
data class ScoredDecisionPoint(
    val decisions: Int,
    val adaptiveCorrect: Int,
    val baseCorrect: Int,
)

data class DemoUiState(
    val listenerConnected: Boolean = false,
    val backendState: BackendState = BackendState.LOCAL_ONLY,
    val modelStatus: ModelStatus = ModelStatus(
        updateIndex = 0,
        checksum = "initializing",
        // Legacy constructor slots in ModelStatus; both values are FP64
        // LoRA tensor norms reported by the ORT training graph.
        adapterNorm = 0.0,
        rawAdapterNorm = 0.0,
        replaySize = 0,
        trainableParameters = 1_572,
        trainableTensors = 1,
        foundationStatus = FoundationModelStatus(
            modelId = "LiquidAI/LFM2.5-230M",
            precision = "FP32",
        ),
    ),
    val events: List<RoutedEventUi> = emptyList(),
    val digestItems: List<DigestInboxItem> = emptyList(),
    val digestSummary: DigestInboxSummary = DigestInboxSummary(
        total = 0,
        unread = 0,
        sourceCounts = emptyList(),
        previewTitles = emptyList(),
    ),
    val lastTraining: TrainingMetrics? = null,
    val evaluation: EvaluationMetrics = EvaluationMetrics(
        decisions = 0,
        correct = 0,
        onlineAccuracy = 0.0,
        cumulativeRegret = 0.0,
        baseCorrect = 0,
        baseAccuracy = 0.0,
        baseCumulativeRegret = 0.0,
        lastStepRegret = 0.0,
    ),
    /** Chronological scored decisions for this app session's progress trend. */
    val scoredHistory: List<ScoredDecisionPoint> = emptyList(),
    val labRunning: Boolean = false,
    val pendingCorrectionEventIds: Set<String> = emptySet(),
    val completedCorrectionEventIds: Set<String> = emptySet(),
    val statusMessage: String = "Ready for user-authorized notification access",
    val runId: String = "",
    val selectedPage: DemoPage = DemoPage.ROUTER,
    val pageRequestId: Long = 0L,
    val highlightedEventId: String? = null,
)
