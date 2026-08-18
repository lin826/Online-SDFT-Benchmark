package ai.onlinesdft.router.model

/** Scores a frozen decision before feedback or model updates are allowed. */
class PrequentialScorer(
    private val store: PrequentialMetricsStore = PrequentialMetricsStore(),
) {
    private var metrics = store.load()

    @Synchronized
    fun score(
        decision: DecisionSnapshot,
        truth: EvaluationTruth?,
        labMode: Boolean,
    ): EvaluationMetrics? {
        if (truth == null) return null
        require(labMode) {
            "gold routes and utilities are accepted only in labeled synthetic lab mode"
        }
        val bestUtility = truth.utilities.maxOrNull()!!.toDouble()
        val stepRegret = bestUtility - truth.utilities[decision.chosenRoute.ordinal]
        val baseStepRegret = bestUtility - truth.utilities[decision.baseRoute.ordinal]
        val decisions = metrics.decisions + 1
        val correct = metrics.correct + if (decision.chosenRoute == truth.goldRoute) 1 else 0
        val baseCorrect = metrics.baseCorrect + if (decision.baseRoute == truth.goldRoute) 1 else 0
        metrics = EvaluationMetrics(
            decisions = decisions,
            correct = correct,
            onlineAccuracy = correct.toDouble() / decisions,
            cumulativeRegret = metrics.cumulativeRegret + stepRegret,
            baseCorrect = baseCorrect,
            baseAccuracy = baseCorrect.toDouble() / decisions,
            baseCumulativeRegret = metrics.baseCumulativeRegret + baseStepRegret,
            lastStepRegret = stepRegret,
        )
        store.save(metrics)
        return metrics
    }

    @Synchronized
    fun current(): EvaluationMetrics = metrics

    @Synchronized
    fun reset(): Boolean {
        metrics = PrequentialMetricsStore.emptyMetrics()
        return store.clear()
    }
}
