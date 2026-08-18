package ai.onlinesdft.router.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class PrequentialScorerTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `score uses frozen action and orders accuracy and regret correctly`() {
        val scorer = PrequentialScorer()
        val decision = decision(chosen = Route.LATER, base = Route.ARCHIVE)
        val truth = EvaluationTruth(Route.LATER, floatArrayOf(-0.5f, 1.4f, 0.2f))

        val metrics = scorer.score(decision, truth, labMode = true)!!
        assertEquals(1, metrics.decisions)
        assertEquals(1, metrics.correct)
        assertEquals(1.0, metrics.onlineAccuracy, 0.0)
        assertEquals(0.0, metrics.cumulativeRegret, 1e-9)
        assertEquals(0, metrics.baseCorrect)
        assertEquals(1.2, metrics.baseCumulativeRegret, 1e-6)
    }

    @Test
    fun `evaluator truth is rejected outside labeled lab mode`() {
        assertThrows(IllegalArgumentException::class.java) {
            PrequentialScorer().score(
                decision(Route.INTERRUPT, Route.INTERRUPT),
                EvaluationTruth(Route.INTERRUPT, floatArrayOf(1f, 0f, -1f)),
                labMode = false,
            )
        }
    }

    @Test
    fun `aggregate evaluation restores across scorer process recreation`() {
        val file = File(temporaryFolder.root, "metrics/prequential.bin")
        val truth = EvaluationTruth(Route.LATER, floatArrayOf(-0.5f, 1.4f, 0.2f))
        PrequentialScorer(PrequentialMetricsStore(file)).score(
            decision(Route.LATER, Route.ARCHIVE),
            truth,
            labMode = true,
        )

        val restored = PrequentialScorer(PrequentialMetricsStore(file)).current()
        assertEquals(1, restored.decisions)
        assertEquals(1, restored.correct)
        assertEquals(0, restored.baseCorrect)
        assertEquals(1.2, restored.baseCumulativeRegret, 1e-6)
    }

    @Test
    fun `reset atomically persists the zero aggregate`() {
        val file = File(temporaryFolder.root, "metrics/prequential.bin")
        val scorer = PrequentialScorer(PrequentialMetricsStore(file))
        scorer.score(
            decision(Route.LATER, Route.ARCHIVE),
            EvaluationTruth(Route.LATER, floatArrayOf(-0.5f, 1.4f, 0.2f)),
            labMode = true,
        )
        scorer.reset()

        assertEquals(0, PrequentialScorer(PrequentialMetricsStore(file)).current().decisions)
    }

    @Test
    fun `failed zero write deletes stale aggregate instead of resurrecting scores`() {
        val file = File(temporaryFolder.root, "metrics/prequential.bin")
        val store = PrequentialMetricsStore(file)
        val scorer = PrequentialScorer(store)
        scorer.score(
            decision(Route.LATER, Route.ARCHIVE),
            EvaluationTruth(Route.LATER, floatArrayOf(-0.5f, 1.4f, 0.2f)),
            labMode = true,
        )
        // Force FileOutputStream(<target>.pending) to fail while leaving the
        // committed target deletable by the fail-closed reset path.
        val blockedPending = File(file.parentFile, "${file.name}.pending")
        assertTrue(blockedPending.mkdirs())
        File(blockedPending, "blocker").writeText("block atomic zero write")

        assertTrue(scorer.reset())
        assertEquals(0, PrequentialScorer(PrequentialMetricsStore(file)).current().decisions)
    }

    @Test
    fun `corrupt aggregate falls back to zero without crashing`() {
        val file = File(temporaryFolder.root, "metrics/prequential.bin")
        assertTrue(PrequentialMetricsStore(file).save(PrequentialMetricsStore.emptyMetrics()))
        file.writeBytes(byteArrayOf(1, 2, 3, 4))

        assertEquals(0, PrequentialScorer(PrequentialMetricsStore(file)).current().decisions)
    }

    private fun decision(chosen: Route, base: Route) = DecisionSnapshot(
        context = NotificationContext(
            eventId = "score-1",
            packageName = "demo",
            title = "title",
            body = "body",
            category = "calendar",
            importance = 0.8f,
            regime = Regime.WEEKDAY,
            hourOfDay = 9f,
            postedAtMillis = 1L,
        ),
        studentFeatures = FloatArray(FeatureEncoder.FEATURE_DIM),
        probabilities = floatArrayOf(0.2f, 0.7f, 0.1f),
        baseProbabilities = floatArrayOf(0.1f, 0.2f, 0.7f),
        chosenRoute = chosen,
        baseRoute = base,
        checkpointIndex = 0,
        adapterChecksum = "abcdef1234567890",
        decidedAtMillis = 2L,
        inferenceLatencyMillis = 0.1,
    )
}
