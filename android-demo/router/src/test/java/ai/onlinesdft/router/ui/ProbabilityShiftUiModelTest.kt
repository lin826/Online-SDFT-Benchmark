package ai.onlinesdft.router.ui

import ai.onlinesdft.router.model.DecisionSnapshot
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.Regime
import ai.onlinesdft.router.model.Route
import ai.onlinesdft.router.state.RoutedEventUi
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ProbabilityShiftUiModelTest {
    @Test
    fun `shift formatter uses signed percentage points with at most two decimals`() {
        assertEquals("24.3% (-6.9%)", formatProbabilityWithShift(0.243f, 0.312f))
        assertEquals("1.54% (+1.54%)", formatProbabilityWithShift(0.0154f, 0f))
        assertEquals("74.16% (+5.36%)", formatProbabilityWithShift(0.7416f, 0.688f))
    }

    @Test
    fun `shift formatter canonicalizes exact and rounded zero without a sign`() {
        assertEquals("0% (0%)", formatProbabilityWithShift(-0.0f, 0.0f))
        assertEquals("0% (0%)", formatProbabilityWithShift(0.0f, -0.0f))
        assertEquals("50% (0%)", formatProbabilityWithShift(0.50004f, 0.5f))
    }

    @Test
    fun `each route row shows the post training distribution against its sealed value`() {
        val event = event(
            original = floatArrayOf(0.312f, 0.688f, 0f),
            current = floatArrayOf(0.243f, 0.7416f, 0.0154f),
            updateIndex = 8L,
        )

        val rows = routeProbabilityShiftRows(event)

        assertEquals(Route.entries, rows.map { it.route })
        assertEquals(
            listOf("24.3% (-6.9%)", "74.16% (+5.36%)", "1.54% (+1.54%)"),
            rows.map { it.displayText },
        )
        assertEquals(-6.9, rows[Route.INTERRUPT.ordinal].deltaPercentagePoints!!, 1e-4)
        assertEquals(5.36, rows[Route.LATER.ordinal].deltaPercentagePoints!!, 1e-4)
        assertEquals(1.54, rows[Route.ARCHIVE.ordinal].deltaPercentagePoints!!, 1e-4)
        assertEquals(8L, event.distributionUpdateIndex)
    }

    @Test
    fun `before an update rows show the sealed distribution without a delta`() {
        val original = floatArrayOf(0.312f, 0.688f, 0f)

        val rows = routeProbabilityShiftRows(event(original = original))

        assertEquals(listOf("31.2%", "68.8%", "0%"), rows.map { it.displayText })
        rows.forEachIndexed { index, row ->
            assertEquals(original[index], row.originalProbability, 0f)
            assertEquals(original[index], row.currentProbability, 0f)
            assertNull(row.deltaPercentagePoints)
        }
    }

    @Test
    fun `distribution projection never rewrites the action frozen before feedback`() {
        val original = floatArrayOf(0.51f, 0.49f, 0f)
        val event = event(
            original = original,
            current = floatArrayOf(0.24f, 0.7446f, 0.0154f),
            updateIndex = 12L,
        )

        val rows = routeProbabilityShiftRows(event)

        assertEquals(Route.INTERRUPT, event.decision.recommendedRoute)
        assertEquals(Route.INTERRUPT, event.decision.chosenRoute)
        assertArrayEquals(original, event.decision.probabilities, 0f)
        assertTrue(
            rows[Route.LATER.ordinal].currentProbability >
                rows[Route.INTERRUPT.ordinal].currentProbability,
        )
    }

    private fun event(
        original: FloatArray,
        current: FloatArray? = null,
        updateIndex: Long? = null,
    ): RoutedEventUi = RoutedEventUi(
        decision = DecisionSnapshot(
            context = NotificationContext(
                eventId = "distribution-row",
                packageName = "demo.source",
                title = "Project update",
                body = "A realistic visible notification.",
                category = "teammate",
                importance = 0.6f,
                regime = Regime.WEEKDAY,
                hourOfDay = 14f,
                postedAtMillis = 1L,
            ),
            studentFeatures = floatArrayOf(),
            probabilities = original.copyOf(),
            baseProbabilities = original.copyOf(),
            chosenRoute = Route.INTERRUPT,
            recommendedRoute = Route.INTERRUPT,
            baseRoute = Route.INTERRUPT,
            checkpointIndex = 0L,
            adapterChecksum = "sealed",
            decidedAtMillis = 1L,
            inferenceLatencyMillis = 1.0,
        ),
        postTrainingProbabilities = current?.copyOf(),
        distributionUpdateIndex = updateIndex,
    )
}
