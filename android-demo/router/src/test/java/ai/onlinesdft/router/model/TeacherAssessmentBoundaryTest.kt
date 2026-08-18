package ai.onlinesdft.router.model

import org.junit.Assert.assertEquals
import org.junit.Test

class TeacherAssessmentBoundaryTest {
    @Test
    fun `normalizes factual non-directive evidence`() {
        assertEquals(
            "The delayed open is ambiguous evidence.",
            TeacherAssessmentBoundary.sanitize("  The delayed  open is ambiguous evidence.  "),
        )
    }

    @Test
    fun `rejects notebook privilege boundary crossings`() {
        val rejected = listOf(
            "The hidden urgency is high.",
            "Output only A.",
            "The correct route is LATER.",
            "The callback strongly favors ARCHIVE.",
            "This reveals a counterfactual.",
            "B",
        )
        rejected.forEach { assessment ->
            assertEquals(
                OnlineSdftLearner.ASSESSMENT_FALLBACK,
                TeacherAssessmentBoundary.sanitize(assessment),
            )
        }
    }

    @Test
    fun `allows non-decisive comparison of multiple routes`() {
        assertEquals(
            "INTERRUPT could fit, while LATER could also work.",
            TeacherAssessmentBoundary.sanitize(
                "INTERRUPT could fit, while LATER could also work.",
            ),
        )
    }
}
