package ai.onlinesdft.router.model

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SdftTargetBuilderTest {
    @Test
    fun `reliable causal fusion uses exact main 05 05 90 profile`() {
        val target = SdftTargetBuilder.build(
            teacher = doubleArrayOf(0.6, 0.3, 0.1),
            sealedDecision = doubleArrayOf(0.2, 0.5, 0.3),
            feedback = feedback(Route.INTERRUPT, Outcome.OPENED_IMMEDIATELY, Route.INTERRUPT),
        )!!

        assertEquals(SdftEvidenceReliability.RELIABLE_SINGLETON, target.reliability)
        assertEquals("INTERRUPT", target.replayLabel)
        assertArrayEquals(floatArrayOf(1f, 0f, 0f), target.behaviorSupport, 0f)
        assertArrayEquals(floatArrayOf(0.94f, 0.04f, 0.02f), target.probabilities, 2e-7f)
        assertArrayEquals(doubleArrayOf(0.94, 0.04, 0.02), target.probabilitiesFp64, 2e-15)
        assertEquals(1f, target.probabilities.sum(), 2e-7f)
    }

    @Test
    fun `ambiguous digest open projects the sealed decision onto causal support`() {
        val target = SdftTargetBuilder.build(
            teacher = doubleArrayOf(0.6, 0.3, 0.1),
            sealedDecision = doubleArrayOf(0.2, 0.5, 0.3),
            feedback = feedback(Route.LATER, Outcome.OPENED_DIGEST, Route.LATER),
        )!!

        assertEquals(SdftEvidenceReliability.AMBIGUOUS_DIGEST_OPEN, target.reliability)
        assertEquals("AMBIGUOUS", target.replayLabel)
        assertArrayEquals(floatArrayOf(0.5f, 0.5f, 0f), target.behaviorSupport, 0f)
        assertEquals(2f / 7f, target.probabilities[0], 2e-7f)
        assertEquals(5f / 7f, target.probabilities[1], 2e-7f)
        assertTrue(target.probabilities[2] in 0f..2e-8f)
        assertEquals((2.0 / 7.0) / (1.0 + 1e-8), target.probabilitiesFp64[0], 2e-15)
        assertEquals((5.0 / 7.0) / (1.0 + 1e-8), target.probabilitiesFp64[1], 2e-15)
        assertEquals(1e-8 / (1.0 + 1e-8), target.probabilitiesFp64[2], 2e-15)
        assertEquals(1f, target.probabilities.sum(), 2e-7f)
    }

    @Test
    fun `unknown archive callback creates no target replay row or preference`() {
        val unknown = feedback(
            route = Route.ARCHIVE,
            outcome = Outcome.NO_OBSERVABLE_SELECTION,
            selection = null,
        )

        assertNull(SdftTargetBuilder.support(unknown))
        assertNull(SdftTargetBuilder.maximumEntropySupport(unknown))
        assertNull(
            SdftTargetBuilder.build(
                teacher = floatArrayOf(0.3f, 0.3f, 0.4f),
                sealedDecision = floatArrayOf(0.2f, 0.5f, 0.3f),
                feedback = unknown,
            ),
        )
    }

    @Test
    fun `opening a reversible archive teaches Later`() {
        val engagement = feedback(
            route = Route.ARCHIVE,
            outcome = Outcome.OPENED_DIGEST,
            selection = Route.LATER,
        ).copy(source = FeedbackSource.DIGEST_CALLBACK)
        val target = SdftTargetBuilder.build(
            teacher = doubleArrayOf(0.6, 0.2, 0.2),
            sealedDecision = doubleArrayOf(0.1, 0.2, 0.7),
            feedback = engagement,
        )!!

        assertArrayEquals(
            booleanArrayOf(false, true, false),
            SdftTargetBuilder.support(engagement),
        )
        assertEquals(SdftEvidenceReliability.RELIABLE_SINGLETON, target.reliability)
        assertEquals("LATER", target.replayLabel)
    }

    @Test
    fun `explicit correction is singleton only with the dedicated factual source`() {
        val correction = feedback(
            route = Route.LATER,
            outcome = Outcome.EXPLICIT_USER_CORRECTION,
            selection = Route.INTERRUPT,
        ).copy(
            source = FeedbackSource.EXPLICIT_USER_CORRECTION,
            explicitPreference = Route.INTERRUPT,
        )
        val target = SdftTargetBuilder.build(
            teacher = floatArrayOf(0.1f, 0.2f, 0.7f),
            sealedDecision = floatArrayOf(0.2f, 0.5f, 0.3f),
            feedback = correction,
        )!!

        assertArrayEquals(booleanArrayOf(true, false, false), SdftTargetBuilder.support(correction))
        assertEquals("INTERRUPT", target.replayLabel)

        val spoofed = correction.copy(source = FeedbackSource.SYNTHETIC_LAB)
        assertNull(SdftTargetBuilder.support(spoofed))
        assertNull(
            SdftTargetBuilder.build(
                floatArrayOf(0.1f, 0.2f, 0.7f),
                floatArrayOf(0.2f, 0.5f, 0.3f),
                spoofed,
            ),
        )
    }

    @Test
    fun `every supported Android trajectory matches main causal route support`() {
        val cases = listOf(
            Triple(Route.INTERRUPT, Outcome.OPENED_IMMEDIATELY, booleanArrayOf(true, false, false)),
            Triple(Route.INTERRUPT, Outcome.OPENED_AFTER_DELAY, booleanArrayOf(false, true, false)),
            Triple(Route.INTERRUPT, Outcome.DELETED_NOTIFICATION, booleanArrayOf(false, false, true)),
            Triple(Route.LATER, Outcome.OPENED_DIGEST, booleanArrayOf(true, true, false)),
            Triple(Route.LATER, Outcome.DELETED_FROM_DIGEST, booleanArrayOf(false, false, true)),
            Triple(Route.LATER, Outcome.TIMED_OUT_UNTOUCHED, booleanArrayOf(false, true, false)),
            Triple(Route.ARCHIVE, Outcome.OPENED_DIGEST, booleanArrayOf(false, true, false)),
        )

        cases.forEach { (route, outcome, expected) ->
            assertArrayEquals(
                "$route/$outcome",
                expected,
                SdftTargetBuilder.support(feedback(route, outcome, null)),
            )
        }
    }

    @Test
    fun `trusted untouched timeout reinforces Later without pretending it was opened`() {
        val timeout = feedback(
            route = Route.LATER,
            outcome = Outcome.TIMED_OUT_UNTOUCHED,
            selection = Route.LATER,
        ).copy(
            source = FeedbackSource.ANDROID_CALLBACK,
            delayMinutes = 120,
        )

        assertArrayEquals(
            booleanArrayOf(false, true, false),
            SdftTargetBuilder.support(timeout),
        )
        assertEquals("LATER", SdftTargetBuilder.build(
            teacher = doubleArrayOf(0.2, 0.6, 0.2),
            sealedDecision = doubleArrayOf(0.3, 0.5, 0.2),
            feedback = timeout,
        )?.replayLabel)
    }

    private fun feedback(
        route: Route,
        outcome: Outcome,
        selection: Route?,
    ) = FactualFeedback(
        eventId = "event-1",
        executedRoute = route,
        outcome = outcome,
        observedSelection = selection,
        delayMinutes = 1,
        source = FeedbackSource.SYNTHETIC_LAB,
        observedAtMillis = 2L,
    )
}
