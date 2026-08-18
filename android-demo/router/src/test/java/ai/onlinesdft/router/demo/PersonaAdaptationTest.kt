package ai.onlinesdft.router.demo

import ai.onlinesdft.router.model.DeterministicFrozenFoundationRuntime
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.OnlineSdftLearner
import ai.onlinesdft.router.model.Regime
import ai.onlinesdft.router.model.LoraReplayStore
import ai.onlinesdft.router.model.Route
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PersonaAdaptationTest {
    @Test
    fun `repeated visible preference shifts LoRA policy while foundation stays frozen`() {
        val learner = OnlineSdftLearner(
            foundationRuntime = DeterministicFrozenFoundationRuntime(),
            replayStore = LoraReplayStore(),
            clockMillis = { 1_700_000_000_000L },
        )
        val initial = learner.decide(promo("persona-initial"))
        val initialArchive = initial.probabilities[Route.ARCHIVE.ordinal]
        val frozenBase = initial.baseProbabilities.copyOf()

        repeat(12) { index ->
            val decision = learner.decide(promo("persona-train-$index"))
            learner.learn(
                decision,
                PersonaFeedbackSimulator.scriptedCorrection(
                    decision,
                    Route.ARCHIVE,
                    observedAtMillis = 1_700_000_000_001L + index,
                ),
            )
        }

        val adapted = learner.decide(promo("persona-adapted"))
        assertArrayEquals(frozenBase, adapted.baseProbabilities, 0f)
        assertTrue(
            "archive ${initialArchive} -> ${adapted.probabilities[Route.ARCHIVE.ordinal]}",
            adapted.probabilities[Route.ARCHIVE.ordinal] > initialArchive + 0.20f,
        )
        assertEquals(Route.ARCHIVE, adapted.recommendedRoute)
        assertTrue(learner.status().loraNorm > 0.0)
        assertEquals(172_032, learner.status().trainableParameters)
    }

    @Test
    fun `two semantic clusters learn different user routes from the same frozen prior`() {
        val learner = OnlineSdftLearner(
            foundationRuntime = DeterministicFrozenFoundationRuntime(),
            replayStore = LoraReplayStore(),
            clockMillis = { 1_700_000_000_000L },
        )
        val packageInitial = learner.decide(packageNotice("package-initial"))
        val securityInitial = learner.decide(securityNotice("security-initial"))

        repeat(14) { index ->
            val packageDecision = learner.decide(packageNotice("package-train-$index"))
            learner.learn(
                packageDecision,
                PersonaFeedbackSimulator.scriptedCorrection(
                    packageDecision,
                    Route.LATER,
                    observedAtMillis = 1_700_000_000_100L + 2L * index,
                ),
            )
            val securityDecision = learner.decide(securityNotice("security-train-$index"))
            learner.learn(
                securityDecision,
                PersonaFeedbackSimulator.scriptedCorrection(
                    securityDecision,
                    Route.INTERRUPT,
                    observedAtMillis = 1_700_000_000_101L + 2L * index,
                ),
            )
        }

        val packageAdapted = learner.decide(packageNotice("package-adapted"))
        val securityAdapted = learner.decide(securityNotice("security-adapted"))
        assertArrayEquals(packageInitial.baseProbabilities, packageAdapted.baseProbabilities, 0f)
        assertArrayEquals(securityInitial.baseProbabilities, securityAdapted.baseProbabilities, 0f)
        assertTrue(
            packageAdapted.probabilities[Route.LATER.ordinal] >
                packageInitial.probabilities[Route.LATER.ordinal],
        )
        assertTrue(
            securityAdapted.probabilities[Route.INTERRUPT.ordinal] >
                securityInitial.probabilities[Route.INTERRUPT.ordinal] + 0.15f,
        )
        assertEquals(Route.LATER, packageAdapted.recommendedRoute)
        assertEquals(Route.INTERRUPT, securityAdapted.recommendedRoute)
    }

    private fun promo(eventId: String) = NotificationContext(
        eventId = eventId,
        packageName = "ai.example.shopping",
        title = "Weekend offers selected for you",
        body = "Browse this week's general promotions whenever you have time.",
        category = "promo",
        importance = 0.1f,
        regime = Regime.OFF_HOURS,
        hourOfDay = 20.5f,
        postedAtMillis = 1L,
    )

    private fun packageNotice(eventId: String) = NotificationContext(
        eventId = eventId,
        packageName = "ai.example.delivery",
        title = "Package delivered at your front door",
        body = "Your Atlas Market order arrived safely.",
        category = "commerce",
        importance = 0.55f,
        regime = Regime.WEEKDAY,
        hourOfDay = 14f,
        postedAtMillis = 1L,
    )

    private fun securityNotice(eventId: String) = NotificationContext(
        eventId = eventId,
        packageName = "ai.example.bank",
        title = "Card used for a new online purchase",
        body = "Review this security activity now if you do not recognize it.",
        category = "monitoring",
        importance = 0.95f,
        regime = Regime.ON_CALL,
        hourOfDay = 2f,
        postedAtMillis = 1L,
    )
}
