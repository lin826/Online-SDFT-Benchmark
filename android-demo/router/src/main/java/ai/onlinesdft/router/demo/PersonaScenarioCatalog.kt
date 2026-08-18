package ai.onlinesdft.router.demo

import ai.onlinesdft.router.model.EvaluationTruth
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.Regime
import ai.onlinesdft.router.model.Route

data class LabScenario(
    val context: NotificationContext,
    /** Hidden persona behavior used only to simulate a user's observable response. */
    val simulatedPreference: Route,
    /** Evaluator-only labels consumed exclusively by PrequentialScorer. */
    val truth: EvaluationTruth,
)

object PersonaScenarioCatalog {
    /**
     * Deterministic synthetic inputs aligned with backend suite
     * `alex-workday-v1`. Evaluator truth is a separate lab-only value and is
     * never placed in NotificationContext, FeatureEncoder, feedback, or a
     * training target. The persona simulator independently emits only the
     * delivery-surface behavior or explicit correction a learner could observe.
     */
    fun cases(session: String, now: Long = System.currentTimeMillis()): List<LabScenario> {
        val templates = templates()
        return List(20) { index ->
            val template = templates[index % templates.size]
            val round = index / templates.size + 1
            val eventId = "lab-$session-${index.toString().padStart(3, '0')}"
            LabScenario(
                context = NotificationContext(
                    eventId = eventId,
                    packageName = template.packageName,
                    title = if (round == 1) template.title else "${template.title} · follow-up",
                    body = if (round == 1) {
                        template.body
                    } else {
                        "${template.body} A follow-up arrived for the same preference context."
                    },
                    category = template.category,
                    importance = template.importance,
                    regime = template.regime,
                    hourOfDay = when (template.regime) {
                        Regime.WEEKDAY -> 14.5f
                        Regime.ON_CALL -> 2.25f
                        Regime.OFF_HOURS -> 19.75f
                    },
                    postedAtMillis = now + index * 1_000L,
                    caseId = template.caseId,
                ),
                simulatedPreference = template.simulatedPreference,
                truth = EvaluationTruth(
                    goldRoute = Route.entries[
                        template.utilities.indices.maxBy { template.utilities[it] }
                    ],
                    utilities = template.utilities.copyOf(),
                ),
            )
        }
    }

    private fun templates() = listOf(
        Template("monitoring-checkout-errors", "ai.onlinesdft.publisher.chat", "monitoring", "Checkout error rate above 8%", "Production alert assigned to the mobile on-call rotation.", 0.98f, Regime.ON_CALL, Route.INTERRUPT, floatArrayOf(2.0f, -0.1f, -1.4f)),
        Template("social-reaction", "ai.onlinesdft.publisher.chat", "social", "Sam and 12 others liked your photo", "See the newest reactions to your weekend post.", 0.18f, Regime.WEEKDAY, Route.ARCHIVE, floatArrayOf(-0.9f, 0.15f, 1.05f)),
        Template("calendar-design-review", "ai.onlinesdft.publisher.calendar", "calendar", "Design review starts in 10 minutes", "Maya is waiting in the Cedar video room. Tap to join.", 0.91f, Regime.WEEKDAY, Route.INTERRUPT, floatArrayOf(1.7f, 0.25f, -0.9f)),
        Template("bank-card-security", "ai.onlinesdft.publisher.mail", "security", "Was this card purchase yours?", "\$642.18 was charged at an unfamiliar electronics store.", 0.96f, Regime.OFF_HOURS, Route.INTERRUPT, floatArrayOf(1.9f, 0.1f, -1.2f)),
        Template("chat-project-update", "ai.onlinesdft.publisher.chat", "teammate", "Three updates in the Aurora project chat", "The latest build notes and screenshots are ready for review.", 0.55f, Regime.WEEKDAY, Route.LATER, floatArrayOf(0.05f, 1.05f, 0.15f)),
        Template("package-delivered", "ai.onlinesdft.publisher.mail", "delivery", "Your package was delivered", "The parcel was left by the front door at 2:14 PM.", 0.43f, Regime.WEEKDAY, Route.LATER, floatArrayOf(-0.35f, 0.9f, 0.25f)),
        Template("expense-receipt", "ai.onlinesdft.publisher.mail", "receipt", "Receipt for your rideshare trip", "Your Tuesday trip receipt is ready to view.", 0.31f, Regime.WEEKDAY, Route.LATER, floatArrayOf(-0.45f, 0.72f, 0.4f)),
        Template("shopping-promotion", "ai.onlinesdft.publisher.mail", "promo", "Store newsletter: extra 20% off", "The offer ends tonight. Browse recommended items now.", 0.12f, Regime.OFF_HOURS, Route.ARCHIVE, floatArrayOf(-1.1f, 0.05f, 1.3f)),
        Template("calendar-webinar-suggestion", "ai.onlinesdft.publisher.calendar", "promo", "Suggested webinar this Friday", "Add the optional product trends webinar to your calendar.", 0.16f, Regime.WEEKDAY, Route.ARCHIVE, floatArrayOf(-0.95f, 0.1f, 1.1f)),
        Template("calendar-weekly-agenda", "ai.onlinesdft.publisher.calendar", "calendar", "Your agenda for next week is ready", "Five planned meetings were added to the weekly overview.", 0.48f, Regime.WEEKDAY, Route.LATER, floatArrayOf(-0.15f, 0.95f, 0.2f)),
    )

    private data class Template(
        val caseId: String,
        val packageName: String,
        val category: String,
        val title: String,
        val body: String,
        val importance: Float,
        val regime: Regime,
        val simulatedPreference: Route,
        val utilities: FloatArray,
    )

    const val CURRICULUM_SIZE = 10
    const val STREAM_VERSION = "alex-correction-probe-v1"
}
