package ai.onlinesdft.router.demo

import ai.onlinesdft.router.model.EvaluationTruth
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.Regime
import ai.onlinesdft.router.model.Route

/**
 * A demo stream in which the person's preferences change partway through.
 *
 * The same five notifications repeat across four rounds. Rounds 1-2 use one set
 * of preferences and rounds 3-4 use another: promotional mail and the optional
 * webinar become worth interrupting for, the project chat stops being worth
 * anything, and two anchors keep their original preference so the shift is
 * partial rather than a wholesale relabel.
 *
 * Within each phase, round one is taught by explicit correction and round two
 * re-poses the same notification as a probe, so the recovery after the shift is
 * the learner generalising rather than being told the answer again.
 *
 * Kept separate from [PersonaScenarioCatalog] on purpose: that stream is pinned
 * by the reproducible research capture and must not move.
 */
object PreferenceShiftCatalog {
    fun cases(session: String, now: Long = System.currentTimeMillis()): List<LabScenario> {
        val templates = templates()
        return List(ROUNDS * templates.size) { index ->
            val template = templates[index % templates.size]
            val round = index / templates.size + 1
            val shifted = round > ROUNDS / 2
            val preference = if (shifted) template.shiftedPreference else template.preference
            val repeat = round % 2 == 0
            LabScenario(
                context = NotificationContext(
                    eventId = "shift-$session-${index.toString().padStart(3, '0')}",
                    packageName = template.packageName,
                    title = if (repeat) "${template.title} · again" else template.title,
                    body = if (repeat) {
                        "${template.body} The same kind of notification arrived again."
                    } else {
                        template.body
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
                simulatedPreference = preference,
                truth = EvaluationTruth(
                    goldRoute = preference,
                    utilities = utilitiesFor(preference),
                ),
            )
        }
    }

    /** True when this decision index is taught rather than probed. */
    fun isTaught(index: Int): Boolean = (index / PHASE_TEMPLATES) % 2 == 0

    /** True once the person's preferences have changed. */
    fun isShifted(index: Int): Boolean = index >= TOTAL_CASES / 2

    private fun utilitiesFor(gold: Route): FloatArray {
        val utilities = floatArrayOf(-0.6f, -0.6f, -0.6f)
        utilities[gold.ordinal] = 1.6f
        // The neighbouring choice stays mildly acceptable so regret reflects a
        // near miss differently from the clearly wrong option.
        utilities[(gold.ordinal + 1) % utilities.size] = 0.1f
        return utilities
    }

    private fun templates() = listOf(
        Template(
            caseId = "shift-promo-newsletter",
            packageName = "ai.onlinesdft.publisher.mail",
            category = "promo",
            title = "Store newsletter: extra 20% off",
            body = "The offer ends tonight. Browse recommended items now.",
            importance = 0.12f,
            regime = Regime.OFF_HOURS,
            preference = Route.ARCHIVE,
            shiftedPreference = Route.INTERRUPT,
        ),
        Template(
            caseId = "shift-project-chat",
            packageName = "ai.onlinesdft.publisher.chat",
            category = "teammate",
            title = "Three updates in the Aurora project chat",
            body = "The latest build notes and screenshots are ready for review.",
            importance = 0.55f,
            regime = Regime.WEEKDAY,
            preference = Route.LATER,
            shiftedPreference = Route.ARCHIVE,
        ),
        Template(
            caseId = "shift-monitoring-anchor",
            packageName = "ai.onlinesdft.publisher.chat",
            category = "monitoring",
            title = "Checkout error rate above 8%",
            body = "Production alert assigned to the mobile on-call rotation.",
            importance = 0.98f,
            regime = Regime.ON_CALL,
            preference = Route.INTERRUPT,
            shiftedPreference = Route.INTERRUPT,
        ),
        Template(
            caseId = "shift-delivery-anchor",
            packageName = "ai.onlinesdft.publisher.mail",
            category = "delivery",
            title = "Your package was delivered",
            body = "The parcel was left by the front door at 2:14 PM.",
            importance = 0.43f,
            regime = Regime.WEEKDAY,
            preference = Route.LATER,
            shiftedPreference = Route.LATER,
        ),
        Template(
            caseId = "shift-webinar",
            packageName = "ai.onlinesdft.publisher.calendar",
            category = "promo",
            title = "Suggested webinar this Friday",
            body = "Add the optional product trends webinar to your calendar.",
            importance = 0.16f,
            regime = Regime.WEEKDAY,
            preference = Route.ARCHIVE,
            shiftedPreference = Route.INTERRUPT,
        ),
    )

    private data class Template(
        val caseId: String,
        val packageName: String,
        val category: String,
        val title: String,
        val body: String,
        val importance: Float,
        val regime: Regime,
        val preference: Route,
        val shiftedPreference: Route,
    )

    const val ROUNDS = 4
    const val PHASE_TEMPLATES = 5
    const val TOTAL_CASES = ROUNDS * PHASE_TEMPLATES
    const val STREAM_VERSION = "preference-shift-v1"
}
