package ai.onlinesdft.router.model

enum class SdftEvidenceReliability {
    RELIABLE_SINGLETON,
    AMBIGUOUS_DIGEST_OPEN,
}

data class SdftTarget(
    val probabilities: FloatArray,
    val probabilitiesFp64: DoubleArray,
    val behaviorSupport: FloatArray,
    val reliability: SdftEvidenceReliability,
    val replayLabel: String,
)

object SdftTargetBuilder {
    const val TEACHER_WEIGHT = 0.05f
    const val DECISION_WEIGHT = 0.05f
    const val BEHAVIOR_WEIGHT = 0.90f
    const val AMBIGUOUS_TEACHER_WEIGHT = 0f
    const val AMBIGUOUS_DECISION_WEIGHT = 1f
    const val AMBIGUOUS_BEHAVIOR_WEIGHT = 0f

    fun support(feedback: FactualFeedback): BooleanArray? {
        explicitCorrection(feedback)?.let { preferred ->
            return mask(preferred)
        }
        return when (feedback.executedRoute to feedback.outcome) {
            Route.INTERRUPT to Outcome.OPENED_IMMEDIATELY -> mask(Route.INTERRUPT)
            Route.INTERRUPT to Outcome.OPENED_AFTER_DELAY -> mask(Route.LATER)
            Route.INTERRUPT to Outcome.TIMED_OUT_UNTOUCHED -> mask(Route.LATER)
            Route.INTERRUPT to Outcome.DELETED_NOTIFICATION -> mask(Route.ARCHIVE)
            Route.LATER to Outcome.OPENED_DIGEST -> mask(Route.INTERRUPT, Route.LATER)
            Route.LATER to Outcome.DELETED_FROM_DIGEST -> mask(Route.ARCHIVE)
            Route.LATER to Outcome.TIMED_OUT_UNTOUCHED -> mask(Route.LATER)
            Route.ARCHIVE to Outcome.OPENED_DIGEST -> mask(Route.LATER)
            Route.ARCHIVE to Outcome.NO_OBSERVABLE_SELECTION -> null
            else -> null
        }
    }

    fun maximumEntropySupport(feedback: FactualFeedback): FloatArray? {
        val support = support(feedback) ?: return null
        val count = support.count { it }
        if (count == 0) return null
        return FloatArray(support.size) { index ->
            if (support[index]) 1f / count else 0f
        }
    }

    /** Build the exact reliability-conditioned c9de64c causal target. */
    fun build(
        teacher: FloatArray,
        sealedDecision: FloatArray,
        feedback: FactualFeedback,
    ): SdftTarget? = build(
        teacher = teacher.toDoubleArrayExact(),
        sealedDecision = sealedDecision.toDoubleArrayExact(),
        feedback = feedback,
    )

    fun build(
        teacher: DoubleArray,
        sealedDecision: DoubleArray,
        feedback: FactualFeedback,
    ): SdftTarget? {
        val support = support(feedback) ?: return null
        val behavior = maximumEntropy(support)
        val behaviorFp64 = behavior.toDoubleArrayExact()
        val supportedRoutes = support.count { it }
        return when (supportedRoutes) {
            1 -> {
                val fused = fuse(teacher, sealedDecision, behaviorFp64)
                SdftTarget(
                    probabilities = fused.toFloatArrayExact(),
                    probabilitiesFp64 = fused,
                    behaviorSupport = behavior,
                    reliability = SdftEvidenceReliability.RELIABLE_SINGLETON,
                    replayLabel = explicitCorrection(feedback)?.name
                        ?: feedback.observedSelection?.name
                        ?: Route.entries[support.indexOfFirst { it }].name,
                )
            }
            2 -> {
                require(
                    feedback.executedRoute == Route.LATER &&
                        feedback.outcome == Outcome.OPENED_DIGEST,
                ) { "Only a digest open has two-route causal support" }
                val projectedDecision = project(sealedDecision, support)
                // Canonical ambiguous weights are 0/1/0. Projecting the
                // teacher as Python does is immaterial numerically but keeps
                // this branch explicit and reviewable.
                project(teacher, support)
                val fused = normalize(projectedDecision)
                SdftTarget(
                    probabilities = fused.toFloatArrayExact(),
                    probabilitiesFp64 = fused,
                    behaviorSupport = behavior,
                    reliability = SdftEvidenceReliability.AMBIGUOUS_DIGEST_OPEN,
                    replayLabel = "AMBIGUOUS",
                )
            }
            else -> null
        }
    }

    /** Reliable singleton fusion retained as a direct parity/test primitive. */
    fun fuse(
        teacher: FloatArray,
        sealedDecision: FloatArray,
        behavior: FloatArray,
    ): FloatArray = fuse(
        teacher.toDoubleArrayExact(),
        sealedDecision.toDoubleArrayExact(),
        behavior.toDoubleArrayExact(),
    ).toFloatArrayExact()

    fun fuse(
        teacher: DoubleArray,
        sealedDecision: DoubleArray,
        behavior: DoubleArray,
    ): DoubleArray {
        val normalizedTeacher = normalize(teacher)
        val normalizedDecision = normalize(sealedDecision)
        val normalizedBehavior = normalizeWithoutFloor(behavior)
        val target = DoubleArray(Route.entries.size) { index ->
            RELIABLE_TEACHER_WEIGHT_FP64 * normalizedTeacher[index] +
                RELIABLE_DECISION_WEIGHT_FP64 * normalizedDecision[index] +
                RELIABLE_BEHAVIOR_WEIGHT_FP64 * normalizedBehavior[index]
        }
        return normalize(target)
    }

    fun normalize(values: FloatArray): FloatArray {
        return normalize(values.toDoubleArrayExact()).toFloatArrayExact()
    }

    private fun explicitCorrection(feedback: FactualFeedback): Route? {
        if (
            feedback.outcome != Outcome.EXPLICIT_USER_CORRECTION ||
            feedback.source != FeedbackSource.EXPLICIT_USER_CORRECTION
        ) {
            return null
        }
        return requireNotNull(feedback.explicitPreference) {
            "Explicit correction requires the user's selected route"
        }
    }

    private fun project(values: DoubleArray, support: BooleanArray): DoubleArray {
        require(values.size == support.size)
        val normalized = normalize(values)
        val projected = DoubleArray(values.size) { index ->
            if (support[index]) normalized[index] else 0.0
        }
        return normalizeWithoutFloor(projected)
    }

    private fun normalize(values: DoubleArray): DoubleArray {
        require(values.size == Route.entries.size)
        values.forEach { require(it.isFinite()) }
        val clipped = DoubleArray(values.size) { values[it].coerceAtLeast(1e-8) }
        val total = clipped.sum().coerceAtLeast(1e-8)
        return DoubleArray(clipped.size) { clipped[it] / total }
    }

    private fun normalizeWithoutFloor(values: DoubleArray): DoubleArray {
        require(values.size == Route.entries.size)
        values.forEach { require(it.isFinite() && it >= 0.0) }
        val total = values.sum()
        require(total.isFinite() && total > 0.0)
        return DoubleArray(values.size) { values[it] / total }
    }

    private fun FloatArray.toDoubleArrayExact(): DoubleArray =
        DoubleArray(size) { this[it].toDouble() }

    private fun DoubleArray.toFloatArrayExact(): FloatArray =
        FloatArray(size) { this[it].toFloat() }

    private fun maximumEntropy(support: BooleanArray): FloatArray {
        val count = support.count { it }
        require(count > 0)
        return FloatArray(support.size) { index -> if (support[index]) 1f / count else 0f }
    }

    private fun mask(vararg routes: Route): BooleanArray =
        BooleanArray(Route.entries.size) { index -> routes.any { it.ordinal == index } }

    private const val RELIABLE_TEACHER_WEIGHT_FP64 = 0.05
    private const val RELIABLE_DECISION_WEIGHT_FP64 = 0.05
    private const val RELIABLE_BEHAVIOR_WEIGHT_FP64 = 0.90
}
