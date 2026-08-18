package ai.onlinesdft.router.model

/** Lifecycle reported by the immutable on-device Liquid inference runtime. */
enum class FoundationModelPhase {
    NOT_STARTED,
    LOADING,
    READY,
    ERROR,
}

data class FoundationModelStatus(
    val modelId: String,
    val precision: String,
    val phase: FoundationModelPhase = FoundationModelPhase.NOT_STARTED,
    val downloadedBytes: Long = 0L,
    val totalBytes: Long = MODEL_BUNDLE_BYTES,
    val lastError: String? = null,
    val lastInferenceMillis: Double? = null,
    val lastPromptTokens: Long? = null,
    val lastCompletionTokens: Long? = null,
    val lastTokensPerSecond: Float? = null,
) {
    val isReady: Boolean get() = phase == FoundationModelPhase.READY

    companion object {
        // The manifest replaces this estimate with the verified deployed byte
        // total as soon as the inference-only bundle loads.
        const val MODEL_BUNDLE_BYTES = 1_000_000_000L
    }
}
