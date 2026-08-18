package ai.onlinesdft.router.notification

import ai.onlinesdft.router.model.DecisionSnapshot
import ai.onlinesdft.router.model.ExecutionConstraint
import ai.onlinesdft.router.model.FeatureEncoder
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.Regime
import ai.onlinesdft.router.model.Route
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

enum class DigestPendingAction {
    OPEN,
    REMOVE,
    SHOW_NEXT,
    KEEP_SILENT,
}

enum class DigestInboxOrigin {
    LIVE_NOTIFICATION,
    ROUTER_ARCHIVE,
    SYNTHETIC_LAB,
}

data class DigestInboxItem(
    val eventId: String,
    val openToken: String,
    val sourcePackage: String,
    val title: String,
    val body: String,
    val routedAtMillis: Long,
    val origin: DigestInboxOrigin = DigestInboxOrigin.LIVE_NOTIFICATION,
    val readAtMillis: Long? = null,
    val pendingAction: DigestPendingAction? = null,
    val pendingActionAtMillis: Long? = null,
    val learningSnapshot: DigestLearningSnapshot? = null,
) {
    val isUnread: Boolean get() = readAtMillis == null
}

/** Sealed decision state required to learn from a delayed digest callback. */
data class DigestLearningSnapshot(
    val context: NotificationContext,
    val foundationModelId: String,
    val decidedAtMillis: Long,
    val checkpointIndex: Long,
    val runEpoch: Long,
    val studentPrompt: String,
    val foundationProbabilitiesFp64: DoubleArray,
    val adaptiveDecisionProbabilities: DoubleArray,
)

fun DecisionSnapshot.toDigestLearningSnapshot(): DigestLearningSnapshot {
    require(chosenRoute == Route.LATER || chosenRoute == Route.ARCHIVE) {
        "Only Later and Archive decisions belong in Saved"
    }
    return DigestLearningSnapshot(
        context = context.copy(),
        foundationModelId = foundationModelId,
        decidedAtMillis = decidedAtMillis,
        checkpointIndex = checkpointIndex,
        runEpoch = runEpoch,
        studentPrompt = studentPrompt,
        foundationProbabilitiesFp64 = foundationProbabilitiesFp64.copyOf(),
        adaptiveDecisionProbabilities = adaptiveDecisionProbabilities.copyOf(),
    )
}

/**
 * Reconstructs the exact sealed learner inputs after process recreation.
 * Display-only fields that are not consumed by learning are derived safely.
 */
fun DigestInboxItem.toDecision(runEpoch: Long): DecisionSnapshot? {
    val executedRoute = when (origin) {
        DigestInboxOrigin.LIVE_NOTIFICATION -> Route.LATER
        DigestInboxOrigin.ROUTER_ARCHIVE -> Route.ARCHIVE
        DigestInboxOrigin.SYNTHETIC_LAB -> return null
    }
    val snapshot = learningSnapshot ?: return null
    if (
        snapshot.runEpoch != runEpoch ||
        snapshot.context.eventId != eventId ||
        snapshot.context.packageName != sourcePackage ||
        snapshot.studentPrompt.isBlank() ||
        snapshot.foundationProbabilitiesFp64.size != Route.entries.size ||
        snapshot.adaptiveDecisionProbabilities.size != Route.entries.size
    ) return null
    val foundationRoute = snapshot.foundationProbabilitiesFp64.argmaxRoute()
    val recommendedRoute = snapshot.adaptiveDecisionProbabilities.argmaxRoute()
    return DecisionSnapshot(
        context = snapshot.context.copy(),
        studentPrompt = snapshot.studentPrompt,
        studentFeatures = FloatArray(FeatureEncoder.FEATURE_DIM),
        probabilities = snapshot.adaptiveDecisionProbabilities.toFloatArrayCopy(),
        baseProbabilities = snapshot.foundationProbabilitiesFp64.toFloatArrayCopy(),
        foundationProbabilitiesFp64 = snapshot.foundationProbabilitiesFp64.copyOf(),
        baselineAvailable = true,
        chosenRoute = executedRoute,
        recommendedRoute = recommendedRoute,
        executionConstraint = ExecutionConstraint.NONE,
        baseRoute = foundationRoute,
        baseRecommendedRoute = foundationRoute,
        checkpointIndex = snapshot.checkpointIndex,
        adapterChecksum = RESTORED_DIGEST_CHECKSUM,
        decidedAtMillis = snapshot.decidedAtMillis,
        inferenceLatencyMillis = 0.0,
        foundationModelId = snapshot.foundationModelId,
        foundationPrecision = RESTORED_DIGEST_PRECISION,
        foundationRoute = foundationRoute,
        foundationAvailable = true,
        runEpoch = runEpoch,
        adaptiveDecisionProbabilities = snapshot.adaptiveDecisionProbabilities.copyOf(),
    )
}

private fun DoubleArray.argmaxRoute(): Route = Route.entries[
    indices.maxByOrNull { index -> this[index] } ?: Route.INTERRUPT.ordinal
]

private fun DoubleArray.toFloatArrayCopy(): FloatArray = FloatArray(size) { index -> this[index].toFloat() }

private fun DigestInboxItem.deepCopy(): DigestInboxItem = copy(
    learningSnapshot = learningSnapshot?.deepCopy(),
)

private fun DigestLearningSnapshot.deepCopy(): DigestLearningSnapshot = copy(
    context = context.copy(),
    foundationProbabilitiesFp64 = foundationProbabilitiesFp64.copyOf(),
    adaptiveDecisionProbabilities = adaptiveDecisionProbabilities.copyOf(),
)

private const val RESTORED_DIGEST_CHECKSUM = "restored-digest"
private const val RESTORED_DIGEST_PRECISION = "restored-sealed-inputs"

data class DigestInboxSummary(
    val total: Int,
    val unread: Int,
    val sourceCounts: List<Pair<String, Int>>,
    val previewTitles: List<String>,
)

/**
 * Small, app-private inbox for notifications routed to Later.
 *
 * Mutations are serialized, persisted before becoming visible in memory, and
 * keyed by event id so a cancellation callback can be replayed safely.
 */
class DigestInboxStore(
    private val file: File? = null,
    private val maxItems: Int = DEFAULT_MAX_ITEMS,
) {
    private var inboxItems: List<DigestInboxItem>

    init {
        require(maxItems in 1..MAX_SUPPORTED_ITEMS) {
            "maxItems must be in 1..$MAX_SUPPORTED_ITEMS"
        }
        inboxItems = loadPersisted()
    }

    @Synchronized
    fun items(): List<DigestInboxItem> = inboxItems.map(DigestInboxItem::deepCopy)

    /** Returns true only when a new event id is retained successfully. */
    @Synchronized
    fun upsert(item: DigestInboxItem): Boolean {
        validate(item)
        if (inboxItems.any { it.eventId == item.eventId }) return false

        // A synthetic lab preview must never evict a live notification. A new
        // live delivery evicts the oldest preview first, preserving the
        // production cancellation obligation whenever possible.
        val retained = if (inboxItems.size < maxItems) {
            inboxItems
        } else {
            val previewIndex = inboxItems.indexOfLast {
                it.origin == DigestInboxOrigin.SYNTHETIC_LAB
            }
            val evictionIndex = when (item.origin) {
                DigestInboxOrigin.LIVE_NOTIFICATION,
                DigestInboxOrigin.ROUTER_ARCHIVE,
                ->
                    if (previewIndex >= 0) previewIndex else inboxItems.lastIndex
                DigestInboxOrigin.SYNTHETIC_LAB ->
                    if (previewIndex >= 0) previewIndex else return false
            }
            inboxItems.filterIndexed { index, _ -> index != evictionIndex }
        }
        val next = (retained + item.deepCopy())
            .sortedWith(NEWEST_FIRST)
        if (!persist(next)) return false
        inboxItems = next
        return true
    }

    /**
     * Durably claims the first terminal action for an authenticated item.
     * Repeating that action returns the original claim; its opposite loses.
     */
    @Synchronized
    fun claimAction(
        eventId: String,
        openToken: String,
        action: DigestPendingAction,
        atMillis: Long,
    ): DigestInboxItem? {
        if (!isValidEventId(eventId) || !isValidOpenToken(openToken) || atMillis < 0L) return null
        val index = inboxItems.indexOfFirst { it.eventId == eventId }
        if (index < 0) return null
        val current = inboxItems[index]
        if (!tokensEqual(current.openToken, openToken)) return null
        current.pendingAction?.let { pending ->
            return if (pending == action) current.deepCopy() else null
        }

        val claimed = current.copy(
            pendingAction = action,
            pendingActionAtMillis = atMillis,
        )
        val next = inboxItems.toMutableList().apply { this[index] = claimed }
        if (!persist(next)) return null
        inboxItems = next
        return claimed.deepCopy()
    }

    /** Returns true only for the first unread-to-read transition. */
    @Synchronized
    fun markRead(eventId: String, atMillis: Long): Boolean {
        requireValidEventId(eventId)
        require(atMillis >= 0L) { "atMillis must be non-negative" }
        val index = inboxItems.indexOfFirst { it.eventId == eventId }
        if (index < 0) return false
        val current = inboxItems[index]
        if (!current.isUnread && current.pendingAction == null) return false

        val next = inboxItems.toMutableList().apply {
            this[index] = current.copy(
                readAtMillis = current.readAtMillis ?: atMillis,
                pendingAction = null,
                pendingActionAtMillis = null,
            )
        }
        if (!persist(next)) return false
        inboxItems = next
        return current.isUnread
    }

    @Synchronized
    fun remove(eventId: String): Boolean {
        requireValidEventId(eventId)
        if (inboxItems.none { it.eventId == eventId }) return false
        val next = inboxItems.filterNot { it.eventId == eventId }
        if (!persist(next)) return false
        inboxItems = next
        return true
    }

    /**
     * Forgets sensitive content in memory first. If the atomic empty write
     * fails, both the committed and pending files are deleted best-effort.
     */
    @Synchronized
    fun clear(): Boolean {
        inboxItems = emptyList()
        if (persist(emptyList())) return true
        return deleteDurableState()
    }

    @Synchronized
    fun summary(): DigestInboxSummary {
        val sourceCounts = inboxItems
            .groupingBy { friendlySourceLabel(it.sourcePackage) }
            .eachCount()
            .map { (source, count) -> source to count }
            .sortedWith(
                compareByDescending<Pair<String, Int>> { it.second }
                    .thenBy { it.first },
            )
            .take(SUMMARY_LIMIT)
        return DigestInboxSummary(
            total = inboxItems.size,
            unread = inboxItems.count(DigestInboxItem::isUnread),
            sourceCounts = sourceCounts,
            previewTitles = inboxItems.take(SUMMARY_LIMIT).map(DigestInboxItem::title),
        )
    }

    private fun loadPersisted(): List<DigestInboxItem> {
        val target = file?.absoluteFile ?: return emptyList()
        File(target.parentFile, "${target.name}.pending").delete()
        if (!target.isFile) return emptyList()
        if (target.length() > MAX_ENCODED_BYTES) {
            target.delete()
            return emptyList()
        }
        val decoded = runCatching { decode(target.readBytes()) }.getOrElse {
            // Unsupported versions and corrupt records may contain raw content;
            // never surface them, and remove them best-effort.
            target.delete()
            return emptyList()
        }
        return decoded
            .sortedWith(NEWEST_FIRST)
            .distinctBy(DigestInboxItem::eventId)
            .take(maxItems)
    }

    private fun persist(items: List<DigestInboxItem>): Boolean {
        items.forEach(::validate)
        val target = file ?: return true
        val absoluteTarget = target.absoluteFile
        val parent = absoluteTarget.parentFile ?: return false
        val pending = File(parent, "${absoluteTarget.name}.pending")
        return runCatching {
            require(parent.isDirectory || parent.mkdirs())
            val encoded = encode(items)
            FileOutputStream(pending).use { output ->
                output.write(encoded)
                output.fd.sync()
            }
            try {
                Files.move(
                    pending.toPath(),
                    absoluteTarget.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(
                    pending.toPath(),
                    absoluteTarget.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                )
            }
            true
        }.getOrElse {
            pending.delete()
            false
        }
    }

    private fun deleteDurableState(): Boolean {
        val target = file?.absoluteFile ?: return true
        val pending = File(target.parentFile, "${target.name}.pending")
        return runCatching {
            listOf(pending, target).forEach { candidate ->
                if (candidate.exists()) candidate.delete()
            }
            !pending.exists() && !target.exists()
        }.getOrDefault(false)
    }

    private fun encode(items: List<DigestInboxItem>): ByteArray {
        val payload = ByteArrayOutputStream().use { bytes ->
            DataOutputStream(bytes).use { output ->
                output.writeInt(items.size)
                items.forEach { item ->
                    output.writeString(item.eventId)
                    output.writeString(item.openToken)
                    output.writeString(item.origin.name)
                    output.writeString(item.sourcePackage)
                    output.writeString(item.title)
                    output.writeString(item.body)
                    output.writeLong(item.routedAtMillis)
                    output.writeBoolean(item.readAtMillis != null)
                    item.readAtMillis?.let(output::writeLong)
                    output.writeBoolean(item.pendingAction != null)
                    item.pendingAction?.let { action ->
                        output.writeString(action.name)
                        output.writeLong(requireNotNull(item.pendingActionAtMillis))
                    }
                    output.writeBoolean(item.learningSnapshot != null)
                    item.learningSnapshot?.let { snapshot ->
                        output.writeNotificationContext(snapshot.context)
                        output.writeString(snapshot.foundationModelId)
                        output.writeLong(snapshot.decidedAtMillis)
                        output.writeLong(snapshot.checkpointIndex)
                        output.writeLong(snapshot.runEpoch)
                        output.writeString(snapshot.studentPrompt)
                        output.writeDoubleArray(snapshot.foundationProbabilitiesFp64)
                        output.writeDoubleArray(snapshot.adaptiveDecisionProbabilities)
                    }
                }
            }
            bytes.toByteArray()
        }
        require(payload.size <= MAX_PAYLOAD_BYTES)
        val digest = MessageDigest.getInstance("SHA-256").digest(payload)
        return ByteArrayOutputStream().use { bytes ->
            DataOutputStream(bytes).use { output ->
                output.writeInt(MAGIC)
                output.writeInt(VERSION)
                output.writeInt(payload.size)
                output.write(payload)
                output.write(digest)
            }
            bytes.toByteArray()
        }
    }

    private fun decode(encoded: ByteArray): List<DigestInboxItem> {
        DataInputStream(ByteArrayInputStream(encoded)).use { input ->
            require(input.readInt() == MAGIC)
            val version = input.readInt()
            require(version == VERSION)
            val payloadSize = input.readInt()
            require(payloadSize in 0..MAX_PAYLOAD_BYTES)
            require(encoded.size.toLong() == HEADER_BYTES + payloadSize.toLong() + DIGEST_BYTES)
            val payload = ByteArray(payloadSize).also(input::readFully)
            val expectedDigest = ByteArray(DIGEST_BYTES).also(input::readFully)
            require(
                MessageDigest.isEqual(
                    expectedDigest,
                    MessageDigest.getInstance("SHA-256").digest(payload),
                ),
            )
            DataInputStream(ByteArrayInputStream(payload)).use { data ->
                val count = data.readInt()
                require(count in 0..MAX_SUPPORTED_ITEMS)
                val items = List(count) {
                    val eventId = data.readString(MAX_EVENT_ID_BYTES)
                    val openToken = data.readString(MAX_OPEN_TOKEN_BYTES)
                    val origin = DigestInboxOrigin.valueOf(data.readString(MAX_ORIGIN_BYTES))
                    val sourcePackage = data.readString(MAX_SOURCE_PACKAGE_BYTES)
                    val title = data.readString(MAX_TITLE_BYTES)
                    val body = data.readString(MAX_BODY_BYTES)
                    val routedAtMillis = data.readLong()
                    val readAtMillis = if (data.readBoolean()) data.readLong() else null
                    val pendingAction = if (data.readBoolean()) {
                        DigestPendingAction.valueOf(data.readString(MAX_PENDING_ACTION_BYTES))
                    } else {
                        null
                    }
                    val pendingActionAtMillis = if (pendingAction == null) null else data.readLong()
                    val learningSnapshot = if (data.readBoolean()) {
                        DigestLearningSnapshot(
                            context = data.readNotificationContext(),
                            foundationModelId = data.readString(MAX_MODEL_ID_BYTES),
                            decidedAtMillis = data.readLong(),
                            checkpointIndex = data.readLong(),
                            runEpoch = data.readLong(),
                            studentPrompt = data.readString(MAX_STUDENT_PROMPT_BYTES),
                            foundationProbabilitiesFp64 = data.readDoubleArray(
                                ROUTE_PROBABILITY_COUNT,
                            ),
                            adaptiveDecisionProbabilities = data.readDoubleArray(
                                ROUTE_PROBABILITY_COUNT,
                            ),
                        )
                    } else {
                        null
                    }
                    DigestInboxItem(
                        eventId = eventId,
                        openToken = openToken,
                        sourcePackage = sourcePackage,
                        title = title,
                        body = body,
                        routedAtMillis = routedAtMillis,
                        origin = origin,
                        readAtMillis = readAtMillis,
                        pendingAction = pendingAction,
                        pendingActionAtMillis = pendingActionAtMillis,
                        learningSnapshot = learningSnapshot,
                    ).also(::validate)
                }
                require(data.available() == 0)
                require(items.distinctBy(DigestInboxItem::eventId).size == items.size)
                return items
            }
        }
    }

    private fun validate(item: DigestInboxItem) {
        requireValidEventId(item.eventId)
        require(isValidOpenToken(item.openToken)) {
            "openToken must contain 1..$MAX_OPEN_TOKEN_CHARS characters"
        }
        require(item.sourcePackage.isNotBlank()) { "sourcePackage must not be blank" }
        require(item.sourcePackage.length <= MAX_SOURCE_PACKAGE_CHARS)
        require(item.sourcePackage.utf8Size() <= MAX_SOURCE_PACKAGE_BYTES)
        require(item.title.isNotBlank()) { "title must not be blank" }
        require(item.title.length <= MAX_TITLE_CHARS)
        require(item.title.utf8Size() <= MAX_TITLE_BYTES)
        require(item.body.length <= MAX_BODY_CHARS)
        require(item.body.utf8Size() <= MAX_BODY_BYTES)
        require(item.routedAtMillis >= 0L) { "routedAtMillis must be non-negative" }
        require(item.readAtMillis == null || item.readAtMillis >= 0L) {
            "readAtMillis must be non-negative"
        }
        require((item.pendingAction == null) == (item.pendingActionAtMillis == null)) {
            "pending action and timestamp must either both be present or both be absent"
        }
        require(item.pendingActionAtMillis == null || item.pendingActionAtMillis >= 0L) {
            "pendingActionAtMillis must be non-negative"
        }
        if (item.origin == DigestInboxOrigin.SYNTHETIC_LAB) {
            require(item.learningSnapshot == null) {
                "Synthetic lab previews must never carry a learnable digest snapshot"
            }
        }
        item.learningSnapshot?.let { snapshot -> validateSnapshot(item, snapshot) }
    }

    private fun validateSnapshot(item: DigestInboxItem, snapshot: DigestLearningSnapshot) {
        val notification = snapshot.context
        require(notification.eventId == item.eventId) {
            "learning snapshot eventId must match the inbox item"
        }
        require(notification.packageName == item.sourcePackage) {
            "learning snapshot package must match the inbox item"
        }
        requireValidEventId(notification.eventId)
        require(notification.packageName.isNotBlank())
        require(notification.packageName.length <= MAX_SOURCE_PACKAGE_CHARS)
        require(notification.packageName.utf8Size() <= MAX_SOURCE_PACKAGE_BYTES)
        require(notification.title.length <= MAX_CONTEXT_TITLE_CHARS)
        require(notification.title.utf8Size() <= MAX_CONTEXT_TITLE_BYTES)
        require(notification.body.length <= MAX_CONTEXT_BODY_CHARS)
        require(notification.body.utf8Size() <= MAX_CONTEXT_BODY_BYTES)
        require(notification.category.length <= MAX_CATEGORY_CHARS)
        require(notification.category.utf8Size() <= MAX_CATEGORY_BYTES)
        require(notification.importance.isFinite())
        require(notification.hourOfDay.isFinite())
        require(notification.postedAtMillis >= 0L)
        notification.caseId?.let { caseId ->
            require(caseId.length <= MAX_CASE_ID_CHARS)
            require(caseId.utf8Size() <= MAX_CASE_ID_BYTES)
        }
        require(snapshot.foundationModelId.isNotBlank())
        require(snapshot.foundationModelId.length <= MAX_MODEL_ID_CHARS)
        require(snapshot.foundationModelId.utf8Size() <= MAX_MODEL_ID_BYTES)
        require(snapshot.decidedAtMillis >= 0L)
        require(snapshot.checkpointIndex >= 0L)
        require(snapshot.runEpoch >= 0L)
        require(snapshot.studentPrompt.isNotBlank())
        require(snapshot.studentPrompt.utf8Size() <= MAX_STUDENT_PROMPT_BYTES)
        validateProbabilities(snapshot.foundationProbabilitiesFp64)
        validateProbabilities(snapshot.adaptiveDecisionProbabilities)
    }

    private fun validateProbabilities(probabilities: DoubleArray) {
        require(probabilities.size == ROUTE_PROBABILITY_COUNT)
        require(probabilities.all { it.isFinite() && it >= 0.0 })
        val total = probabilities.sum()
        require(total.isFinite() && total > 0.0)
    }

    private fun requireValidEventId(eventId: String) {
        require(isValidEventId(eventId)) {
            "eventId must contain 1..$MAX_EVENT_ID_CHARS characters"
        }
    }

    private fun isValidEventId(eventId: String): Boolean =
        eventId.isNotBlank() &&
            eventId.length <= MAX_EVENT_ID_CHARS &&
            eventId.utf8Size() <= MAX_EVENT_ID_BYTES

    private fun isValidOpenToken(openToken: String): Boolean =
        openToken.isNotBlank() &&
            openToken.length <= MAX_OPEN_TOKEN_CHARS &&
            openToken.utf8Size() <= MAX_OPEN_TOKEN_BYTES

    private fun tokensEqual(expected: String, actual: String): Boolean = MessageDigest.isEqual(
        expected.toByteArray(Charsets.UTF_8),
        actual.toByteArray(Charsets.UTF_8),
    )

    private fun friendlySourceLabel(sourcePackage: String): String {
        val segment = sourcePackage.substringAfterLast('.').ifBlank { sourcePackage }
        return segment
            .split(SOURCE_WORD_SEPARATOR)
            .filter(String::isNotBlank)
            .joinToString(" ") { word ->
                word.replaceFirstChar { character ->
                    if (character.isLowerCase()) character.titlecase() else character.toString()
                }
            }
            .ifBlank { sourcePackage }
    }

    private fun DataOutputStream.writeString(value: String) {
        val encoded = value.toByteArray(Charsets.UTF_8)
        writeInt(encoded.size)
        write(encoded)
    }

    private fun DataInputStream.readString(maxBytes: Int): String {
        val size = readInt()
        require(size in 0..maxBytes)
        require(size <= available())
        return ByteArray(size).also(::readFully).toString(Charsets.UTF_8)
    }

    private fun DataOutputStream.writeNotificationContext(context: NotificationContext) {
        writeString(context.eventId)
        writeString(context.packageName)
        writeString(context.title)
        writeString(context.body)
        writeString(context.category)
        writeFloat(context.importance)
        writeString(context.regime.name)
        writeFloat(context.hourOfDay)
        writeLong(context.postedAtMillis)
        writeBoolean(context.caseId != null)
        context.caseId?.let { caseId -> writeString(caseId) }
        writeBoolean(context.isClearable)
        writeBoolean(context.isOngoing)
        writeBoolean(context.isForegroundService)
        writeBoolean(context.isCall)
        writeBoolean(context.isMedia)
        writeBoolean(context.isGroupSummary)
        writeBoolean(context.isNoClear)
        writeBoolean(context.canPublishDigest)
    }

    private fun DataInputStream.readNotificationContext(): NotificationContext =
        NotificationContext(
            eventId = readString(MAX_EVENT_ID_BYTES),
            packageName = readString(MAX_SOURCE_PACKAGE_BYTES),
            title = readString(MAX_CONTEXT_TITLE_BYTES),
            body = readString(MAX_CONTEXT_BODY_BYTES),
            category = readString(MAX_CATEGORY_BYTES),
            importance = readFloat(),
            regime = Regime.valueOf(readString(MAX_REGIME_BYTES)),
            hourOfDay = readFloat(),
            postedAtMillis = readLong(),
            caseId = if (readBoolean()) readString(MAX_CASE_ID_BYTES) else null,
            isClearable = readBoolean(),
            isOngoing = readBoolean(),
            isForegroundService = readBoolean(),
            isCall = readBoolean(),
            isMedia = readBoolean(),
            isGroupSummary = readBoolean(),
            isNoClear = readBoolean(),
            canPublishDigest = readBoolean(),
        )

    private fun DataOutputStream.writeDoubleArray(values: DoubleArray) {
        writeInt(values.size)
        values.forEach(::writeDouble)
    }

    private fun DataInputStream.readDoubleArray(expectedSize: Int): DoubleArray {
        require(readInt() == expectedSize)
        return DoubleArray(expectedSize) { readDouble() }
    }

    private fun String.utf8Size(): Int = toByteArray(Charsets.UTF_8).size

    companion object {
        private val NEWEST_FIRST = compareByDescending<DigestInboxItem> {
            it.routedAtMillis
        }.thenBy(DigestInboxItem::eventId)
        private val SOURCE_WORD_SEPARATOR = Regex("[-_\\s]+")

        private const val DEFAULT_MAX_ITEMS = 100
        private const val MAX_SUPPORTED_ITEMS = 1_000
        private const val SUMMARY_LIMIT = 3

        private const val MAX_EVENT_ID_CHARS = 200
        private const val MAX_OPEN_TOKEN_CHARS = 128
        private const val MAX_SOURCE_PACKAGE_CHARS = 255
        private const val MAX_TITLE_CHARS = 512
        private const val MAX_BODY_CHARS = 4_096
        private const val MAX_CONTEXT_TITLE_CHARS = 4_096
        private const val MAX_CONTEXT_BODY_CHARS = 16_384
        private const val MAX_CATEGORY_CHARS = 512
        private const val MAX_CASE_ID_CHARS = 512
        private const val MAX_MODEL_ID_CHARS = 512
        private const val MAX_EVENT_ID_BYTES = MAX_EVENT_ID_CHARS * 4
        private const val MAX_OPEN_TOKEN_BYTES = MAX_OPEN_TOKEN_CHARS * 4
        private const val MAX_ORIGIN_BYTES = 32
        private const val MAX_SOURCE_PACKAGE_BYTES = MAX_SOURCE_PACKAGE_CHARS * 4
        private const val MAX_TITLE_BYTES = MAX_TITLE_CHARS * 4
        private const val MAX_BODY_BYTES = MAX_BODY_CHARS * 4
        private const val MAX_CONTEXT_TITLE_BYTES = MAX_CONTEXT_TITLE_CHARS * 4
        private const val MAX_CONTEXT_BODY_BYTES = MAX_CONTEXT_BODY_CHARS * 4
        private const val MAX_CATEGORY_BYTES = MAX_CATEGORY_CHARS * 4
        private const val MAX_CASE_ID_BYTES = MAX_CASE_ID_CHARS * 4
        private const val MAX_MODEL_ID_BYTES = MAX_MODEL_ID_CHARS * 4
        private const val MAX_REGIME_BYTES = 64
        private const val MAX_PENDING_ACTION_BYTES = 16
        private const val MAX_STUDENT_PROMPT_BYTES = 128 * 1024
        private const val ROUTE_PROBABILITY_COUNT = 3
        private const val MAX_RECORD_BYTES =
            MAX_EVENT_ID_BYTES + MAX_OPEN_TOKEN_BYTES + MAX_ORIGIN_BYTES +
                MAX_SOURCE_PACKAGE_BYTES +
                MAX_TITLE_BYTES + MAX_BODY_BYTES + MAX_EVENT_ID_BYTES +
                MAX_SOURCE_PACKAGE_BYTES + MAX_CONTEXT_TITLE_BYTES + MAX_CONTEXT_BODY_BYTES +
                MAX_CATEGORY_BYTES + MAX_CASE_ID_BYTES + MAX_MODEL_ID_BYTES +
                MAX_REGIME_BYTES + MAX_STUDENT_PROMPT_BYTES +
                ROUTE_PROBABILITY_COUNT * 16 + 256
        private const val MAX_PAYLOAD_BYTES = 4 + MAX_SUPPORTED_ITEMS * MAX_RECORD_BYTES
        private const val HEADER_BYTES = 12L
        private const val DIGEST_BYTES = 32
        private const val MAX_ENCODED_BYTES = HEADER_BYTES + MAX_PAYLOAD_BYTES + DIGEST_BYTES

        private const val MAGIC = 0x44494749 // DIGI
        private const val VERSION_WITH_ORIGIN = 5
        private const val VERSION = 6
    }
}
