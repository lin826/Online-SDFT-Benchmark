package ai.onlinesdft.router.state

import ai.onlinesdft.router.demo.PersonaScenarioCatalog
import ai.onlinesdft.router.demo.PersonaFeedbackSimulator
import ai.onlinesdft.router.demo.PreferenceShiftCatalog
import ai.onlinesdft.router.lfm.LfmCompactPromptCodec
import ai.onlinesdft.router.model.DecisionSnapshot
import ai.onlinesdft.router.model.EvaluationMetrics
import ai.onlinesdft.router.model.EvaluationTruth
import ai.onlinesdft.router.model.ExecutionConstraint
import ai.onlinesdft.router.model.FactualFeedback
import ai.onlinesdft.router.model.FeedbackSource
import ai.onlinesdft.router.model.FeatureEncoder
import ai.onlinesdft.router.model.NotificationContext
import ai.onlinesdft.router.model.FoundationModelPhase
import ai.onlinesdft.router.model.FoundationModelStatus
import ai.onlinesdft.router.model.LiquidOrtFoundationModel
import ai.onlinesdft.router.model.OnlineSdftLearner
import ai.onlinesdft.router.model.Outcome
import ai.onlinesdft.router.model.PrequentialMetricsStore
import ai.onlinesdft.router.model.PrequentialScorer
import ai.onlinesdft.router.model.LoraReplayStore
import ai.onlinesdft.router.model.Route
import ai.onlinesdft.router.notification.ArchiveNotificationPublisher
import ai.onlinesdft.router.notification.RouterNotificationListenerService
import ai.onlinesdft.router.notification.DigestInboxItem
import ai.onlinesdft.router.notification.DigestInboxOrigin
import ai.onlinesdft.router.notification.DigestInboxStore
import ai.onlinesdft.router.notification.DigestPendingAction
import ai.onlinesdft.router.notification.DigestNotificationPublisher
import ai.onlinesdft.router.notification.NotificationSnapshotFactory
import ai.onlinesdft.router.notification.toDecision
import ai.onlinesdft.router.notification.toDigestLearningSnapshot
import ai.onlinesdft.router.telemetry.BackendState
import ai.onlinesdft.router.telemetry.TelemetryClient
import ai.onlinesdft.router.telemetry.TelemetryPayloads
import android.content.Context
import android.content.Intent
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.Executors

class DemoRuntime(private val context: Context) {
    private val _state = MutableStateFlow(DemoUiState())
    val state: StateFlow<DemoUiState> = _state.asStateFlow()
    private val foundationRuntime = LiquidOrtFoundationModel(
        context = context,
        onStatus = ::setFoundationStatus,
    )
    private val learner = OnlineSdftLearner(
        foundationRuntime = foundationRuntime,
        replayStore = LoraReplayStore(
            java.io.File(context.filesDir, "model/lora-replay-v1.bin"),
        ),
        promptCodec = LfmCompactPromptCodec(),
    )
    private val scorer = PrequentialScorer(
        PrequentialMetricsStore(
            java.io.File(context.filesDir, "metrics/prequential-v1.bin"),
        ),
    )
    private val digestInbox = DigestInboxStore(
        java.io.File(context.filesDir, "digest/saved-for-later-v1.bin"),
    )
    private val decisions = LinkedHashMap<String, DecisionSnapshot>()
    private val events = LinkedHashMap<String, RoutedEventUi>()
    private val feedbackFingerprints = mutableSetOf<String>()
    private val digestActionsInFlight = mutableSetOf<String>()
    private val epochPreferences = context.getSharedPreferences(
        "runtime_epoch_v1",
        Context.MODE_PRIVATE,
    )
    private var runEpoch = epochPreferences.getLong("run_epoch", 0L)
    private var resetInProgress = false
    private val reservedLiveEventIds = epochPreferences
        .getStringSet("reserved_live_event_ids", emptySet())
        .orEmpty()
        .toMutableSet()
    private val worker = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "sdft-learning").apply { isDaemon = true }
    }
    private val telemetry = TelemetryClient(context.filesDir, onState = ::setBackendState)

    data class DigestRecordHandle(val eventId: String, val openToken: String)

    private data class DigestActionWork(
        val actionKey: String,
        val item: DigestInboxItem,
        val decision: DecisionSnapshot,
        val feedback: FactualFeedback,
    )

    fun showRouter() = selectPage(DemoPage.ROUTER)

    @Synchronized
    fun showRouterEvent(rawEventId: String?): Boolean {
        val eventId = rawEventId?.trim().orEmpty()
        if (eventId.isEmpty() || !events.containsKey(eventId)) {
            Log.w(
                PROOF_TAG,
                "ROUTER_EVENT_REQUEST_REJECTED reason=" +
                    if (eventId.isEmpty()) "missing_event_id" else "unknown_event_id",
            )
            return false
        }
        var requestId = 0L
        updateState {
            requestId = it.pageRequestId + 1L
            it.copy(
                selectedPage = DemoPage.ROUTER,
                pageRequestId = requestId,
                highlightedEventId = eventId,
            )
        }
        Log.i(
            PROOF_TAG,
            "PAGE_REQUESTED page=ROUTER request=$requestId highlighted_event_id=$eventId",
        )
        return true
    }

    fun showDigest() = selectPage(DemoPage.DIGEST)

    fun showScores() = selectPage(DemoPage.SCORES)

    /** Selects a card only after the user has brought it into view and tapped it. */
    fun selectRouterEvent(rawEventId: String?) {
        val eventId = rawEventId?.trim().orEmpty()
        val current = _state.value
        if (
            current.selectedPage != DemoPage.ROUTER ||
            eventId.isEmpty() ||
            current.events.none { it.decision.context.eventId == eventId }
        ) {
            Log.w(
                PROOF_TAG,
                "ROUTER_EVENT_SELECTION_REJECTED source=user_tap reason=" +
                    when {
                        current.selectedPage != DemoPage.ROUTER -> "router_not_visible"
                        eventId.isEmpty() -> "missing_event_id"
                        else -> "unknown_event_id"
                    },
            )
            return
        }
        updateState { it.copy(highlightedEventId = eventId) }
        Log.i(
            PROOF_TAG,
            "ROUTER_EVENT_SELECTED event_id=$eventId source=user_tap " +
                "page_request=${current.pageRequestId}",
        )
    }

    /** Reserves a collision-free identity before a live decision is sealed. */
    @Synchronized
    fun disambiguateLiveNotification(notification: NotificationContext): NotificationContext {
        val reserved = buildSet {
            addAll(reservedLiveEventIds)
            addAll(decisions.keys)
            addAll(events.keys)
            digestInbox.items().forEach { add(it.eventId) }
        }
        val unique = NotificationSnapshotFactory.disambiguateEventId(notification, reserved)
        reservedLiveEventIds.add(unique.eventId)
        if (!epochPreferences.edit()
                .putStringSet("reserved_live_event_ids", reservedLiveEventIds.toSet())
                .commit()
        ) {
            Log.w(PROOF_TAG, "Unable to persist live notification identity reservation")
        }
        return unique
    }

    init {
        updateState {
            it.copy(
                runId = telemetry.runId,
                modelStatus = learner.status(),
                evaluation = scorer.current(),
                digestItems = digestInbox.items(),
                digestSummary = digestInbox.summary(),
            )
        }
        ArchiveNotificationPublisher.sync(context, digestInbox.items())
        foundationRuntime.preload()
    }

    @Synchronized
    fun syncArchiveNotifications() {
        ArchiveNotificationPublisher.sync(context, digestInbox.items())
    }

    @Synchronized
    fun route(
        notification: NotificationContext,
        truth: EvaluationTruth? = null,
        labMode: Boolean = false,
        emitTelemetry: Boolean = true,
    ): DecisionSnapshot {
        val decision = learner.decide(notification).copy(runEpoch = runEpoch)
        // Prequential ordering: the action and score are frozen here, before
        // any callback can be created or any optimizer step can run.
        val evaluation = scorer.score(decision, truth, labMode)
        decisions[notification.eventId] = decision
        events[notification.eventId] = RoutedEventUi(
            decision = decision,
            lessonStatus = if (
                decision.executionConstraint != ExecutionConstraint.NONE
            ) {
                "Left alone — this kind of notification is never changed"
            } else if (!labMode && decision.chosenRoute == Route.INTERRUPT) {
                "Showing now"
            } else if (!labMode && decision.chosenRoute == Route.LATER) {
                "Moving to Saved for later"
            } else if (!labMode) {
                "Silencing"
            } else {
                "Waiting to see what you do"
            },
            syntheticLab = labMode,
        )
        trimEvents()
        updateState {
            it.copy(
                events = chronologicalEventValues(events),
                modelStatus = learner.status(),
                evaluation = evaluation ?: it.evaluation,
                scoredHistory = if (evaluation == null) {
                    it.scoredHistory
                } else {
                    appendScoredPoint(it.scoredHistory, evaluation)
                },
                statusMessage = if (
                    decision.executionConstraint != ExecutionConstraint.NONE
                ) {
                    "Left this one alone"
                } else if (!labMode && decision.chosenRoute == Route.INTERRUPT) {
                    "Shown now"
                } else if (!labMode && decision.chosenRoute == Route.LATER) {
                    "Saved for later"
                } else if (!labMode) {
                    "Silenced"
                } else {
                    "Decision made — now watching what you do"
                },
            )
        }
        if (emitTelemetry) telemetry.decision(decision, labMode)
        return decision
    }

    /**
     * Route a live listener snapshot only if its listener generation is still
     * valid. The predicate is checked while holding the same monitor as reset,
     * so a callback captured before Reset cannot enter the new model/run.
     */
    @Synchronized
    fun routeIfValid(
        notification: NotificationContext,
        validityCheck: () -> Boolean,
        afterCommit: (DecisionSnapshot) -> String? = { null },
    ): DecisionSnapshot? {
        if (resetInProgress || !validityCheck()) return null
        val decision = route(notification, emitTelemetry = false)
        val routeExecution = afterCommit(decision)
        telemetry.decision(decision, labMode = false, routeExecution = routeExecution)
        return decision
    }

    /** Run an Android side effect in the same reset epoch as its decision. */
    @Synchronized
    fun <T> withCurrentDecision(decision: DecisionSnapshot, block: () -> T): T? {
        if (decision.runEpoch != runEpoch) return null
        return block()
    }

    fun submitFeedback(feedback: FactualFeedback) {
        worker.execute { applyFeedback(feedback, fallbackDecision = null) }
    }

    fun submitFeedback(decision: DecisionSnapshot, feedback: FactualFeedback) {
        worker.execute { applyFeedback(feedback, fallbackDecision = decision) }
    }

    fun recordDigest(
        decision: DecisionSnapshot,
        allowStaleDelivery: Boolean = false,
    ): DigestRecordHandle? = recordSavedDecision(
        decision,
        DigestInboxOrigin.LIVE_NOTIFICATION,
        allowStaleDelivery,
    )

    fun recordArchive(
        decision: DecisionSnapshot,
        allowStaleDelivery: Boolean = false,
    ): DigestRecordHandle? = recordSavedDecision(
        decision,
        DigestInboxOrigin.ROUTER_ARCHIVE,
        allowStaleDelivery,
    )

    @Synchronized
    private fun recordSavedDecision(
        decision: DecisionSnapshot,
        origin: DigestInboxOrigin,
        allowStaleDelivery: Boolean,
    ): DigestRecordHandle? {
        val expectedRoute = when (origin) {
            DigestInboxOrigin.LIVE_NOTIFICATION -> Route.LATER
            DigestInboxOrigin.ROUTER_ARCHIVE -> Route.ARCHIVE
            DigestInboxOrigin.SYNTHETIC_LAB -> return null
        }
        if (decision.chosenRoute != expectedRoute) return null
        val currentEpoch = isCurrentDecision(decision)
        if (!currentEpoch && !allowStaleDelivery) return null
        val existingItems = digestInbox.items()
        existingItems.firstOrNull { it.eventId == decision.context.eventId }?.let { existing ->
            val sameSource = existing.origin == origin &&
                existing.sourcePackage == decision.context.packageName &&
                existing.title == decision.context.title.ifBlank { "Saved notification" }.take(512) &&
                existing.body == decision.context.body.take(4_096) &&
                existing.routedAtMillis == decision.decidedAtMillis
            if (sameSource) return DigestRecordHandle(existing.eventId, existing.openToken)
            // A pre-reset cancellation can confirm after a new run has reused
            // the publisher's human-readable id. Preserve that old delivery as
            // a separate UI-only row; never alias it to the replacement.
            if (!allowStaleDelivery) return null
        }
        val storageEventId = if (existingItems.any { it.eventId == decision.context.eventId }) {
            uniqueDigestEventId(decision.context.eventId, existingItems.mapTo(mutableSetOf()) { it.eventId })
        } else {
            decision.context.eventId
        }
        val beforeIds = existingItems.mapTo(mutableSetOf()) { it.eventId }
        val openToken = UUID.randomUUID().toString()
        val learningSnapshot = if (
            currentEpoch &&
            decision.foundationAvailable &&
            decision.foundationModelId.isNotBlank() &&
            decision.foundationModelId.length <= 512 &&
            decision.context.title.length <= 4_096 &&
            decision.context.body.length <= 16_384 &&
            decision.context.category.length <= 512 &&
            (decision.context.caseId?.length ?: 0) <= 512 &&
            decision.studentPrompt.isNotBlank() &&
            decision.foundationProbabilitiesFp64.size == Route.entries.size &&
            decision.adaptiveDecisionProbabilities.size == Route.entries.size
        ) {
            decision.toDigestLearningSnapshot()
        } else {
            null
        }
        val stored = runCatching {
            digestInbox.upsert(
                DigestInboxItem(
                    eventId = storageEventId,
                    openToken = openToken,
                    sourcePackage = decision.context.packageName,
                    title = decision.context.title.ifBlank { "Saved notification" }.take(512),
                    body = decision.context.body.take(4_096),
                    routedAtMillis = decision.decidedAtMillis,
                    origin = origin,
                    learningSnapshot = learningSnapshot,
                ),
            )
        }.getOrDefault(false)
        if (stored) {
            val currentItems = digestInbox.items()
            val afterIds = currentItems.mapTo(mutableSetOf()) { it.eventId }
            (beforeIds - afterIds).forEach { evictedEventId ->
                DigestNotificationPublisher.cancel(context, evictedEventId)
                if (existingItems.any {
                        it.eventId == evictedEventId &&
                            it.origin == DigestInboxOrigin.ROUTER_ARCHIVE
                    }
                ) {
                    ArchiveNotificationPublisher.cancel(context, evictedEventId, currentItems)
                }
            }
            ArchiveNotificationPublisher.sync(context, currentItems)
            updateState {
                it.copy(
                    digestItems = currentItems,
                    digestSummary = digestInbox.summary(),
                    statusMessage = if (origin == DigestInboxOrigin.ROUTER_ARCHIVE) {
                        "Silenced · ${decision.context.title}"
                    } else {
                        "Saved for later · ${decision.context.title}"
                    },
                )
            }
            Log.i(
                PROOF_TAG,
                "SAVED_INBOX_ITEM event_id=$storageEventId " +
                    "origin=${origin.name} source_package=${decision.context.packageName} " +
                    "saved=true",
            )
        }
        return if (stored) DigestRecordHandle(storageEventId, openToken) else null
    }

    private fun uniqueDigestEventId(base: String, reserved: Set<String>): String {
        while (true) {
            val suffix = UUID.randomUUID().toString()
            val candidate = "${base.take(163)}-$suffix"
            if (candidate !in reserved) return candidate
        }
    }

    fun openDigest(eventId: String) = openDigestAuthorized(eventId, expectedToken = null)

    private fun openDigestAuthorized(eventId: String, expectedToken: String?) {
        var authorized = false
        val transition = synchronized(this) {
            val item = digestInbox.items().firstOrNull { it.eventId == eventId }
                ?: return@synchronized null
            if (expectedToken != null && !tokensEqual(item.openToken, expectedToken)) {
                return@synchronized null
            }
            authorized = true
            if (!item.isUnread) return@synchronized null
            val claimed = digestInbox.claimAction(
                eventId = eventId,
                openToken = item.openToken,
                action = DigestPendingAction.OPEN,
                atMillis = System.currentTimeMillis(),
            ) ?: return@synchronized null
            // The inbox snapshot is the correlation authority. Falling back to
            // a same-named live decision could train a new run from a stale
            // pre-reset delivery obligation.
            val decision = claimed.toDecision(runEpoch)
                ?.takeIf { it.chosenRoute == Route.LATER }
            if (decision == null) {
                markDigestReadLocked(claimed)
                return@synchronized null
            }
            val actionKey = "digest:$eventId:${claimed.openToken}"
            if (!digestActionsInFlight.add(actionKey)) return@synchronized null
            DigestActionWork(
                actionKey,
                claimed,
                decision,
                digestFeedback(
                    decision,
                    Outcome.OPENED_DIGEST,
                    Route.LATER,
                    requireNotNull(claimed.pendingActionAtMillis),
                ),
            )
        }
        if (!authorized) return
        showDigest()
        DigestNotificationPublisher.cancel(context, eventId)
        transition?.let { work ->
            worker.execute {
                val stillCurrent = synchronized(this) {
                    digestRecordMatches(work.item) && isCurrentDecision(work.decision)
                }
                val completed = stillCurrent && runCatching {
                    applyFeedback(
                        work.feedback,
                        fallbackDecision = work.decision,
                    )
                }.onFailure { error ->
                    Log.e(PROOF_TAG, "Saved-item open feedback failed", error)
                }.getOrDefault(FeedbackApplyResult.REJECTED).completed
                synchronized(this) {
                    digestActionsInFlight.remove(work.actionKey)
                    if (completed) markDigestReadLocked(work.item)
                }
            }
        }
    }

    fun openDigestFromNotification(eventId: String, openToken: String) {
        openDigestAuthorized(eventId, expectedToken = openToken)
    }

    /** Keeps alert publication linearized with Reset and inbox removal. */
    @Synchronized
    fun <T> withDigestRecord(
        eventId: String,
        openToken: String,
        block: () -> T,
    ): T? {
        val expected = digestInbox.items()
            .firstOrNull { it.eventId == eventId }
            ?.openToken
            ?: return null
        if (
            !MessageDigest.isEqual(
                expected.toByteArray(Charsets.UTF_8),
                openToken.toByteArray(Charsets.UTF_8),
            )
        ) return null
        return block()
    }

    fun dismissDigest(eventId: String) = dismissDigestAuthorized(eventId, expectedToken = null)

    fun dismissDigestFromNotification(eventId: String, openToken: String) {
        dismissDigestAuthorized(eventId, expectedToken = openToken)
    }

    private fun dismissDigestAuthorized(eventId: String, expectedToken: String?) {
        var authorized = false
        val transition = synchronized(this) {
            val item = digestInbox.items().firstOrNull { it.eventId == eventId }
                ?: return@synchronized null
            if (expectedToken != null && !tokensEqual(item.openToken, expectedToken)) {
                return@synchronized null
            }
            authorized = true
            if (!item.isUnread) {
                removeDigestLocked(item)
                return@synchronized null
            }
            val claimed = digestInbox.claimAction(
                eventId = eventId,
                openToken = item.openToken,
                action = DigestPendingAction.REMOVE,
                atMillis = System.currentTimeMillis(),
            ) ?: return@synchronized null
            val decision = claimed.toDecision(runEpoch)
                ?.takeIf { it.chosenRoute == Route.LATER }
            if (decision == null) {
                removeDigestLocked(claimed)
                return@synchronized null
            }
            val actionKey = "digest:$eventId:${claimed.openToken}"
            if (!digestActionsInFlight.add(actionKey)) return@synchronized null
            DigestActionWork(
                actionKey,
                claimed,
                decision,
                digestFeedback(
                    decision,
                    Outcome.DELETED_FROM_DIGEST,
                    Route.ARCHIVE,
                    requireNotNull(claimed.pendingActionAtMillis),
                ),
            )
        }
        if (!authorized) return
        DigestNotificationPublisher.cancel(context, eventId)
        transition?.let { work ->
            worker.execute {
                val stillCurrent = synchronized(this) {
                    digestRecordMatches(work.item) && isCurrentDecision(work.decision)
                }
                val completed = stillCurrent && runCatching {
                    applyFeedback(
                        work.feedback,
                        fallbackDecision = work.decision,
                    )
                }.onFailure { error ->
                    Log.e(PROOF_TAG, "Saved-item removal feedback failed", error)
                }.getOrDefault(FeedbackApplyResult.REJECTED).completed
                synchronized(this) {
                    digestActionsInFlight.remove(work.actionKey)
                    if (completed) removeDigestLocked(work.item)
                }
            }
        }
    }

    /**
     * A model-selected Archive remains locally inspectable. The user's first
     * action on that row closes the otherwise censored learning loop against
     * the exact sealed Archive decision, including after process recreation.
     */
    fun submitArchivedPreference(eventId: String, preferred: Route) {
        val transition = synchronized(this) {
            val item = digestInbox.items().firstOrNull {
                it.eventId == eventId &&
                    it.origin == DigestInboxOrigin.ROUTER_ARCHIVE &&
                    it.isUnread
            } ?: return@synchronized null
            val action = when (preferred) {
                Route.INTERRUPT -> DigestPendingAction.SHOW_NEXT
                Route.LATER -> DigestPendingAction.OPEN
                Route.ARCHIVE -> DigestPendingAction.KEEP_SILENT
            }
            val claimed = digestInbox.claimAction(
                eventId = item.eventId,
                openToken = item.openToken,
                action = action,
                atMillis = System.currentTimeMillis(),
            ) ?: return@synchronized null
            val decision = claimed.toDecision(runEpoch)
                ?.takeIf { it.chosenRoute == Route.ARCHIVE }
            if (decision == null) {
                markDigestReadLocked(claimed)
                return@synchronized null
            }
            val actionKey = "archive:$eventId:${claimed.openToken}"
            if (!digestActionsInFlight.add(actionKey)) return@synchronized null
            DigestActionWork(
                actionKey = actionKey,
                item = claimed,
                decision = decision,
                feedback = archivedFeedback(
                    decision = decision,
                    preferred = preferred,
                    observedAtMillis = requireNotNull(claimed.pendingActionAtMillis),
                ),
            )
        } ?: return
        showDigest()
        if (preferred == Route.LATER) launchSourceApp(transition.item.sourcePackage)
        updateState {
            it.copy(
                statusMessage = when (preferred) {
                    Route.INTERRUPT -> "Got it — showing these next time…"
                    Route.LATER -> "Got it — saving these for later…"
                    Route.ARCHIVE -> "Got it — keeping these quiet…"
                },
            )
        }
        worker.execute {
            val stillCurrent = synchronized(this) {
                digestRecordMatches(transition.item) && isCurrentDecision(transition.decision)
            }
            val completed = stillCurrent && runCatching {
                applyFeedback(
                    transition.feedback,
                    fallbackDecision = transition.decision,
                )
            }.onFailure { error ->
                Log.e(PROOF_TAG, "Archived-item preference feedback failed", error)
            }.getOrDefault(FeedbackApplyResult.REJECTED).completed
            synchronized(this) {
                digestActionsInFlight.remove(transition.actionKey)
                if (completed) {
                    if (preferred == Route.ARCHIVE) {
                        removeDigestLocked(transition.item)
                    } else {
                        markDigestReadLocked(transition.item)
                    }
                }
            }
        }
    }

    /**
     * Digest engagement is learned before the durable inbox transition. If the
     * process dies while work is queued, the unread/visible item remains and
     * the customer can retry instead of silently losing the callback.
     */
    @Synchronized
    private fun markDigestReadLocked(item: DigestInboxItem) {
        if (!digestRecordMatches(item)) return
        if (!digestInbox.markRead(item.eventId, System.currentTimeMillis())) return
        if (item.origin == DigestInboxOrigin.ROUTER_ARCHIVE) {
            ArchiveNotificationPublisher.cancel(context, item.eventId, digestInbox.items())
        }
        updateState {
            it.copy(
                digestItems = digestInbox.items(),
                digestSummary = digestInbox.summary(),
                statusMessage = "Marked as read · ${item.title}",
            )
        }
    }

    @Synchronized
    private fun removeDigestLocked(item: DigestInboxItem) {
        if (!digestRecordMatches(item)) return
        if (!digestInbox.remove(item.eventId)) return
        if (item.origin == DigestInboxOrigin.ROUTER_ARCHIVE) {
            ArchiveNotificationPublisher.cancel(context, item.eventId, digestInbox.items())
        }
        updateState {
            it.copy(
                digestItems = digestInbox.items(),
                digestSummary = digestInbox.summary(),
                statusMessage = "Removed from Saved for later · ${item.title}",
            )
        }
    }

    private fun digestRecordMatches(item: DigestInboxItem): Boolean {
        val current = digestInbox.items().firstOrNull { it.eventId == item.eventId } ?: return false
        return tokensEqual(current.openToken, item.openToken)
    }

    private fun tokensEqual(left: String, right: String): Boolean = MessageDigest.isEqual(
        left.toByteArray(Charsets.UTF_8),
        right.toByteArray(Charsets.UTF_8),
    )

    private fun digestFeedback(
        decision: DecisionSnapshot,
        outcome: Outcome,
        observedSelection: Route,
        observedAtMillis: Long = System.currentTimeMillis(),
    ): FactualFeedback = FactualFeedback(
        eventId = decision.context.eventId,
        executedRoute = Route.LATER,
        outcome = outcome,
        observedSelection = observedSelection,
        delayMinutes = (
            (observedAtMillis - decision.decidedAtMillis).coerceAtLeast(0L) /
                60_000L
            ).toInt(),
        source = FeedbackSource.DIGEST_CALLBACK,
        observedAtMillis = observedAtMillis,
    )

    private fun archivedFeedback(
        decision: DecisionSnapshot,
        preferred: Route,
        observedAtMillis: Long,
    ): FactualFeedback = FactualFeedback(
        eventId = decision.context.eventId,
        executedRoute = Route.ARCHIVE,
        outcome = if (preferred == Route.LATER) {
            Outcome.OPENED_DIGEST
        } else {
            Outcome.EXPLICIT_USER_CORRECTION
        },
        observedSelection = preferred,
        delayMinutes = (
            (observedAtMillis - decision.decidedAtMillis).coerceAtLeast(0L) /
                60_000L
            ).toInt(),
        source = if (preferred == Route.LATER) {
            FeedbackSource.DIGEST_CALLBACK
        } else {
            FeedbackSource.EXPLICIT_USER_CORRECTION
        },
        explicitPreference = preferred.takeUnless { it == Route.LATER },
        observedAtMillis = observedAtMillis,
    )

    private fun launchSourceApp(sourcePackage: String) {
        val intent = context.packageManager.getLaunchIntentForPackage(sourcePackage) ?: return
        runCatching {
            context.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        }.onFailure { error ->
            Log.w(PROOF_TAG, "Unable to open archived notification source app", error)
        }
    }

    @Synchronized
    fun confirmRouteExecution(decision: DecisionSnapshot): Boolean {
        if (!isCurrentDecision(decision)) return false
        val message = when (decision.chosenRoute) {
            Route.LATER -> "Saved for later"
            Route.ARCHIVE -> "Silenced — you can still find it in Saved"
            Route.INTERRUPT -> return false
        }
        events[decision.context.eventId]?.let { event ->
            events[decision.context.eventId] = event.copy(lessonStatus = message)
        }
        updateState {
            it.copy(
                events = chronologicalEventValues(events),
                statusMessage = message,
            )
        }
        return true
    }

    @Synchronized
    fun failRouteExecution(decision: DecisionSnapshot, reason: String): Boolean {
        if (!isCurrentDecision(decision)) return false
        // The reason is engineering detail; it belongs in the proof log, not on
        // a screen a person reads.
        Log.i(PROOF_TAG, "ROUTE_NOT_APPLIED event_id=${decision.context.eventId} reason=$reason")
        val message = "Left where it was"
        events[decision.context.eventId]?.let { event ->
            events[decision.context.eventId] = event.copy(lessonStatus = message)
        }
        updateState {
            it.copy(
                events = chronologicalEventValues(events),
                statusMessage = message,
            )
        }
        return true
    }

    @Synchronized
    fun isCurrentDecision(decision: DecisionSnapshot): Boolean = decision.runEpoch == runEpoch

    @Synchronized
    fun purgeNotificationDataForLockdown() {
        val epochAdvanced = advanceRunEpoch()
        decisions.clear()
        events.clear()
        feedbackFingerprints.clear()
        digestActionsInFlight.clear()
        reservedLiveEventIds.clear()
        epochPreferences.edit().remove("reserved_live_event_ids").commit()
        val digestCleared = digestInbox.clear()
        DigestNotificationPublisher.clearAll(context)
        updateState {
            it.copy(
                events = emptyList(),
                digestItems = emptyList(),
                digestSummary = digestInbox.summary(),
                pendingCorrectionEventIds = emptySet(),
                completedCorrectionEventIds = emptySet(),
                statusMessage = if (digestCleared && epochAdvanced) {
                    "Phone locked — notification content cleared from this screen"
                } else {
                    "Phone locked — content hidden, but some of it could not be erased"
                },
            )
        }
    }

    fun submitExplicitCorrection(eventId: String?, preferred: Route) {
        if (_state.value.labRunning) return
        // Resolve the compact FEEDBACK(route) shortcut from the immutable UI
        // snapshot. Looking it up under the runtime monitor would put the
        // Compose/broadcast caller back behind slow model inference.
        val requestedEventId = eventId?.trim()?.takeIf(String::isNotEmpty)
            ?: _state.value.events.lastOrNull()?.decision?.context?.eventId
            ?: return
        val pendingKey = requestedEventId
        val preferredLabel = when (preferred) {
            Route.INTERRUPT -> "Show now"
            Route.LATER -> "Later"
            Route.ARCHIVE -> "Keep silent"
        }
        var accepted = false
        updateState { current ->
            if (
                current.labRunning ||
                current.pendingCorrectionEventIds.isNotEmpty() ||
                pendingKey in current.completedCorrectionEventIds
            ) {
                current
            } else {
                accepted = true
                current.copy(
                    pendingCorrectionEventIds = current.pendingCorrectionEventIds + pendingKey,
                    statusMessage = "Learning “$preferredLabel” from you…",
                )
            }
        }
        if (!accepted) return

        // Decision lookup and the full teacher forward stay on the learning
        // worker. The Compose UI thread must never wait on the runtime monitor
        // while applyFeedback owns it for model inference and persistence.
        val observedAtMillis = System.currentTimeMillis()
        worker.execute {
            try {
                val decision = synchronized(this@DemoRuntime) {
                    decisions[requestedEventId]
                }
                if (decision == null) {
                    updateState { current ->
                        if (pendingKey !in current.pendingCorrectionEventIds) current else {
                            current.copy(statusMessage = "That notification is no longer available")
                        }
                    }
                    return@execute
                }
                val result = applyFeedback(
                    FactualFeedback(
                        eventId = decision.context.eventId,
                        executedRoute = decision.chosenRoute,
                        outcome = Outcome.EXPLICIT_USER_CORRECTION,
                        observedSelection = preferred,
                        delayMinutes = 0,
                        source = FeedbackSource.EXPLICIT_USER_CORRECTION,
                        explicitPreference = preferred,
                        observedAtMillis = observedAtMillis,
                    ),
                    fallbackDecision = decision,
                )
                // Reset/lockdown use the same monitor. Holding it while the
                // result is published prevents an old correction from
                // restoring completion state or status in a new epoch.
                synchronized(this@DemoRuntime) {
                    if (decision.runEpoch == runEpoch) updateState { current ->
                        if (pendingKey !in current.pendingCorrectionEventIds) {
                            current
                        } else when (result) {
                            FeedbackApplyResult.UPDATED -> current.copy(
                                completedCorrectionEventIds =
                                    current.completedCorrectionEventIds + decision.context.eventId,
                                statusMessage = "Learned — “$preferredLabel” next time",
                            )
                            FeedbackApplyResult.RECORDED -> current.copy(
                                completedCorrectionEventIds =
                                    current.completedCorrectionEventIds + decision.context.eventId,
                                statusMessage = "Noted — “$preferredLabel”",
                            )
                            FeedbackApplyResult.DUPLICATE -> current.copy(
                                completedCorrectionEventIds =
                                    current.completedCorrectionEventIds + decision.context.eventId,
                                statusMessage =
                                    "Already set to “$preferredLabel”",
                            )
                            FeedbackApplyResult.REJECTED -> current.copy(
                                statusMessage =
                                    "That notification is no longer available",
                            )
                        }
                    }
                }
            } catch (error: Throwable) {
                Log.e(PROOF_TAG, "Explicit correction failed", error)
                updateState { current ->
                    if (pendingKey !in current.pendingCorrectionEventIds) current else {
                        current.copy(
                            statusMessage =
                                "Could not save that just now — nothing was lost",
                        )
                    }
                }
            } finally {
                updateState { current ->
                    current.copy(
                        pendingCorrectionEventIds =
                            current.pendingCorrectionEventIds - pendingKey,
                    )
                }
            }
        }
    }

    fun runAcceleratedLab() {
        if (_state.value.labRunning) return
        updateState {
            it.copy(
                labRunning = true,
                statusMessage = "Practice round starting…",
            )
        }
        worker.execute {
            if (learner.status().updateIndex != 0L) {
                updateState {
                    it.copy(
                        labRunning = false,
                        statusMessage = "Tap Reset first — the practice round starts from scratch",
                    )
                }
                return@execute
            }
            val session = UUID.randomUUID().toString().take(8)
            val cases = PersonaScenarioCatalog.cases(session)
            var savedPreviewCount = 0
            learner.prepareFrozenBaseline(cases.map { it.context })
            Log.i(
                PROOF_TAG,
                "LAB_STARTED session=$session decisions_expected=${cases.size} " +
                    "stream=${PersonaScenarioCatalog.STREAM_VERSION} checkpoint=0",
            )
            updateState {
                it.copy(statusMessage = "Practice round running — learning as it goes")
            }
            cases.forEachIndexed { index, scenario ->
                val decision = route(scenario.context, scenario.truth, labMode = true)
                if (saveSyntheticLabLaterPreview(decision)) savedPreviewCount += 1
                // The learner receives a simulated observable response, never
                // EvaluationTruth or its utility vector.
                applyFeedback(
                    if (index < PersonaScenarioCatalog.CURRICULUM_SIZE) {
                        PersonaFeedbackSimulator.scriptedCorrection(
                            decision,
                            scenario.simulatedPreference,
                        )
                    } else {
                        PersonaFeedbackSimulator.observe(
                            decision,
                            scenario.simulatedPreference,
                        )
                    },
                    fallbackDecision = decision,
                )
                Thread.sleep(55)
            }
            val finalEvaluation = scorer.current()
            val finalModel = learner.status()
            Log.i(
                PROOF_TAG,
                "LAB_COMPLETE decisions=${finalEvaluation.decisions} " +
                    "adaptive_accuracy=${formatMetric(finalEvaluation.onlineAccuracy)} " +
                    "frozen_accuracy=${formatMetric(finalEvaluation.baseAccuracy)} " +
                    "adaptive_regret=${formatMetric(finalEvaluation.cumulativeRegret)} " +
                    "frozen_regret=${formatMetric(finalEvaluation.baseCumulativeRegret)} " +
                    "lora_norm=${formatMetric(finalModel.loraNorm)} " +
                    "checkpoint=${finalModel.loraCheckpointChecksum} " +
                    "saved_previews=$savedPreviewCount",
            )
            updateState {
                it.copy(
                    labRunning = false,
                    statusMessage =
                        "Correction curriculum + probe pass complete · " +
                            "$savedPreviewCount Later previews added to Saved",
                )
            }
        }
    }

    /**
     * Drives the preference-shift demo: the same five notifications repeat
     * while the person's preferences change halfway through. Separate from
     * [runAcceleratedLab] so the pinned research stream stays untouched.
     */
    fun runPreferenceShiftDemo() {
        if (_state.value.labRunning) return
        updateState {
            it.copy(labRunning = true, statusMessage = "Learning your preferences…")
        }
        worker.execute {
            if (learner.status().updateIndex != 0L) {
                updateState {
                    it.copy(
                        labRunning = false,
                        statusMessage = "Start over first — this demo begins from scratch",
                    )
                }
                return@execute
            }
            val session = UUID.randomUUID().toString().take(8)
            val cases = PreferenceShiftCatalog.cases(session)
            learner.prepareFrozenBaseline(cases.map { it.context })
            Log.i(
                PROOF_TAG,
                "SHIFT_STARTED session=$session decisions_expected=${cases.size} " +
                    "stream=${PreferenceShiftCatalog.STREAM_VERSION} checkpoint=0",
            )
            cases.forEachIndexed { index, scenario ->
                val shifted = PreferenceShiftCatalog.isShifted(index)
                if (index == cases.size / 2) {
                    Log.i(PROOF_TAG, "SHIFT_PREFERENCES_CHANGED at_decision=$index")
                }
                updateState {
                    it.copy(
                        statusMessage = if (shifted) {
                            "Your preferences changed — catching up"
                        } else {
                            "Learning your preferences…"
                        },
                    )
                }
                val decision = route(scenario.context, scenario.truth, labMode = true)
                saveSyntheticLabLaterPreview(decision)
                applyFeedback(
                    if (PreferenceShiftCatalog.isTaught(index)) {
                        PersonaFeedbackSimulator.scriptedCorrection(
                            decision,
                            scenario.simulatedPreference,
                        )
                    } else {
                        PersonaFeedbackSimulator.observe(decision, scenario.simulatedPreference)
                    },
                    fallbackDecision = decision,
                )
                Thread.sleep(SHIFT_STEP_PAUSE_MILLIS)
            }
            val finalEvaluation = scorer.current()
            Log.i(
                PROOF_TAG,
                "SHIFT_COMPLETE decisions=${finalEvaluation.decisions} " +
                    "accuracy=${formatMetric(finalEvaluation.onlineAccuracy)} " +
                    "checkpoint=${learner.status().loraCheckpointChecksum}",
            )
            updateState {
                it.copy(labRunning = false, statusMessage = "Caught up with your new preferences")
            }
        }
    }

    @Synchronized
    private fun saveSyntheticLabLaterPreview(decision: DecisionSnapshot): Boolean {
        val item = syntheticLabSavedPreview(
            decision = decision,
            openToken = UUID.randomUUID().toString(),
        ) ?: return false
        if (!digestInbox.upsert(item)) return false
        updateState {
            it.copy(
                digestItems = digestInbox.items(),
                digestSummary = digestInbox.summary(),
            )
        }
        return true
    }

    @Synchronized
    fun decision(eventId: String): DecisionSnapshot? = decisions[eventId]

    fun setListenerConnected(connected: Boolean) {
        updateState {
            it.copy(
                listenerConnected = connected,
                statusMessage = if (connected) {
                    "Connected — watching your notifications"
                } else {
                    "Not connected — notifications are untouched"
                },
            )
        }
    }

    fun reset() {
        var digestResetPersisted = true
        val invalidated = synchronized(this) {
            if (!advanceRunEpoch()) return@synchronized false
            resetInProgress = true
            RouterNotificationListenerService.clearPendingDecisions()
            decisions.clear()
            events.clear()
            feedbackFingerprints.clear()
            digestActionsInFlight.clear()
            reservedLiveEventIds.clear()
            epochPreferences.edit().remove("reserved_live_event_ids").commit()
            digestResetPersisted = digestInbox.clear()
            DigestNotificationPublisher.clearAll(context)
            updateState {
                it.copy(
                    events = emptyList(),
                    digestItems = emptyList(),
                    digestSummary = digestInbox.summary(),
                    pendingCorrectionEventIds = emptySet(),
                    completedCorrectionEventIds = emptySet(),
                    statusMessage = "Starting over…",
                )
            }
            true
        }
        if (!invalidated) {
            updateState {
                it.copy(statusMessage = "Could not start over just now — try again")
            }
            return
        }
        worker.execute {
            var evaluationResetPersisted = true
            val modelReset = runCatching { learner.reset() }.isSuccess
            evaluationResetPersisted = runCatching { scorer.reset() }.getOrDefault(false)
            val newRunId = runCatching { telemetry.startNewRun() }.getOrDefault(telemetry.runId)
            synchronized(this) { resetInProgress = false }
            Log.i(
                PROOF_TAG,
                "MODEL_RESET checkpoint=${learner.status().checksum} run_id=$newRunId " +
                    "evaluation_reset_persisted=$evaluationResetPersisted",
            )
            updateState {
                DemoUiState(
                    listenerConnected = it.listenerConnected,
                    backendState = it.backendState,
                    modelStatus = learner.status(),
                    statusMessage = if (
                        modelReset && evaluationResetPersisted && digestResetPersisted
                    ) {
                        "Starting fresh — everything it learned about you was erased"
                    } else {
                        "Started fresh — some saved data could not be erased"
                    },
                    runId = newRunId,
                    selectedPage = it.selectedPage,
                    pageRequestId = it.pageRequestId,
                )
            }
        }
    }

    private fun selectPage(page: DemoPage) {
        var requestId = 0L
        updateState {
            requestId = it.pageRequestId + 1L
            it.copy(
                selectedPage = page,
                pageRequestId = requestId,
                highlightedEventId = null,
            )
        }
        Log.i(PROOF_TAG, "PAGE_REQUESTED page=${page.name} request=$requestId")
    }

    @Synchronized
    private fun advanceRunEpoch(): Boolean {
        val next = if (runEpoch == Long.MAX_VALUE) 0L else runEpoch + 1L
        if (!epochPreferences.edit().putLong("run_epoch", next).commit()) {
            Log.e(PROOF_TAG, "Unable to persist runtime epoch $next")
            return false
        }
        runEpoch = next
        return true
    }

    fun pageVisible(page: DemoPage, requestId: Long, highlightedEventId: String?) {
        val current = _state.value
        if (
            current.selectedPage != page ||
            current.pageRequestId != requestId ||
            current.highlightedEventId != highlightedEventId
        ) return
        val highlightProof = highlightedEventId?.let { " highlighted_event_id=$it" }.orEmpty()
        Log.i(PROOF_TAG, "PAGE_ACTIVE page=${page.name} request=$requestId$highlightProof")
        if (page == DemoPage.ROUTER && highlightedEventId != null) {
            Log.i(
                PROOF_TAG,
                "ROUTER_EVENT_ACTIVE request=$requestId event_id=$highlightedEventId",
            )
        }
    }

    @Synchronized
    private fun applyFeedback(
        feedback: FactualFeedback,
        fallbackDecision: DecisionSnapshot?,
    ): FeedbackApplyResult {
        // A persisted digest callback is sealed to the exact decision that
        // created the inbox item. It must win over any newer live decision
        // that happens to reuse the publisher-supplied event id.
        val decision = fallbackDecision ?: decisions[feedback.eventId]
            ?: return FeedbackApplyResult.REJECTED
        if (!isCurrentDecision(decision)) return FeedbackApplyResult.REJECTED
        val fingerprint = listOf(
            feedback.eventId,
            feedback.outcome.name,
            feedback.source.name,
            feedback.explicitPreference?.name,
        ).joinToString(":")
        if (fingerprint in feedbackFingerprints) return FeedbackApplyResult.DUPLICATE
        telemetry.feedback(feedback)
        val training = learner.learn(decision, feedback)
        val modelStatus = learner.status()
        if (training != null) {
            capturePostTrainingDistribution(
                eventId = feedback.eventId,
                updateIndex = training.updateIndex,
                replaceExisting = feedback.source == FeedbackSource.EXPLICIT_USER_CORRECTION,
            )
        }
        val bufferedForWarmup = training == null && modelStatus.lastLessonBuffered
        if (bufferedForWarmup) {
            Log.i(
                PROOF_TAG,
                "ON_DEVICE_WARMUP event_id=${feedback.eventId} " +
                    "replay_size=${modelStatus.replaySize} " +
                    "remaining=${modelStatus.warmupRemaining} " +
                    "checkpoint=${modelStatus.loraCheckpointChecksum} " +
                    "thread=${Thread.currentThread().name}",
            )
        }
        if (training != null) {
            Log.i(
                PROOF_TAG,
                "ON_DEVICE_UPDATE event_id=${feedback.eventId} " +
                    "update=${training.updateIndex} " +
                    "loss_before=${formatMetric(training.lossBefore)} " +
                    "loss_after=${formatMetric(training.lossAfter)} " +
                    "grad_norm=${training.gradientNorm?.let(::formatMetric) ?: "unavailable"} " +
                    "delta_from_reset=${formatMetric(training.deltaNorm)} " +
                    "callback_update_norm=${formatMetric(training.callbackUpdateNorm)} " +
                    "optimizer_steps=${training.optimizerSteps} " +
                    "optimizer_step_proofs=${TelemetryPayloads.optimizerStepProofs(training.optimizerStepProofs)} " +
                    "checksum_before=${training.checksumBefore} " +
                    "checksum_after=${training.checksumAfter} " +
                    "thread=${Thread.currentThread().name}",
            )
        }
        events[feedback.eventId]?.let { event ->
            events[feedback.eventId] = event.copy(
                feedback = feedback,
                training = training,
                lessonStatus = if (bufferedForWarmup) {
                    "Noted — still gathering a few first examples"
                } else if (training == null) {
                    "Noted — not clear enough to learn from"
                } else {
                    "Learned from this one"
                },
            )
        }
        if (training != null) telemetry.trainingUpdate(training)
        updateState {
            it.copy(
                events = chronologicalEventValues(events),
                lastTraining = training ?: it.lastTraining,
                modelStatus = modelStatus,
                statusMessage = if (bufferedForWarmup) {
                    "Noted — ${modelStatus.warmupRemaining} more examples before it starts learning"
                } else if (training == null) {
                    "Noted — not clear enough to learn from"
                } else {
                    "Learned from you · ${training.updateIndex} lessons so far"
                },
            )
        }
        feedbackFingerprints.add(fingerprint)
        return if (training == null) {
            FeedbackApplyResult.RECORDED
        } else {
            FeedbackApplyResult.UPDATED
        }
    }

    private fun trimEvents() {
        while (events.size > MAX_VISIBLE_EVENTS) {
            val key = events.keys.first()
            events.remove(key)
        }
        // Keep a larger bounded decision history than the visible card list so
        // callbacks from long-lived ongoing items and Later digests do not lose
        // their sealed action merely because the UI has received 40 newer rows.
        while (decisions.size > MAX_DECISION_HISTORY) {
            val key = decisions.keys.first()
            decisions.remove(key)
        }
    }

    /** Freeze this callback's distribution immediately after its own update. */
    private fun capturePostTrainingDistribution(
        eventId: String,
        updateIndex: Long,
        replaceExisting: Boolean = false,
    ) {
        val refreshed = capturedEventDistribution(
            events = events,
            eventId = eventId,
            updateIndex = updateIndex,
            replaceExisting = replaceExisting,
            reevaluate = learner::reevaluateDistribution,
        )
        events.clear()
        events.putAll(refreshed)
        events[eventId]
            ?.takeIf { it.distributionUpdateIndex == updateIndex }
            ?.postTrainingProbabilities
            ?.let { probabilities ->
                Log.i(
                    PROOF_TAG,
                    "DISTRIBUTION_SNAPSHOT event_id=$eventId update=$updateIndex " +
                        "show_now=${formatMetric(probabilities[Route.INTERRUPT.ordinal].toDouble())} " +
                        "later=${formatMetric(probabilities[Route.LATER.ordinal].toDouble())} " +
                        "keep_silent=${formatMetric(probabilities[Route.ARCHIVE.ordinal].toDouble())}",
                )
            }
    }

    private fun setBackendState(backendState: BackendState) {
        updateState { it.copy(backendState = backendState) }
    }

    private fun setFoundationStatus(status: FoundationModelStatus) {
        if (status.phase == FoundationModelPhase.READY) {
            Log.i(
                PROOF_TAG,
                "FOUNDATION_READY model_id=${status.modelId} " +
                    "precision=${status.precision} bundle_bytes=${status.totalBytes}",
            )
        }
        updateState { current ->
            current.copy(
                modelStatus = current.modelStatus.copy(foundationStatus = status),
                statusMessage = when (status.phase) {
                    FoundationModelPhase.NOT_STARTED -> "Starting up…"
                    FoundationModelPhase.LOADING -> "Getting ready…"
                    FoundationModelPhase.READY -> "Ready — learning on this phone only"
                    FoundationModelPhase.ERROR ->
                        "Paused — your notifications are passing through untouched"
                },
            )
        }
        if (status.phase == FoundationModelPhase.READY) {
            runCatching {
                worker.execute {
                    val refreshed = learner.status()
                    updateState { current -> current.copy(modelStatus = refreshed) }
                }
            }
        }
    }

    private inline fun updateState(transform: (DemoUiState) -> DemoUiState) {
        synchronized(_state) {
            _state.value = transform(_state.value)
        }
    }

    /**
     * Records one scored decision for the progress trend. The evaluator only
     * advances on labelled decisions, so an unchanged decision count leaves the
     * history alone rather than appending a duplicate point.
     */
    private fun appendScoredPoint(
        history: List<ScoredDecisionPoint>,
        evaluation: EvaluationMetrics,
    ): List<ScoredDecisionPoint> {
        if (evaluation.decisions <= (history.lastOrNull()?.decisions ?: 0)) return history
        val point = ScoredDecisionPoint(
            decisions = evaluation.decisions,
            adaptiveCorrect = evaluation.correct,
            baseCorrect = evaluation.baseCorrect,
        )
        return (history + point).takeLast(MAX_SCORED_HISTORY)
    }

    private fun formatMetric(value: Double): String = "%.6f".format(java.util.Locale.US, value)

    private enum class FeedbackApplyResult {
        UPDATED,
        RECORDED,
        DUPLICATE,
        REJECTED;

        val completed: Boolean
            get() = this != REJECTED
    }

    companion object {
        private const val PROOF_TAG = "OnlineSdftProof"
        private const val MAX_VISIBLE_EVENTS = 40
        private const val MAX_DECISION_HISTORY = 512
        private const val MAX_SCORED_HISTORY = 200

        /** Paced so each decision is legible on screen in a recorded demo. */
        private const val SHIFT_STEP_PAUSE_MILLIS = 900L
    }
}
