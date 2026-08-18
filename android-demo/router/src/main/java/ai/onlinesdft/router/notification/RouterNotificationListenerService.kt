package ai.onlinesdft.router.notification

import ai.onlinesdft.router.OnlineSdftApplication
import ai.onlinesdft.router.model.DecisionSnapshot
import ai.onlinesdft.router.model.FactualFeedback
import ai.onlinesdft.router.model.FeedbackSource
import ai.onlinesdft.router.model.Outcome
import ai.onlinesdft.router.model.Route
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.UUID

internal fun backlogKeysToIgnore(
    activeKeys: Set<String>,
    pendingCancellationKeys: Set<String>,
): Set<String> = activeKeys - pendingCancellationKeys

/**
 * User-authorized listener for eligible cross-package notifications.
 * Android invokes this after posting, so LATER/ARCHIVE are honest post-time
 * cancellation/reposting decisions rather than guaranteed pre-alert blocking.
 */
class RouterNotificationListenerService : NotificationListenerService() {
    private class PendingCancellation(
        val decision: DecisionSnapshot,
        @Volatile var confirmed: Boolean = false,
    ) {
        @Volatile
        var timeoutReported = false
        @Volatile
        var timeoutFuture: ScheduledFuture<*>? = null
    }

    private val keyToDecision = ConcurrentHashMap<String, DecisionSnapshot>()
    private val pendingCancellations = ConcurrentHashMap<String, PendingCancellation>()
    private val ignoredBacklogKeys = ConcurrentHashMap.newKeySet<String>()
    private val listenerExecutor: ScheduledExecutorService =
        Executors.newSingleThreadScheduledExecutor { runnable ->
            Thread(runnable, "sdft-notification-events").apply { isDaemon = true }
        }
    private val eventGeneration = AtomicLong(0L)
    private val listenerConnected = AtomicBoolean(false)
    private val lockdownPending = AtomicBoolean(false)
    private val actuationLock = Any()
    private val listenerSessionId = UUID.randomUUID().toString().take(8)
    private val activeResumeCounter = AtomicInteger(0)

    override fun onCreate() {
        super.onCreate()
        PendingCancellationStore.load(this).forEach { stored ->
            val pending = PendingCancellation(stored.decision, stored.confirmed)
            pendingCancellations[stored.key] = pending
            keyToDecision[stored.key] = stored.decision
        }
        synchronized(lifecycleLock) {
            activeInstance = this
        }
    }

    private fun clearActiveInstance() {
        synchronized(lifecycleLock) {
            if (activeInstance === this) {
                activeInstance = null
            }
        }
    }

    override fun onDestroy() {
        synchronized(actuationLock) {
            listenerConnected.set(false)
            eventGeneration.incrementAndGet()
            // The app-private pending journal permits an identity-checked retry
            // or confirmed delivery after recreation; ambiguous absence fails
            // closed rather than guessing that Android honored the request.
            clearActiveInstance()
            clearPendingState(reportFailure = false)
        }
        if (lockdownPending.getAndSet(false)) {
            if (!clearPendingJournalForLockdown()) lockdownPending.set(true)
            OnlineSdftApplication.runtime(this).purgeNotificationDataForLockdown()
        }
        listenerExecutor.shutdownNow()
        super.onDestroy()
    }

    override fun onListenerConnected() {
        super.onListenerConnected()
        synchronized(actuationLock) { listenerConnected.set(true) }
        val generation = eventGeneration.get()
        val ranking = currentRanking
        val active = try {
            activeNotifications?.mapNotNull { sbn ->
                if (NotificationSnapshotFactory.eligible(sbn, packageName)) {
                    sbn.key to NotificationSnapshotFactory.create(sbn, ranking)
                } else {
                    null
                }
            }
        } catch (error: RuntimeException) {
            Log.w(PROOF_TAG, "Unable to snapshot active notifications", error)
            null
        }
        if (active != null) {
            ignoredBacklogKeys.clear()
            ignoredBacklogKeys.addAll(
                backlogKeysToIgnore(
                    active.mapTo(linkedSetOf()) { it.first },
                    pendingCancellations.keys,
                ),
            )
        }
        submitListenerTask listenerTask@{
            if (!isGenerationActive(generation)) return@listenerTask
            OnlineSdftApplication.runtime(this).setListenerConnected(true)
            if (active == null) {
                abandonUnarmedCancellations(
                    "active notification state was unavailable after reconnect",
                )
                return@listenerTask
            }
            reconcilePendingCancellations(active.associate { it.first to it.second })
            Log.i(
                PROOF_TAG,
                "ACTIVE_BACKLOG_IGNORED count=${ignoredBacklogKeys.size} " +
                    "new_notifications_only=true",
            )
        }
    }

    override fun onListenerDisconnected() {
        synchronized(actuationLock) {
            listenerConnected.set(false)
            eventGeneration.incrementAndGet()
        }
        submitListenerTask {
            // Preserve any cancellation already issued. Its removal callback
            // can still arrive with the old generation; reconnect either
            // retries a still-active source or reports an uncertain outcome.
            OnlineSdftApplication.runtime(this).setListenerConnected(false)
        }
        super.onListenerDisconnected()
    }

    override fun onNotificationPosted(sbn: StatusBarNotification, rankingMap: RankingMap) {
        if (!NotificationSnapshotFactory.eligible(sbn, packageName)) return
        if (ignoredBacklogKeys.contains(sbn.key)) return
        val generation = eventGeneration.get()
        val key = sbn.key
        val context = runCatching { NotificationSnapshotFactory.create(sbn, rankingMap) }
            .getOrElse { error ->
                Log.w(PROOF_TAG, "NOTIFICATION_SNAPSHOT_REJECTED source_package=${sbn.packageName}", error)
                return
            }
        submitListenerTask { processPosted(key, context, generation) }
    }

    private fun processPosted(
        key: String,
        capturedContext: ai.onlinesdft.router.model.NotificationContext,
        generation: Long,
        fromActiveSet: Boolean = false,
    ) {
        if (!isGenerationActive(generation)) return
        // Ongoing services, calls, media, and progress notifications commonly
        // update the same StatusBarNotification key. One active key represents
        // one sealed decision; updates must not double-count telemetry/training.
        if (keyToDecision.containsKey(key)) return
        val runtime = OnlineSdftApplication.runtime(this)
        // Reconnect correlation is keyed by Android's source key/post time in
        // the pending-actuation journal. A publisher-supplied event id is not
        // an identity and may be reused by another active notification.
        val context = runtime.disambiguateLiveNotification(
            capturedContext.copy(
                // A process restart loses in-memory decision correlation while the
                // backend run survives. Give an already-active item a fresh event
                // identity so reconnect seeding cannot wedge the outbox on a 409.
                eventId = if (fromActiveSet) {
                    "resume-$listenerSessionId-${activeResumeCounter.incrementAndGet()}"
                } else {
                    capturedContext.eventId
                },
                canPublishDigest = DigestNotificationPublisher.canPublish(this),
            ),
        )
        Log.i(
            PROOF_TAG,
            "CROSS_APP_RECEIVED event_id=${context.eventId} " +
                "source_package=${context.packageName} listener_package=$packageName",
        )
        runtime.routeIfValid(
            notification = context,
            validityCheck = { isGenerationActive(generation) },
        ) { decision ->
            synchronized(actuationLock) {
                if (!isGenerationActive(generation)) {
                    if (decision.chosenRoute == Route.INTERRUPT) {
                        Log.i(
                            PROOF_TAG,
                            "ACTION_RESULT event_id=${context.eventId} " +
                                "original_visible=true digest_posted=false " +
                                "listener_disconnected_after_commit=true",
                        )
                        return@synchronized visibleExecutionState(decision)
                    }
                    runtime.failRouteExecution(
                        decision,
                        "notification listener disconnected before actuation",
                    )
                    Log.w(
                        PROOF_TAG,
                        "ACTION_RESULT event_id=${context.eventId} " +
                            "original_visible=true digest_posted=false " +
                            "cancellation=not_requested_listener_disconnected",
                    )
                    return@synchronized "not_applied_listener_disconnected"
                }
                // Registration and Android actuation run under both the
                // runtime reset monitor and the listener lifecycle barrier.
                keyToDecision[key] = decision
                Log.i(
                    PROOF_TAG,
                    "ROUTE_COMMITTED event_id=${context.eventId} " +
                        "source_package=${context.packageName} " +
                        "recommended=${decision.recommendedRoute.name} " +
                        "applied=${decision.chosenRoute.name} " +
                        "constraint=${decision.executionConstraint.wireName} " +
                        "ongoing=${context.isOngoing} fgs=${context.isForegroundService} " +
                        "call=${context.isCall} media=${context.isMedia} " +
                        "checkpoint=${decision.checkpointIndex} " +
                        "thread=${Thread.currentThread().name}",
                )
                when (decision.chosenRoute) {
                    Route.INTERRUPT -> {
                        Log.i(
                            PROOF_TAG,
                            "ACTION_RESULT event_id=${context.eventId} " +
                                "original_visible=true digest_posted=false",
                        )
                        visibleExecutionState(decision)
                    }
                    Route.LATER -> if (NotificationSnapshotFactory.isTrustedDemoTimeout(context)) {
                        Log.i(
                            PROOF_TAG,
                            "ACTION_RESULT event_id=${context.eventId} " +
                                "original_visible=true digest_posted=false " +
                                "timeout_pending=true demo_timeout_ms=${context.demoTimeoutMillis} " +
                                "semantic_delay_minutes=${context.semanticDelayMinutes}",
                        )
                        "trusted_timeout_pending"
                    } else if (requestCancellation(key, decision)) {
                        "cancellation_requested"
                    } else {
                        "cancellation_request_failed"
                    }
                    Route.ARCHIVE -> if (requestCancellation(key, decision)) {
                        "cancellation_requested"
                    } else {
                        "cancellation_request_failed"
                    }
                }
            }
        }
    }

    override fun onNotificationRemoved(
        sbn: StatusBarNotification,
        rankingMap: RankingMap,
        reason: Int,
    ) {
        val key = sbn.key
        if (ignoredBacklogKeys.remove(key)) return
        val postedAtMillis = sbn.postTime
        if (reason == REASON_LOCKDOWN) {
            var journalCleared: Boolean
            synchronized(actuationLock) {
                // Linearize privacy invalidation with any route side effect so
                // a queued decision cannot cancel a notification after Android
                // has entered lockdown.
                lockdownPending.set(true)
                eventGeneration.incrementAndGet()
                clearPendingState(reportFailure = false)
                journalCleared = clearPendingJournalForLockdown()
            }
            val accepted = submitListenerTask {
                if (!journalCleared) journalCleared = clearPendingJournalForLockdown()
                OnlineSdftApplication.runtime(this).purgeNotificationDataForLockdown()
                if (journalCleared) {
                    lockdownPending.set(false)
                } else {
                    Log.e(PROOF_TAG, "LOCKDOWN_JOURNAL_CLEAR_FAILED routing_remains_disabled=true")
                }
            }
            if (!accepted && lockdownPending.getAndSet(false)) {
                journalCleared = journalCleared || clearPendingJournalForLockdown()
                OnlineSdftApplication.runtime(this).purgeNotificationDataForLockdown()
                if (!journalCleared) lockdownPending.set(true)
            }
            return
        }
        if (
            reason == REASON_LISTENER_CANCEL
        ) {
            synchronized(actuationLock) {
                val pending = pendingCancellations[key]
                if (
                    pending != null &&
                    pending.decision.context.postedAtMillis == postedAtMillis
                ) {
                    // Seal disk and memory under the same lifecycle barrier as
                    // lockdown/reconciliation before yielding to the worker.
                    val durable = PendingCancellationStore.markConfirmed(this, key)
                    pending.confirmed = true
                    if (!durable) {
                        Log.e(
                            PROOF_TAG,
                            "CONFIRMATION_JOURNAL_FAILED " +
                                "event_id=${pending.decision.context.eventId}",
                        )
                    }
                }
            }
        }
        val generation = eventGeneration.get()
        submitListenerTask { processRemoved(key, postedAtMillis, reason, generation) }
    }

    private fun processRemoved(
        key: String,
        postedAtMillis: Long,
        reason: Int,
        generation: Long,
    ) {
        var claimedCancellation: PendingCancellation? = null
        val decision = synchronized(actuationLock) {
            val pendingCancellation = pendingCancellations[key]?.takeIf {
                it.decision.context.postedAtMillis == postedAtMillis
            }
            // A confirmed pre-reset LATER cancellation remains a delivery
            // obligation even though its model epoch is stale. Ordinary callbacks
            // from an invalidated generation are dropped.
            if (!isGenerationActive(generation) && pendingCancellation == null) return
            val activeDecision = keyToDecision[key]?.takeIf {
                it.context.postedAtMillis == postedAtMillis
            }
            val claimedDecision = pendingCancellation?.decision ?: activeDecision ?: return
            if (pendingCancellation != null) {
                if (!pendingCancellations.remove(key, pendingCancellation)) return
                keyToDecision.remove(key, claimedDecision)
                pendingCancellation.timeoutFuture?.cancel(false)
                claimedCancellation = pendingCancellation
            } else {
                keyToDecision.remove(key, claimedDecision)
            }
            claimedDecision
        }
        if (claimedCancellation != null) {
            // cancelNotification() has no return value. Only the listener-cancel
            // callback confirms that Android actually removed the source. Wait
            // for confirmation before creating a Later digest, avoiding an
            // original-plus-digest duplicate on protected surfaces.
            if (reason == REASON_LISTENER_CANCEL) {
                claimedCancellation.confirmed = true
                if (fulfillCancellation(decision, reconciledAfterDisconnect = false)) {
                    removePendingJournal(key, decision)
                }
            } else {
                removePendingJournal(key, decision)
                OnlineSdftApplication.runtime(this).failRouteExecution(
                    decision,
                    "cancellation was not confirmed (${reason.wireName()})",
                )
            }
            return
        }
        val runtime = OnlineSdftApplication.runtime(this)
        if (
            decision.chosenRoute == Route.LATER &&
            reason == REASON_TIMEOUT &&
            NotificationSnapshotFactory.isTrustedDemoTimeout(decision.context)
        ) {
            if (!runtime.isCurrentDecision(decision)) return
            val observedAtMillis = System.currentTimeMillis()
            val wallElapsedMillis =
                (observedAtMillis - decision.context.postedAtMillis).coerceAtLeast(0L)
            if (!fulfillTrustedTimeoutLater(decision, wallElapsedMillis)) return
            val feedback = FactualFeedback(
                eventId = decision.context.eventId,
                executedRoute = Route.LATER,
                outcome = Outcome.TIMED_OUT_UNTOUCHED,
                observedSelection = Route.LATER,
                delayMinutes = requireNotNull(decision.context.semanticDelayMinutes),
                source = FeedbackSource.ANDROID_CALLBACK,
                observedAtMillis = observedAtMillis,
            )
            Log.i(
                PROOF_TAG,
                "ANDROID_FEEDBACK event_id=${decision.context.eventId} " +
                    "source_package=${decision.context.packageName} " +
                    "reason=${reason.wireName()} outcome=${feedback.outcome.name} " +
                    "observed_selection=LATER interaction=none " +
                    "semantic_delay_minutes=${feedback.delayMinutes} " +
                    "demo_timeout_ms=${decision.context.demoTimeoutMillis} " +
                    "wall_elapsed_ms=$wallElapsedMillis " +
                    "feedback_source=${feedback.source.name}",
            )
            runtime.submitFeedback(decision, feedback)
            return
        }
        if (decision.chosenRoute != Route.INTERRUPT) return
        if (!runtime.isCurrentDecision(decision)) return
        val elapsedMinutes = (
            (System.currentTimeMillis() - decision.decidedAtMillis).coerceAtLeast(0) / 60_000L
        ).toInt()
        val feedback = when (reason) {
            REASON_CLICK -> FactualFeedback(
                eventId = decision.context.eventId,
                executedRoute = Route.INTERRUPT,
                outcome = if (elapsedMinutes <= 1) {
                    Outcome.OPENED_IMMEDIATELY
                } else {
                    Outcome.OPENED_AFTER_DELAY
                },
                observedSelection = if (elapsedMinutes <= 1) Route.INTERRUPT else Route.LATER,
                delayMinutes = elapsedMinutes,
                source = FeedbackSource.ANDROID_CALLBACK,
                observedAtMillis = System.currentTimeMillis(),
            )
            REASON_CANCEL, REASON_CANCEL_ALL -> FactualFeedback(
                eventId = decision.context.eventId,
                executedRoute = Route.INTERRUPT,
                outcome = Outcome.DELETED_NOTIFICATION,
                observedSelection = Route.ARCHIVE,
                delayMinutes = elapsedMinutes,
                source = FeedbackSource.ANDROID_CALLBACK,
                observedAtMillis = System.currentTimeMillis(),
            )
            REASON_TIMEOUT -> if (
                NotificationSnapshotFactory.isTrustedDemoTimeout(decision.context)
            ) FactualFeedback(
                eventId = decision.context.eventId,
                executedRoute = Route.INTERRUPT,
                outcome = Outcome.TIMED_OUT_UNTOUCHED,
                // The product's explicit timeout policy treats an untouched
                // expiry as a request to defer a similar future notification.
                observedSelection = Route.LATER,
                delayMinutes = requireNotNull(decision.context.semanticDelayMinutes),
                source = FeedbackSource.ANDROID_CALLBACK,
                observedAtMillis = System.currentTimeMillis(),
            ) else return
            // App cancellation, grouping, channel changes, and other system
            // reasons do not reveal a user's route preference. Do not turn
            // them into a learner callback.
            else -> return
        }
        Log.i(
            PROOF_TAG,
            "ANDROID_FEEDBACK event_id=${decision.context.eventId} " +
                "source_package=${decision.context.packageName} " +
                "reason=${reason.wireName()} outcome=${feedback.outcome.name} " +
                "observed_selection=${feedback.observedSelection?.name ?: "UNKNOWN"} " +
                "interaction=${if (reason == REASON_TIMEOUT) "none" else "user"} " +
                if (reason == REASON_TIMEOUT) {
                    "semantic_delay_minutes=${feedback.delayMinutes} " +
                        "demo_timeout_ms=${decision.context.demoTimeoutMillis} "
                } else {
                    ""
                } +
                "feedback_source=${feedback.source.name}",
        )
        runtime.submitFeedback(decision, feedback)
    }

    private fun fulfillTrustedTimeoutLater(
        decision: DecisionSnapshot,
        wallElapsedMillis: Long,
    ): Boolean {
        val runtime = OnlineSdftApplication.runtime(this)
        var saved = false
        val handled = runtime.withCurrentDecision(decision) {
            synchronized(actuationLock) {
                if (lockdownPending.get()) return@synchronized false
                val digestDelivery = DigestNotificationPublisher.publish(this, decision)
                saved = digestDelivery.saved
                if (saved) {
                    runtime.confirmRouteExecution(decision)
                } else {
                    runtime.failRouteExecution(
                        decision,
                        "trusted timeout expired, but the Saved item could not be stored",
                    )
                }
                Log.i(
                    PROOF_TAG,
                    "ACTION_RESULT event_id=${decision.context.eventId} " +
                        "original_visible=false " +
                        "digest_posted=${digestDelivery.alertPosted} " +
                        "inbox_saved=${digestDelivery.saved} timeout_confirmed=true " +
                        "semantic_delay_minutes=${decision.context.semanticDelayMinutes} " +
                        "demo_timeout_ms=${decision.context.demoTimeoutMillis} " +
                        "wall_elapsed_ms=$wallElapsedMillis",
                )
                saved
            }
        }
        return handled == true && saved
    }

    private fun requestCancellation(key: String, decision: DecisionSnapshot): Boolean {
        if (!PendingCancellationStore.save(this, key, decision)) {
            OnlineSdftApplication.runtime(this).failRouteExecution(
                decision,
                "could not persist the cancellation obligation",
            )
            return false
        }
        val pending = PendingCancellation(decision)
        pendingCancellations.put(key, pending)?.timeoutFuture?.cancel(false)
        return try {
            // Arm correlation before asking Android to cancel. If destruction
            // has already shut down the executor, no destructive call occurs.
            pending.timeoutFuture = scheduleCancellationTimeout(key, pending)
            cancelNotification(key)
            true
        } catch (error: RuntimeException) {
            pendingCancellations.remove(key, pending)
            pending.timeoutFuture?.cancel(false)
            removePendingJournal(key, decision)
            OnlineSdftApplication.runtime(this).failRouteExecution(
                decision,
                "Android cancellation request failed",
            )
            Log.w(PROOF_TAG, "Cancellation request failed for $key", error)
            false
        }
    }

    private fun visibleExecutionState(decision: DecisionSnapshot): String =
        if (decision.executionConstraint != ai.onlinesdft.router.model.ExecutionConstraint.NONE) {
            "platform_protected_pass_through"
        } else {
            "left_visible"
        }

    private fun scheduleCancellationTimeout(
        key: String,
        pending: PendingCancellation,
    ): ScheduledFuture<*> = listenerExecutor.schedule(
        {
            if (pendingCancellations[key] !== pending) return@schedule
            pending.timeoutReported = true
            OnlineSdftApplication.runtime(this).failRouteExecution(
                pending.decision,
                "Android did not confirm cancellation",
            )
            Log.w(
                PROOF_TAG,
                "ACTION_RESULT event_id=${pending.decision.context.eventId} " +
                    "original_visible=unknown digest_posted=false " +
                    "cancellation=unconfirmed",
            )
        },
        CANCELLATION_CONFIRMATION_TIMEOUT_MILLIS,
        TimeUnit.MILLISECONDS,
    )

    private fun reconcilePendingCancellations(
        activeByKey: Map<String, ai.onlinesdft.router.model.NotificationContext>,
    ) {
        val activeKeys = activeByKey.keys
        val confirmed = mutableListOf<Pair<String, DecisionSnapshot>>()
        val uncertain = mutableListOf<DecisionSnapshot>()
        val uncertainKeys = mutableListOf<String>()
        val missing = mutableListOf<Pair<String, PendingCancellation>>()
        val retry = mutableListOf<Pair<String, PendingCancellation>>()
        synchronized(actuationLock) {
            pendingCancellations.entries.toList().forEach { (key, pending) ->
                if (pending.confirmed) {
                    if (!pendingCancellations.remove(key, pending)) return@forEach
                    keyToDecision.remove(key, pending.decision)
                    pending.timeoutFuture?.cancel(false)
                    confirmed += key to pending.decision
                    return@forEach
                }
                val activeContext = activeByKey[key]
                if (activeContext == null) {
                    // A removal callback can race behind reconnect. Preserve
                    // correlation for one confirmation window instead of
                    // deleting a potentially confirmed LATER immediately.
                    missing += key to pending
                    return@forEach
                }
                if (
                    activeContext.postedAtMillis == pending.decision.context.postedAtMillis
                ) {
                    // A listener instance recreated after the request cannot
                    // inherit ScheduledFuture state. Retry the same sealed
                    // action and restore its confirmation timeout.
                    if (pending.timeoutFuture == null || pending.timeoutReported) {
                        retry += key to pending
                    }
                    return@forEach
                }
                // A package can reuse the same notification id/tag after the
                // original disappeared. Never apply an old sealed action to a
                // replacement merely because Android reused its key.
                if (!pendingCancellations.remove(key, pending)) return@forEach
                keyToDecision.remove(key, pending.decision)
                pending.timeoutFuture?.cancel(false)
                uncertain += pending.decision
                uncertainKeys += key
            }
            // Android may not deliver removal callbacks while the listener is
            // disconnected. Drop inactive, non-pending correlations so an app
            // can safely reuse the StatusBarNotification key after reconnect.
            keyToDecision.entries.toList().forEach { (key, decision) ->
                if (key !in activeKeys && !pendingCancellations.containsKey(key)) {
                    keyToDecision.remove(key, decision)
                }
            }
        }
        confirmed.forEach { (key, decision) ->
            if (fulfillCancellation(decision, reconciledAfterDisconnect = true)) {
                removePendingJournal(key, decision)
            }
        }
        uncertainKeys.zip(uncertain).forEach { (key, decision) ->
            removePendingJournal(key, decision)
        }
        uncertain.forEach { decision ->
            OnlineSdftApplication.runtime(this).failRouteExecution(
                decision,
                "cancellation result was unavailable after listener reconnect",
            )
            Log.w(
                PROOF_TAG,
                "ACTION_RESULT event_id=${decision.context.eventId} " +
                    "original_visible=unknown digest_posted=false " +
                    "cancellation=outcome_unavailable_after_reconnect",
            )
        }
        missing.forEach { (key, pending) -> scheduleMissingOutcomeResolution(key, pending) }
        retry.forEach { (key, pending) -> retryCancellation(key, pending) }
    }

    private fun abandonUnarmedCancellations(reason: String) {
        val confirmed = mutableListOf<Pair<String, DecisionSnapshot>>()
        val unresolved = mutableListOf<Pair<String, PendingCancellation>>()
        synchronized(actuationLock) {
            pendingCancellations.entries.toList().forEach { (key, pending) ->
                if (pending.timeoutFuture != null && !pending.timeoutReported) return@forEach
                if (pending.confirmed) {
                    if (!pendingCancellations.remove(key, pending)) return@forEach
                    keyToDecision.remove(key, pending.decision)
                    confirmed += key to pending.decision
                } else {
                    unresolved += key to pending
                }
            }
        }
        confirmed.forEach { (key, decision) ->
            if (fulfillCancellation(decision, reconciledAfterDisconnect = true)) {
                removePendingJournal(key, decision)
            }
        }
        unresolved.forEach { (key, pending) ->
            scheduleMissingOutcomeResolution(key, pending, reason)
        }
    }

    private fun retryCancellation(key: String, pending: PendingCancellation) {
        var failure: String? = null
        var failureError: RuntimeException? = null
        synchronized(actuationLock) {
            if (!listenerConnected.get() || pendingCancellations[key] !== pending) return
            if (pending.confirmed) return
            if (
                pending.decision.chosenRoute == Route.LATER &&
                !DigestNotificationPublisher.canPublish(this)
            ) {
                pendingCancellations.remove(key, pending)
                keyToDecision.remove(key, pending.decision)
                pending.timeoutFuture?.cancel(false)
                removePendingJournal(key, pending.decision)
                failure = "Later digest is unavailable after listener reconnect"
            } else {
                pending.timeoutFuture?.cancel(false)
                pending.timeoutReported = false
                try {
                    pending.timeoutFuture = scheduleCancellationTimeout(key, pending)
                    cancelNotification(key)
                } catch (error: RuntimeException) {
                    pendingCancellations.remove(key, pending)
                    pending.timeoutFuture?.cancel(false)
                    removePendingJournal(key, pending.decision)
                    failure = "Android cancellation retry failed"
                    failureError = error
                }
            }
        }
        failure?.let { reason ->
            OnlineSdftApplication.runtime(this).failRouteExecution(pending.decision, reason)
            failureError?.let { Log.w(PROOF_TAG, "Cancellation retry failed for $key", it) }
        }
    }

    private fun scheduleMissingOutcomeResolution(
        key: String,
        pending: PendingCancellation,
        failureReason: String = "cancellation result was unavailable after listener reconnect",
    ) {
        synchronized(actuationLock) {
            if (pendingCancellations[key] !== pending) return
            pending.timeoutFuture?.cancel(false)
            pending.timeoutFuture = listenerExecutor.schedule(
                {
                    var confirmedDecision: DecisionSnapshot? = null
                    var uncertainDecision: DecisionSnapshot? = null
                    synchronized(actuationLock) {
                        if (!pendingCancellations.remove(key, pending)) return@schedule
                        keyToDecision.remove(key, pending.decision)
                        if (pending.confirmed) {
                            confirmedDecision = pending.decision
                        } else {
                            uncertainDecision = pending.decision
                        }
                    }
                    confirmedDecision?.let { decision ->
                        if (fulfillCancellation(decision, reconciledAfterDisconnect = true)) {
                            removePendingJournal(key, decision)
                        }
                    }
                    uncertainDecision?.let { decision ->
                        removePendingJournal(key, decision)
                        OnlineSdftApplication.runtime(this).failRouteExecution(
                            decision,
                            failureReason,
                        )
                        Log.w(
                            PROOF_TAG,
                            "ACTION_RESULT event_id=${decision.context.eventId} " +
                                "original_visible=unknown digest_posted=false " +
                                "cancellation=outcome_unavailable_after_grace",
                        )
                    }
                },
                CANCELLATION_CONFIRMATION_TIMEOUT_MILLIS,
                TimeUnit.MILLISECONDS,
            )
        }
    }

    private fun fulfillCancellation(
        decision: DecisionSnapshot,
        reconciledAfterDisconnect: Boolean,
    ): Boolean {
        val runtime = OnlineSdftApplication.runtime(this)
        var promoted = false
        val currentEpochHandled = runtime.withCurrentDecision(decision) {
            synchronized(actuationLock) {
                if (lockdownPending.get()) {
                    Log.w(
                        PROOF_TAG,
                        "ACTION_RESULT event_id=${decision.context.eventId} " +
                            "digest_posted=false delivery_suppressed=lockdown",
                    )
                    return@synchronized true
                }
                val digestDelivery = if (decision.chosenRoute == Route.LATER) {
                    DigestNotificationPublisher.publish(this, decision)
                } else {
                    null
                }
                val archiveSaved = if (decision.chosenRoute == Route.ARCHIVE) {
                    runtime.recordArchive(decision) != null
                } else {
                    false
                }
                val saved = digestDelivery?.saved == true || archiveSaved
                if (decision.chosenRoute != Route.INTERRUPT && !saved) {
                    runtime.failRouteExecution(
                        decision,
                        "source removed, but the Saved item could not be stored",
                    )
                } else {
                    promoted = true
                    runtime.confirmRouteExecution(decision)
                }
                Log.i(
                    PROOF_TAG,
                    "ACTION_RESULT event_id=${decision.context.eventId} " +
                        "original_visible=false " +
                        "digest_posted=${digestDelivery?.alertPosted == true} " +
                        "inbox_saved=$saved " +
                        "reconciled_after_disconnect=$reconciledAfterDisconnect",
                )
                true
            }
        } == true
        if (!currentEpochHandled) {
            // Reset deliberately cleared UI/learner state, but it must not
            // revoke a Later digest after Android already removed the source.
            synchronized(actuationLock) {
                if (lockdownPending.get()) return@synchronized
                val digestDelivery = if (decision.chosenRoute == Route.LATER) {
                    DigestNotificationPublisher.publish(
                        this,
                        decision,
                        allowStaleDelivery = true,
                    )
                } else {
                    null
                }
                val archiveSaved = if (decision.chosenRoute == Route.ARCHIVE) {
                    runtime.recordArchive(decision, allowStaleDelivery = true) != null
                } else {
                    false
                }
                val saved = digestDelivery?.saved == true || archiveSaved
                promoted = saved
                Log.i(
                    PROOF_TAG,
                    "ACTION_RESULT event_id=${decision.context.eventId} " +
                        "original_visible=false " +
                        "digest_posted=${digestDelivery?.alertPosted == true} " +
                        "inbox_saved=$saved " +
                        "stale_epoch_delivery=true " +
                        "reconciled_after_disconnect=$reconciledAfterDisconnect",
                )
            }
        }
        return promoted
    }

    private fun isGenerationActive(generation: Long): Boolean =
        listenerConnected.get() &&
            !lockdownPending.get() &&
            generation == eventGeneration.get()

    private fun invalidatePendingState() {
        // Keep correlations for notifications that were already posted. Their
        // actions stay sealed across Reset; stale factual callbacks are ignored
        // by runEpoch, while an outstanding LATER digest can still be fulfilled.
        // The generation bump drops only queued, not-yet-committed callbacks.
        eventGeneration.incrementAndGet()
    }

    private fun submitListenerTask(block: () -> Unit): Boolean = try {
        listenerExecutor.execute(block)
        true
    } catch (_: RejectedExecutionException) {
        false
    }

    private fun clearPendingState(reportFailure: Boolean) {
        val pending = pendingCancellations.values.toSet()
        pendingCancellations.clear()
        pending.forEach { it.timeoutFuture?.cancel(false) }
        keyToDecision.clear()
        if (reportFailure && pending.isNotEmpty()) {
            val runtime = OnlineSdftApplication.runtime(this)
            pending.forEach { cancellation ->
                runtime.failRouteExecution(
                    cancellation.decision,
                    "notification listener disconnected",
                )
            }
        }
    }

    private fun removePendingJournal(key: String, decision: DecisionSnapshot): Boolean {
        val removed = PendingCancellationStore.remove(this, key)
        if (!removed) {
            Log.e(
                PROOF_TAG,
                "PENDING_JOURNAL_REMOVE_FAILED event_id=${decision.context.eventId}",
            )
        }
        return removed
    }

    private fun clearPendingJournalForLockdown(): Boolean {
        repeat(JOURNAL_COMMIT_ATTEMPTS) { attempt ->
            if (PendingCancellationStore.clear(this)) return true
            Log.e(PROOF_TAG, "LOCKDOWN_JOURNAL_CLEAR_RETRY attempt=${attempt + 1}")
        }
        return false
    }

    private fun Int.wireName(): String = when (this) {
        REASON_CLICK -> "click"
        REASON_CANCEL -> "dismiss"
        REASON_CANCEL_ALL -> "dismiss_all"
        REASON_LISTENER_CANCEL -> "listener_cancel"
        REASON_TIMEOUT -> "timeout"
        else -> "android_$this"
    }

    companion object {
        @Volatile
        private var activeInstance: RouterNotificationListenerService? = null
        private val lifecycleLock = Any()

        fun clearPendingDecisions() {
            val listener = synchronized(lifecycleLock) { activeInstance }
            listener?.invalidatePendingState()
        }

        private const val PROOF_TAG = "OnlineSdftProof"
        private const val CANCELLATION_CONFIRMATION_TIMEOUT_MILLIS = 2_500L
        private const val JOURNAL_COMMIT_ATTEMPTS = 3
    }
}
