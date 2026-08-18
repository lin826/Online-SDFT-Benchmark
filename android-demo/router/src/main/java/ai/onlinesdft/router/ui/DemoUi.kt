package ai.onlinesdft.router.ui

import ai.onlinesdft.router.model.ExecutionConstraint
import ai.onlinesdft.router.model.Route
import ai.onlinesdft.router.notification.DigestInboxItem
import ai.onlinesdft.router.notification.DigestInboxOrigin
import ai.onlinesdft.router.notification.DigestInboxSummary
import ai.onlinesdft.router.state.DemoPage
import ai.onlinesdft.router.state.DemoUiState
import ai.onlinesdft.router.state.RoutedEventUi
import android.content.Context
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.calculateEndPadding
import androidx.compose.foundation.layout.calculateStartPadding
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kotlinx.coroutines.flow.StateFlow
import java.util.Locale

private val CardShape = RoundedCornerShape(18.dp)
private val InnerShape = RoundedCornerShape(12.dp)
private val PillShape = RoundedCornerShape(999.dp)

@Composable
fun OnlineSdftApp(
    stateFlow: StateFlow<DemoUiState>,
    listenerAccess: Boolean,
    digestNotificationsEnabled: Boolean,
    onOpenNotificationAccess: () -> Unit,
    onRequestDigestPermission: () -> Unit,
    onReset: () -> Unit,
    onShowRouter: () -> Unit,
    onShowDigest: () -> Unit,
    onShowScores: () -> Unit,
    onPageVisible: (DemoPage, Long, String?) -> Unit,
    onSelectRouterEvent: (String) -> Unit,
    onOpenDigest: (String) -> Unit,
    onRemoveDigest: (String) -> Unit,
    onArchivePreference: (String, Route) -> Unit,
    onCorrection: (String, Route) -> Unit,
) {
    val state by stateFlow.collectAsStateWithLifecycle()
    val routerListState = rememberLazyListState()
    val digestListState = rememberLazyListState()
    val scoresListState = rememberLazyListState()
    val listState = when (state.selectedPage) {
        DemoPage.ROUTER -> routerListState
        DemoPage.DIGEST -> digestListState
        DemoPage.SCORES -> scoresListState
    }
    val routerEventIds = state.events.map { it.decision.context.eventId }
    val autoScrollTarget = routerAutoScrollTarget(routerEventIds)
    val pageRequestTarget = routerPageRequestTarget(routerEventIds, state.highlightedEventId)
    LaunchedEffect(autoScrollTarget?.newestEventId) {
        if (
            autoScrollTarget != null &&
            state.selectedPage == DemoPage.ROUTER &&
            state.highlightedEventId == null
        ) {
            // Header, assistant summary, setup, and the section heading precede
            // this chronological list. Metric-only updates keep the key stable.
            routerListState.animateScrollToItem(index = autoScrollTarget.itemIndex)
        }
    }
    LaunchedEffect(state.selectedPage, state.pageRequestId) {
        if (state.selectedPage == DemoPage.ROUTER) {
            val requestedTarget = pageRequestTarget ?: return@LaunchedEffect
            routerListState.animateScrollToItem(index = requestedTarget.itemIndex)
        }
        withFrameNanos { }
        onPageVisible(
            state.selectedPage,
            state.pageRequestId,
            pageRequestTarget?.highlightedEventId,
        )
    }
    RouterTheme {
        val colors = routerColors
        Scaffold(
            containerColor = colors.background,
            bottomBar = {
                BottomNav(
                    selectedPage = state.selectedPage,
                    unreadSaved = state.digestItems.count { it.isUnread },
                    onShowRouter = onShowRouter,
                    onShowDigest = onShowDigest,
                    onShowScores = onShowScores,
                )
            },
        ) { scaffoldPadding ->
            val direction = LocalLayoutDirection.current
            LazyColumn(
                state = listState,
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(
                    start = scaffoldPadding.calculateStartPadding(direction) + 16.dp,
                    end = scaffoldPadding.calculateEndPadding(direction) + 16.dp,
                    top = scaffoldPadding.calculateTopPadding() + 12.dp,
                    bottom = scaffoldPadding.calculateBottomPadding() + 24.dp,
                ),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                item { AppHeader(state = state, onReset = onReset) }
                when (state.selectedPage) {
                    DemoPage.ROUTER -> {
                        item { AssistantCard(state) }
                        item {
                            SetupCard(
                                state = state,
                                listenerAccess = listenerAccess,
                                digestNotificationsEnabled = digestNotificationsEnabled,
                                onOpenNotificationAccess = onOpenNotificationAccess,
                                onRequestDigestPermission = onRequestDigestPermission,
                            )
                        }
                        item {
                            SectionHeading(
                                title = "Recent notifications",
                                subtitle = "What your assistant did, and why.",
                            )
                        }
                        if (state.events.isEmpty()) {
                            item { EmptyEventsCard() }
                        } else {
                            items(state.events, key = { it.decision.context.eventId }) { event ->
                                EventCard(
                                    event = event,
                                    highlighted = event.decision.context.eventId ==
                                        state.highlightedEventId,
                                    labRunning = state.labRunning,
                                    correctionPending = event.decision.context.eventId in
                                        state.pendingCorrectionEventIds,
                                    correctionBusy = state.pendingCorrectionEventIds.isNotEmpty(),
                                    correctionCompleted = event.decision.context.eventId in
                                        state.completedCorrectionEventIds,
                                    onSelect = onSelectRouterEvent,
                                    onCorrection = onCorrection,
                                )
                            }
                        }
                    }
                    DemoPage.DIGEST -> item {
                        SavedPage(
                            items = state.digestItems,
                            summary = state.digestSummary,
                            notificationsEnabled = digestNotificationsEnabled,
                            onRequestPermission = onRequestDigestPermission,
                            onOpen = onOpenDigest,
                            onRemove = onRemoveDigest,
                            onArchivePreference = onArchivePreference,
                        )
                    }
                    DemoPage.SCORES -> item { ProgressPage(state) }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- chrome ----

@Composable
private fun AppHeader(state: DemoUiState, onReset: () -> Unit) {
    val colors = routerColors
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Tact",
                    style = MaterialTheme.typography.headlineSmall,
                    color = colors.ink,
                )
                Text(
                    text = "Knows what's worth interrupting you for",
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.inkMuted,
                )
            }
            TextButton(
                onClick = onReset,
                modifier = Modifier
                    .testTag("reset_demo")
                    .semantics { contentDescription = "Reset demo" },
            ) {
                Text("Reset", color = colors.inkMuted)
            }
        }
        if (state.statusMessage.isNotBlank()) {
            Text(
                text = state.statusMessage,
                style = MaterialTheme.typography.labelLarge,
                color = colors.brand,
                modifier = Modifier.padding(top = 6.dp),
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}

@Composable
private fun BottomNav(
    selectedPage: DemoPage,
    unreadSaved: Int,
    onShowRouter: () -> Unit,
    onShowDigest: () -> Unit,
    onShowScores: () -> Unit,
) {
    val colors = routerColors
    NavigationBar(
        containerColor = colors.surface,
        tonalElevation = 0.dp,
        modifier = Modifier.border(
            width = 1.dp,
            color = colors.outline,
            shape = RoundedCornerShape(0.dp),
        ),
    ) {
        NavItem(
            label = "Inbox",
            selected = selectedPage == DemoPage.ROUTER,
            testTag = "tab_router",
            description = "Show notification router",
            icon = NavIcon.FEED,
            onClick = onShowRouter,
        )
        NavItem(
            label = if (unreadSaved > 0) "Saved ($unreadSaved)" else "Saved",
            selected = selectedPage == DemoPage.DIGEST,
            testTag = "tab_digest",
            description = if (unreadSaved > 0) {
                "Show Saved for later, $unreadSaved unread"
            } else {
                "Show Saved for later"
            },
            icon = NavIcon.BOOKMARK,
            onClick = onShowDigest,
        )
        NavItem(
            label = "Progress",
            selected = selectedPage == DemoPage.SCORES,
            testTag = "tab_scores",
            description = "Show scores",
            icon = NavIcon.TREND,
            onClick = onShowScores,
        )
    }
}

private enum class NavIcon { FEED, BOOKMARK, TREND }

@Composable
private fun androidx.compose.foundation.layout.RowScope.NavItem(
    label: String,
    selected: Boolean,
    testTag: String,
    description: String,
    icon: NavIcon,
    onClick: () -> Unit,
) {
    val colors = routerColors
    NavigationBarItem(
        selected = selected,
        onClick = onClick,
        icon = { NavGlyph(icon, selected) },
        label = { Text(label, fontSize = 11.sp) },
        alwaysShowLabel = true,
        colors = NavigationBarItemDefaults.colors(
            selectedIconColor = colors.brand,
            selectedTextColor = colors.brand,
            unselectedIconColor = colors.inkFaint,
            unselectedTextColor = colors.inkFaint,
            indicatorColor = colors.brandSoft,
        ),
        modifier = Modifier
            .testTag(testTag)
            .semantics {
                this.contentDescription = description
                this.selected = selected
            },
    )
}

/** Hand-drawn glyphs so all three tabs share one weight and one language. */
@Composable
private fun NavGlyph(icon: NavIcon, selected: Boolean) {
    val colors = routerColors
    val tint = if (selected) colors.brand else colors.inkFaint
    Canvas(modifier = Modifier.size(22.dp)) {
        val w = size.width
        val h = size.height
        val stroke = Stroke(width = h * 0.11f, cap = StrokeCap.Round)
        when (icon) {
            NavIcon.FEED -> {
                val widths = listOf(1f, 0.78f, 0.52f)
                widths.forEachIndexed { index, fraction ->
                    val y = h * (0.26f + index * 0.24f)
                    drawLine(
                        color = tint,
                        start = androidx.compose.ui.geometry.Offset(w * 0.08f, y),
                        end = androidx.compose.ui.geometry.Offset(w * (0.08f + 0.84f * fraction), y),
                        strokeWidth = stroke.width,
                        cap = StrokeCap.Round,
                    )
                }
            }
            NavIcon.BOOKMARK -> {
                val path = Path().apply {
                    moveTo(w * 0.24f, h * 0.14f)
                    lineTo(w * 0.76f, h * 0.14f)
                    lineTo(w * 0.76f, h * 0.86f)
                    lineTo(w * 0.50f, h * 0.64f)
                    lineTo(w * 0.24f, h * 0.86f)
                    close()
                }
                if (selected) drawPath(path, tint) else drawPath(path, tint, style = stroke)
            }
            NavIcon.TREND -> {
                val heights = listOf(0.34f, 0.58f, 0.84f)
                heights.forEachIndexed { index, fraction ->
                    val x = w * (0.20f + index * 0.30f)
                    drawLine(
                        color = tint,
                        start = androidx.compose.ui.geometry.Offset(x, h * 0.86f),
                        end = androidx.compose.ui.geometry.Offset(x, h * (0.86f - fraction * 0.7f)),
                        strokeWidth = stroke.width * 1.5f,
                        cap = StrokeCap.Round,
                    )
                }
            }
        }
    }
}

// ------------------------------------------------------------- inbox tab ----

@Composable
private fun AssistantCard(state: DemoUiState) {
    val colors = routerColors
    val progress = LearningProgress.from(state)
    ProductCard {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "Your assistant",
                style = MaterialTheme.typography.titleMedium,
                color = colors.ink,
            )
            StatusPill(progress.readiness)
        }
        if (progress.readiness == AssistantReadiness.PAUSED) {
            Spacer(Modifier.height(8.dp))
            Text(
                "Your notifications are passing through untouched while it is paused.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.inkMuted,
            )
        }
        Spacer(Modifier.height(16.dp))
        Text(
            progress.stage.label,
            style = MaterialTheme.typography.titleLarge,
            color = colors.ink,
        )
        Text(
            progress.stage.blurb,
            style = MaterialTheme.typography.bodySmall,
            color = colors.inkMuted,
            modifier = Modifier.padding(top = 2.dp),
        )
        Spacer(Modifier.height(12.dp))
        if (state.labRunning) {
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth().height(8.dp),
                color = colors.brand,
                trackColor = colors.track,
            )
        } else {
            Meter(fraction = progress.stageProgress, color = colors.brand)
        }
        Spacer(Modifier.height(16.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            StatTile(
                label = "Lessons from you",
                value = progress.lessons.toString(),
                tag = "score_updates",
                modifier = Modifier.weight(1f),
            )
            StatTile(
                label = "Matches your call",
                value = progress.accuracy?.let(::percent) ?: "Not yet",
                tag = "score_accuracy_adaptive",
                accent = colors.brand,
                modifier = Modifier.weight(1f),
            )
        }
        Spacer(Modifier.height(10.dp))
        Text(
            "Everything it learns stays on this phone.",
            style = MaterialTheme.typography.labelSmall,
            color = colors.inkFaint,
        )
    }
}

@Composable
private fun SetupCard(
    state: DemoUiState,
    listenerAccess: Boolean,
    digestNotificationsEnabled: Boolean,
    onOpenNotificationAccess: () -> Unit,
    onRequestDigestPermission: () -> Unit,
) {
    val colors = routerColors
    val allSet = listenerAccess && state.listenerConnected && digestNotificationsEnabled
    ProductCard {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Setup", style = MaterialTheme.typography.titleMedium, color = colors.ink)
            if (allSet) {
                Text(
                    "All set",
                    style = MaterialTheme.typography.labelMedium,
                    color = colors.brand,
                    fontWeight = FontWeight.SemiBold,
                )
            }
        }
        Spacer(Modifier.height(12.dp))
        SetupRow(
            label = "Notification access",
            value = if (listenerAccess) "On" else "Needs permission",
            healthy = listenerAccess,
        )
        SetupRow(
            label = "Listening",
            value = if (state.listenerConnected) "Connected" else "Connecting",
            healthy = state.listenerConnected,
        )
        SetupRow(
            label = "Reminders",
            value = if (digestNotificationsEnabled) "On" else "Needs permission",
            healthy = digestNotificationsEnabled,
        )
        if (!allSet) {
            Spacer(Modifier.height(14.dp))
            Text(
                "Access lets it read notifications from your other apps. Reminders let it " +
                    "nudge you about anything it saved for later.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.inkMuted,
            )
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (!listenerAccess || !state.listenerConnected) {
                    PrimaryButton(
                        text = "Allow access",
                        onClick = onOpenNotificationAccess,
                        modifier = Modifier.weight(1f),
                    )
                }
                if (!digestNotificationsEnabled) {
                    SecondaryButton(
                        text = "Turn on reminders",
                        onClick = onRequestDigestPermission,
                        modifier = Modifier.weight(1f),
                    )
                }
            }
        }
    }
}

@Composable
private fun SetupRow(label: String, value: String, healthy: Boolean) {
    val colors = routerColors
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 5.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier
                    .size(7.dp)
                    .background(if (healthy) colors.brand else colors.warning, PillShape),
            )
            Text(
                label,
                modifier = Modifier.padding(start = 10.dp),
                style = MaterialTheme.typography.bodyMedium,
                color = colors.inkMuted,
            )
        }
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
            color = if (healthy) colors.ink else colors.warning,
        )
    }
}


@Composable
private fun EventCard(
    event: RoutedEventUi,
    highlighted: Boolean,
    labRunning: Boolean,
    correctionPending: Boolean,
    correctionBusy: Boolean,
    correctionCompleted: Boolean,
    onSelect: (String) -> Unit,
    onCorrection: (String, Route) -> Unit,
) {
    val colors = routerColors
    val context = LocalContext.current
    val decision = event.decision
    var showDetails by remember(decision.context.eventId) { mutableStateOf(false) }
    ProductCard(
        container = if (highlighted) colors.brandSoft else colors.surface,
        borderColor = if (highlighted) colors.brand else colors.outline,
        modifier = Modifier
            .testTag("router_event_${decision.context.eventId}")
            .clickable { onSelect(decision.context.eventId) }
            .semantics {
                selected = highlighted
                contentDescription = "Select notification ${decision.context.eventId}"
            },
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "${appLabel(context, decision.context.packageName)} · " +
                    relativeTime(decision.context.postedAtMillis),
                style = MaterialTheme.typography.labelMedium,
                color = colors.inkMuted,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )
            RouteBadge(decision.chosenRoute)
        }
        Spacer(Modifier.height(8.dp))
        Text(
            decision.context.title,
            style = MaterialTheme.typography.titleMedium,
            color = colors.ink,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
        )
        if (decision.context.body.isNotBlank()) {
            Text(
                decision.context.body,
                style = MaterialTheme.typography.bodyMedium,
                color = colors.inkMuted,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.padding(top = 3.dp),
            )
        }

        if (decision.executionConstraint != ExecutionConstraint.NONE) {
            Spacer(Modifier.height(12.dp))
            Notice("This kind of notification is always left alone.")
        }

        Spacer(Modifier.height(16.dp))
        Text(
            if (event.postTrainingProbabilities != null) {
                "How it would call this now"
            } else {
                "How sure it was"
            },
            style = MaterialTheme.typography.labelMedium,
            color = colors.inkFaint,
        )
        Spacer(Modifier.height(6.dp))
        routeProbabilityShiftRows(event).forEach { row -> ConfidenceRow(row) }

        Spacer(Modifier.height(14.dp))
        Text(
            event.lessonStatus,
            style = MaterialTheme.typography.labelMedium,
            color = colors.brand,
            fontWeight = FontWeight.Medium,
        )

        Spacer(Modifier.height(14.dp))
        Text(
            when {
                labRunning -> "You can correct this once it finishes catching up"
                correctionPending -> "Learning from you…"
                correctionBusy -> "Finishing another correction…"
                correctionCompleted -> "Thanks — it learned from this"
                else -> "Wrong call? Tell it what you wanted."
            },
            style = MaterialTheme.typography.labelMedium,
            color = colors.inkMuted,
        )
        Spacer(Modifier.height(8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Route.entries.forEach { route ->
                ChoiceChip(
                    text = route.label(),
                    color = route.color(),
                    enabled = !labRunning && !correctionBusy && !correctionCompleted,
                    onClick = { onCorrection(decision.context.eventId, route) },
                    modifier = Modifier
                        .weight(1f)
                        .testTag(
                            "feedback_${route.name.lowercase()}_${decision.context.eventId}",
                        )
                        .semantics {
                            contentDescription =
                                "Teach ${route.name.lowercase()} for ${decision.context.eventId}"
                        },
                )
            }
        }

        Spacer(Modifier.height(12.dp))
        DisclosureRow(
            label = "Details",
            expanded = showDetails,
            onToggle = { showDetails = !showDetails },
        )
        if (showDetails) {
            Spacer(Modifier.height(8.dp))
            DetailText(
                listOfNotNull(
                    "Source: ${decision.context.packageName}",
                    "Reference: ${decision.context.eventId}",
                    "Category: ${decision.context.category} · ${
                        decision.context.regime.name.lowercase().replace('_', ' ')
                    } · importance ${format(decision.context.importance.toDouble(), 2)}",
                    decision.foundationInferenceLatencyMillis?.let {
                        "Decided on-device in ${format(it, 1)} ms"
                    },
                    if (decision.recommendedRoute != decision.chosenRoute) {
                        "Suggested ${decision.recommendedRoute.label()}, applied " +
                            "${decision.chosenRoute.label()} " +
                            "(${decision.executionConstraint.wireName.replace('_', ' ')})"
                    } else {
                        null
                    },
                    if (!decision.foundationAvailable) {
                        "Model unavailable for this one — passed through untouched"
                    } else {
                        null
                    },
                    event.training?.let {
                        "Learning error ${format(it.lossBefore)} → ${format(it.lossAfter)} " +
                            "on lesson ${it.updateIndex}"
                    },
                    if (event.syntheticLab) "From a demo run" else "From a live notification",
                ).joinToString("\n"),
            )
        }
    }
}

@Composable
private fun ConfidenceRow(row: RouteProbabilityShiftUi) {
    val colors = routerColors
    val route = row.route
    val target = row.currentProbability.coerceIn(0f, 1f)
    val animated by animateFloatAsState(targetValue = target, label = "confidence")
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            route.label(),
            modifier = Modifier.width(78.dp),
            style = MaterialTheme.typography.labelMedium,
            color = colors.inkMuted,
            maxLines = 1,
        )
        Box(
            modifier = Modifier
                .weight(1f)
                .height(8.dp)
                .background(colors.track, PillShape),
        ) {
            if (animated > 0f) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(animated)
                        .height(8.dp)
                        .background(route.color(), PillShape),
                )
            }
        }
        Text(
            confidenceText(row),
            modifier = Modifier.width(96.dp).padding(start = 10.dp),
            style = MaterialTheme.typography.labelMedium,
            color = colors.ink,
            textAlign = TextAlign.End,
            maxLines = 1,
            softWrap = false,
        )
    }
}

@Composable
private fun EmptyEventsCard() {
    val colors = routerColors
    ProductCard {
        Text(
            "Nothing here yet",
            style = MaterialTheme.typography.titleMedium,
            color = colors.ink,
        )
        Text(
            "As notifications arrive, you will see what your assistant did with each one, " +
                "and you can tell it when it got one wrong.",
            style = MaterialTheme.typography.bodyMedium,
            color = colors.inkMuted,
            modifier = Modifier.padding(top = 6.dp),
        )
    }
}

// ------------------------------------------------------------- saved tab ----

@Composable
private fun SavedPage(
    items: List<DigestInboxItem>,
    summary: DigestInboxSummary,
    notificationsEnabled: Boolean,
    onRequestPermission: () -> Unit,
    onOpen: (String) -> Unit,
    onRemove: (String) -> Unit,
    onArchivePreference: (String, Route) -> Unit,
) {
    val colors = routerColors
    val context = LocalContext.current
    val silencedCount = items.count { it.origin == DigestInboxOrigin.ROUTER_ARCHIVE }
    val practiceCount = items.count { it.origin == DigestInboxOrigin.SYNTHETIC_LAB }
    val allSourceCounts = items
        .groupingBy { appLabel(context, it.sourcePackage) }
        .eachCount()
        .map { (source, count) -> source to count }
        .sortedWith(compareByDescending<Pair<String, Int>> { it.second }.thenBy { it.first })
    val sourceCounts = allSourceCounts.take(3)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("digest_inbox_page")
            .semantics {
                contentDescription = if (summary.unread > 0) {
                    "Saved and archived, ${summary.unread} unread"
                } else {
                    "Saved and archived"
                }
            },
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        SectionHeading(
            title = "Saved",
            subtitle = "Things it held back so they would not interrupt you.",
        )

        if (!notificationsEnabled) {
            ProductCard(container = colors.warningSoft, borderColor = colors.warning) {
                Text(
                    "Turn on reminders so it can keep saving things for later. Anything already " +
                        "saved stays right here.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = colors.ink,
                )
                Spacer(Modifier.height(12.dp))
                SecondaryButton(
                    text = "Turn on reminders",
                    onClick = onRequestPermission,
                    modifier = Modifier
                        .testTag("allow_digest_notifications")
                        .semantics {
                            contentDescription = "Allow saved notification alerts"
                        },
                )
            }
        }

        if (items.isEmpty()) {
            ProductCard {
                Text(
                    "Nothing saved yet",
                    style = MaterialTheme.typography.titleMedium,
                    color = colors.ink,
                )
                Text(
                    "When your assistant decides something can wait, it will show up here " +
                        "instead of on your lock screen.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = colors.inkMuted,
                    modifier = Modifier.padding(top = 6.dp),
                )
            }
        } else {
            SavedSummaryCard(
                summary = summary,
                sourceCounts = sourceCounts,
                additionalSourceCount = (allSourceCounts.size - sourceCounts.size)
                    .coerceAtLeast(0),
                silencedCount = silencedCount,
                practiceCount = practiceCount,
            )
            items.forEach { item ->
                SavedItemCard(
                    item = item,
                    sourceLabel = appLabel(context, item.sourcePackage),
                    onOpen = { onOpen(item.eventId) },
                    onRemove = { onRemove(item.eventId) },
                    onArchivePreference = { route -> onArchivePreference(item.eventId, route) },
                )
            }
        }
    }
}

@Composable
private fun SavedSummaryCard(
    summary: DigestInboxSummary,
    sourceCounts: List<Pair<String, Int>>,
    additionalSourceCount: Int,
    silencedCount: Int,
    practiceCount: Int,
) {
    val colors = routerColors
    val sourceText = sourceCounts.joinToString(", ") { (source, count) ->
        if (count == 1) source else "$source ($count)"
    } + if (additionalSourceCount > 0) ", +$additionalSourceCount more" else ""
    val previewText = summary.previewTitles.filter(String::isNotBlank).joinToString(" · ")
    ProductCard(
        container = colors.brandSoft,
        modifier = Modifier
            .testTag("digest_summary")
            .semantics(mergeDescendants = true) {
                contentDescription = buildString {
                    append("At a glance. ${summary.unread} unread, ${summary.total} saved")
                    if (sourceText.isNotBlank()) append(" from $sourceText")
                    if (previewText.isNotBlank()) append(". Recent: $previewText")
                }
            },
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            StatTile(
                label = "Unread",
                value = summary.unread.toString(),
                accent = colors.brand,
                modifier = Modifier.weight(1f),
            )
            StatTile(
                label = "Saved",
                value = summary.total.toString(),
                modifier = Modifier.weight(1f),
            )
            StatTile(
                label = "Silenced",
                value = silencedCount.toString(),
                modifier = Modifier.weight(1f),
            )
        }
        if (sourceText.isNotBlank()) {
            Text(
                "From $sourceText",
                style = MaterialTheme.typography.bodySmall,
                color = colors.inkMuted,
                modifier = Modifier.padding(top = 12.dp),
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
        }
        if (practiceCount > 0) {
            Text(
                "$practiceCount from a demo run — no real notification was touched.",
                style = MaterialTheme.typography.labelSmall,
                color = colors.inkFaint,
                modifier = Modifier.padding(top = 6.dp),
            )
        }
    }
}

@Composable
private fun SavedItemCard(
    item: DigestInboxItem,
    sourceLabel: String,
    onOpen: () -> Unit,
    onRemove: () -> Unit,
    onArchivePreference: (Route) -> Unit,
) {
    val colors = routerColors
    val practicePreview = item.origin == DigestInboxOrigin.SYNTHETIC_LAB
    val silenced = item.origin == DigestInboxOrigin.ROUTER_ARCHIVE
    var expanded by remember(item.eventId) { mutableStateOf(false) }
    var titleOverflow by remember(item.eventId) { mutableStateOf(false) }
    var bodyOverflow by remember(item.eventId) { mutableStateOf(false) }
    ProductCard(
        modifier = Modifier
            .testTag("digest_item_${item.eventId}")
            .semantics {
                contentDescription = if (practicePreview) {
                    "Saved synthetic lab preview ${item.eventId}"
                } else if (silenced) {
                    "Notification archived by router ${item.eventId}"
                } else {
                    "Saved live notification ${item.eventId}"
                }
            },
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "$sourceLabel · ${savedTime(item.routedAtMillis)}",
                style = MaterialTheme.typography.labelMedium,
                color = colors.inkMuted,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )
            Tag(
                text = when {
                    practicePreview -> "Demo"
                    silenced -> "Silenced"
                    item.isUnread -> "Unread"
                    else -> "Read"
                },
                color = when {
                    practicePreview -> colors.warning
                    silenced -> colors.silent
                    item.isUnread -> colors.later
                    else -> colors.inkFaint
                },
            )
        }
        Spacer(Modifier.height(8.dp))
        Text(
            item.title,
            style = MaterialTheme.typography.titleMedium,
            color = colors.ink,
            maxLines = if (expanded) Int.MAX_VALUE else 2,
            overflow = TextOverflow.Ellipsis,
            onTextLayout = { result -> if (!expanded) titleOverflow = result.hasVisualOverflow },
        )
        if (item.body.isNotBlank()) {
            Text(
                item.body,
                style = MaterialTheme.typography.bodyMedium,
                color = colors.inkMuted,
                maxLines = if (expanded) Int.MAX_VALUE else 3,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.padding(top = 3.dp),
                onTextLayout = { result -> if (!expanded) bodyOverflow = result.hasVisualOverflow },
            )
        }
        if (expanded || titleOverflow || bodyOverflow) {
            Spacer(Modifier.height(4.dp))
            DisclosureRow(
                label = if (expanded) "Show less" else "Read more",
                expanded = expanded,
                onToggle = { expanded = !expanded },
                description = if (expanded) {
                    "Show less of ${item.title}"
                } else {
                    "View full saved notification ${item.title}"
                },
            )
        }
        Spacer(Modifier.height(14.dp))
        if (silenced && item.isUnread) {
            Text(
                "Was this right?",
                style = MaterialTheme.typography.labelMedium,
                color = colors.inkMuted,
            )
            Spacer(Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                PrimaryButton(
                    text = "Open it",
                    onClick = { onArchivePreference(Route.LATER) },
                    modifier = Modifier
                        .weight(1f)
                        .semantics {
                            contentDescription = "Open ${item.title} and teach Later"
                        },
                )
                SecondaryButton(
                    text = "Show next time",
                    onClick = { onArchivePreference(Route.INTERRUPT) },
                    modifier = Modifier
                        .weight(1f)
                        .semantics { contentDescription = "Show ${item.title} next time" },
                )
            }
            Spacer(Modifier.height(8.dp))
            SecondaryButton(
                text = "Keep these quiet",
                onClick = { onArchivePreference(Route.ARCHIVE) },
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics {
                        contentDescription = "Confirm Keep silent for ${item.title}"
                    },
            )
        } else {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                if (item.isUnread) {
                    PrimaryButton(
                        text = "Mark read",
                        onClick = onOpen,
                        modifier = Modifier
                            .weight(1f)
                            .semantics { contentDescription = "Mark ${item.title} as read" },
                    )
                }
                SecondaryButton(
                    text = "Remove",
                    onClick = onRemove,
                    modifier = Modifier
                        .weight(1f)
                        .semantics {
                            contentDescription = "Remove ${item.title} from Saved"
                        },
                )
            }
        }
    }
}

// ---------------------------------------------------------- progress tab ----

@Composable
private fun ProgressPage(state: DemoUiState) {
    val colors = routerColors
    val progress = LearningProgress.from(state)
    val scores = LocalScoresSnapshot.from(state)
    var showTechnical by remember { mutableStateOf(false) }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("local_scores_page")
            .semantics { contentDescription = "Local scores from on-device state" },
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        SectionHeading(
            title = "Progress",
            subtitle = "How well your assistant is calling it, measured on this phone.",
        )

        ProductCard {
            if (!progress.hasScores) {
                Text(
                    "No results yet",
                    style = MaterialTheme.typography.titleMedium,
                    color = colors.ink,
                )
                Text(
                    "Run a practice round, or correct a few notifications, and its record " +
                        "will show up here.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = colors.inkMuted,
                    modifier = Modifier.padding(top = 6.dp),
                )
            } else {
                // Headline the recent window: with preferences that change, a
                // lifetime average blends two different sets of wishes and
                // understates where the assistant actually is now. The lifetime
                // figure stays visible directly underneath.
                val headline = progress.recentAccuracy ?: progress.accuracy ?: 0.0
                Text(
                    if (progress.recentAccuracy != null) {
                        "Matches what you wanted, lately"
                    } else {
                        "Matches what you wanted"
                    },
                    style = MaterialTheme.typography.labelMedium,
                    color = colors.inkFaint,
                )
                Spacer(Modifier.height(6.dp))
                Row(verticalAlignment = Alignment.Bottom) {
                    Text(
                        percent(headline),
                        style = MaterialTheme.typography.displaySmall,
                        color = colors.ink,
                    )
                    Spacer(Modifier.width(10.dp))
                    Text(
                        if (progress.recentAccuracy != null) {
                            "${percent(progress.accuracy ?: 0.0)} across all " +
                                "${progress.decisions} decisions"
                        } else {
                            "of ${progress.decisions} decisions"
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = colors.inkMuted,
                        modifier = Modifier.padding(bottom = 6.dp),
                    )
                }
                progress.improvementPoints?.let { improvement ->
                    Spacer(Modifier.height(10.dp))
                    Tag(
                        text = "${signedPoints(improvement)} since it started",
                        color = if (improvement >= 0) colors.brand else colors.warning,
                        filled = true,
                    )
                }
            }
        }

        if (progress.trend.size >= 2) {
            ProductCard {
                Text(
                    "Over time",
                    style = MaterialTheme.typography.titleMedium,
                    color = colors.ink,
                )
                Text(
                    "How often it matched you, decision by decision.",
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.inkMuted,
                    modifier = Modifier.padding(top = 2.dp, bottom = 16.dp),
                )
                TrendChart(progress.trend)
            }
        }

        ProductCard {
            Text(
                "What it has picked up",
                style = MaterialTheme.typography.titleMedium,
                color = colors.ink,
            )
            Spacer(Modifier.height(14.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                StatTile(
                    label = "Lessons",
                    value = progress.lessons.toString(),
                    accent = colors.brand,
                    modifier = Modifier.weight(1f),
                )
                StatTile(
                    label = "Examples kept",
                    value = progress.examplesRemembered.toString(),
                    tag = "score_replay",
                    modifier = Modifier.weight(1f),
                )
            }
            Spacer(Modifier.height(14.dp))
            Text(
                "It keeps a small set of recent examples so a new lesson does not overwrite " +
                    "what it already knows about you.",
                style = MaterialTheme.typography.bodySmall,
                color = colors.inkMuted,
            )
        }

        val lessonBefore = progress.lastLessonErrorBefore
        val lessonAfter = progress.lastLessonErrorAfter
        val lessonDescription = if (lessonBefore != null && lessonAfter != null) {
            "Latest loss: ${format(lessonBefore)} to ${format(lessonAfter)}"
        } else {
            "Latest loss: no update in this app session"
        }
        ProductCard(
            modifier = Modifier
                .testTag("score_latest_loss")
                .semantics(mergeDescendants = true) {
                    contentDescription = lessonDescription
                },
        ) {
            Text(
                "Last lesson",
                style = MaterialTheme.typography.titleMedium,
                color = colors.ink,
            )
            Spacer(Modifier.height(8.dp))
            val before = lessonBefore
            val after = lessonAfter
            if (before == null || after == null) {
                Text(
                    "It has not learned anything new since you opened the app.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = colors.inkMuted,
                )
            } else {
                Text(
                    "How far off it was on your last correction, before and after it learned.",
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.inkMuted,
                )
                Spacer(Modifier.height(12.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        format(before, 2),
                        style = MaterialTheme.typography.titleLarge,
                        color = colors.inkMuted,
                    )
                    Text(
                        "  →  ",
                        style = MaterialTheme.typography.titleLarge,
                        color = colors.inkFaint,
                    )
                    Text(
                        format(after, 2),
                        style = MaterialTheme.typography.titleLarge,
                        color = colors.brand,
                    )
                    progress.lastLessonErrorDrop?.let { drop ->
                        Spacer(Modifier.width(12.dp))
                        Tag(
                            text = signedPercent(-drop),
                            color = if (drop >= 0) colors.brand else colors.warning,
                            filled = true,
                        )
                    }
                }
            }
        }

        ProductCard {
            DisclosureRow(
                label = "Technical details",
                expanded = showTechnical,
                onToggle = { showTechnical = !showTechnical },
            )
            if (showTechnical) {
                Spacer(Modifier.height(12.dp))
                DetailText(
                    buildString {
                        appendLine(
                            "Adaptive ${scores.adaptiveCorrect}/${scores.decisions} " +
                                "(${percent(scores.adaptiveAccuracy)}) · " +
                                "frozen ${scores.frozenCorrect}/${scores.decisions} " +
                                "(${percent(scores.frozenAccuracy)})",
                        )
                        appendLine(
                            "Cumulative regret ${format(scores.adaptiveCumulativeRegret, 3)} " +
                                "vs ${format(scores.frozenCumulativeRegret, 3)} · " +
                                "gap ${signed(scores.regretGap, 3)}",
                        )
                        scores.optimizerSteps.forEach { step ->
                            if (step.completed) {
                                appendLine(
                                    "Optimizer step ${step.stepNumber}: " +
                                        "batch ${step.sampleCount} · " +
                                        "loss ${format(step.lossBefore!!)} → " +
                                        "${format(step.lossAfter!!)} · " +
                                        "grad ${format(step.gradientNorm!!, 5)} · " +
                                        "applied ${format(step.appliedUpdateNorm!!, 5)}",
                                )
                            } else {
                                appendLine(
                                    "Optimizer step ${step.stepNumber}: " +
                                        "no update in this app session",
                                )
                            }
                        }
                        appendLine(
                            "Adapter L2 norm ${format(scores.loraNorm, 6)} · " +
                                "replay ${scores.replaySize} · updates ${scores.updates}",
                        )
                        append("Adapter state is checkpointed on-device and restored on restart")
                    },
                    tag = "score_learner_state",
                    description = "Learner state saved on-device",
                )
            }
        }
    }
}

@Composable
private fun TrendChart(points: List<TrendPoint>) {
    val colors = routerColors
    Canvas(modifier = Modifier.fillMaxWidth().height(120.dp)) {
        val w = size.width
        val h = size.height
        listOf(0f, 0.5f, 1f).forEach { fraction ->
            val y = h * fraction
            drawLine(
                color = colors.track,
                start = androidx.compose.ui.geometry.Offset(0f, y),
                end = androidx.compose.ui.geometry.Offset(w, y),
                strokeWidth = 1f,
            )
        }
        if (points.size < 2) return@Canvas
        val step = w / (points.size - 1)
        val path = Path().apply {
            points.forEachIndexed { index, point ->
                val x = step * index
                val y = h - (point.adaptiveAccuracy.coerceIn(0.0, 1.0).toFloat() * h)
                if (index == 0) moveTo(x, y) else lineTo(x, y)
            }
        }
        drawPath(path, colors.brand, style = Stroke(width = 5f, cap = StrokeCap.Round))
        val last = points.last()
        drawCircle(
            color = colors.brand,
            radius = 9f,
            center = androidx.compose.ui.geometry.Offset(
                w,
                h - (last.adaptiveAccuracy.coerceIn(0.0, 1.0).toFloat() * h),
            ),
        )
    }
}


// ------------------------------------------------------------ primitives ----

@Composable
private fun ProductCard(
    container: Color = routerColors.surface,
    borderColor: Color = routerColors.outline,
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(container, CardShape)
            .border(1.dp, borderColor, CardShape)
            .padding(18.dp),
        content = content,
    )
}

@Composable
private fun SectionHeading(title: String, subtitle: String) {
    val colors = routerColors
    Column(modifier = Modifier.padding(top = 4.dp)) {
        Text(title, style = MaterialTheme.typography.titleLarge, color = colors.ink)
        Text(
            subtitle,
            style = MaterialTheme.typography.bodySmall,
            color = colors.inkMuted,
            modifier = Modifier.padding(top = 2.dp),
        )
    }
}

@Composable
private fun StatTile(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    tag: String? = null,
    accent: Color? = null,
) {
    val colors = routerColors
    Column(
        modifier = modifier
            .background(colors.surfaceMuted, InnerShape)
            .then(tag?.let { Modifier.testTag(it) } ?: Modifier)
            .semantics(mergeDescendants = true) { contentDescription = "$label: $value" }
            .padding(horizontal = 12.dp, vertical = 11.dp),
    ) {
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = colors.inkMuted,
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
        )
        Spacer(Modifier.height(3.dp))
        Text(
            value,
            style = MaterialTheme.typography.titleLarge,
            color = accent ?: colors.ink,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
private fun Meter(fraction: Float, color: Color) {
    val colors = routerColors
    val animated by animateFloatAsState(
        targetValue = fraction.coerceIn(0f, 1f),
        label = "meter",
    )
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(8.dp)
            .background(colors.track, PillShape),
    ) {
        if (animated > 0f) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(animated)
                    .height(8.dp)
                    .background(color, PillShape),
            )
        }
    }
}

@Composable
private fun StatusPill(readiness: AssistantReadiness) {
    val colors = routerColors
    val color = when (readiness) {
        AssistantReadiness.READY -> colors.brand
        AssistantReadiness.PAUSED -> colors.warning
        else -> colors.inkMuted
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .background(colors.surfaceMuted, PillShape)
            .padding(horizontal = 10.dp, vertical = 5.dp),
    ) {
        Box(Modifier.size(6.dp).background(color, PillShape))
        Text(
            readiness.label,
            modifier = Modifier.padding(start = 6.dp),
            style = MaterialTheme.typography.labelMedium,
            color = color,
            fontWeight = FontWeight.Medium,
        )
    }
}

@Composable
private fun Tag(text: String, color: Color, filled: Boolean = false) {
    Text(
        text,
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.SemiBold,
        color = color,
        modifier = if (filled) {
            Modifier
                .background(color.copy(alpha = 0.12f), PillShape)
                .padding(horizontal = 10.dp, vertical = 5.dp)
        } else {
            Modifier
        },
    )
}

@Composable
private fun RouteBadge(route: Route) {
    val color = route.color()
    Text(
        route.label(),
        color = color,
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.SemiBold,
        modifier = Modifier
            .background(color.copy(alpha = 0.13f), PillShape)
            .padding(horizontal = 10.dp, vertical = 5.dp),
    )
}

@Composable
private fun Notice(text: String) {
    val colors = routerColors
    Text(
        text,
        style = MaterialTheme.typography.bodySmall,
        color = colors.inkMuted,
        modifier = Modifier
            .fillMaxWidth()
            .background(colors.surfaceMuted, InnerShape)
            .padding(horizontal = 12.dp, vertical = 10.dp),
    )
}

@Composable
private fun PrimaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    val colors = routerColors
    Button(
        onClick = onClick,
        enabled = enabled,
        shape = PillShape,
        colors = ButtonDefaults.buttonColors(
            containerColor = colors.brand,
            contentColor = colors.onBrand,
            disabledContainerColor = colors.surfaceMuted,
            disabledContentColor = colors.inkFaint,
        ),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
        modifier = modifier,
    ) {
        Text(text, style = MaterialTheme.typography.labelLarge, maxLines = 1)
    }
}

@Composable
private fun SecondaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    val colors = routerColors
    Button(
        onClick = onClick,
        enabled = enabled,
        shape = PillShape,
        colors = ButtonDefaults.buttonColors(
            containerColor = colors.surfaceMuted,
            contentColor = colors.ink,
            disabledContainerColor = colors.surfaceMuted,
            disabledContentColor = colors.inkFaint,
        ),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
        modifier = modifier,
    ) {
        Text(text, style = MaterialTheme.typography.labelLarge, maxLines = 1)
    }
}

@Composable
private fun ChoiceChip(
    text: String,
    color: Color,
    enabled: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val colors = routerColors
    Button(
        onClick = onClick,
        enabled = enabled,
        shape = PillShape,
        colors = ButtonDefaults.buttonColors(
            containerColor = color.copy(alpha = 0.12f),
            contentColor = color,
            disabledContainerColor = colors.surfaceMuted,
            disabledContentColor = colors.inkFaint,
        ),
        contentPadding = PaddingValues(horizontal = 6.dp, vertical = 11.dp),
        modifier = modifier,
    ) {
        Text(text, style = MaterialTheme.typography.labelMedium, maxLines = 1)
    }
}

@Composable
private fun DisclosureRow(
    label: String,
    expanded: Boolean,
    onToggle: () -> Unit,
    description: String? = null,
) {
    val colors = routerColors
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .clickable(onClick = onToggle)
            .then(
                description?.let { text ->
                    Modifier.semantics { contentDescription = text }
                } ?: Modifier,
            )
            .padding(vertical = 4.dp),
    ) {
        Text(
            label,
            style = MaterialTheme.typography.labelLarge,
            color = colors.inkMuted,
        )
        Spacer(Modifier.width(6.dp))
        Canvas(modifier = Modifier.size(10.dp)) {
            val path = Path().apply {
                if (expanded) {
                    moveTo(0f, size.height * 0.72f)
                    lineTo(size.width / 2f, size.height * 0.28f)
                    lineTo(size.width, size.height * 0.72f)
                } else {
                    moveTo(0f, size.height * 0.28f)
                    lineTo(size.width / 2f, size.height * 0.72f)
                    lineTo(size.width, size.height * 0.28f)
                }
            }
            drawPath(path, colors.inkMuted, style = Stroke(width = 2f, cap = StrokeCap.Round))
        }
    }
}

@Composable
private fun DetailText(text: String, tag: String? = null, description: String? = null) {
    val colors = routerColors
    Text(
        text,
        style = MaterialTheme.typography.labelSmall,
        color = colors.inkFaint,
        modifier = Modifier
            .fillMaxWidth()
            .then(tag?.let { Modifier.testTag(it) } ?: Modifier)
            .then(
                description?.let { value ->
                    Modifier.semantics { contentDescription = value }
                } ?: Modifier,
            )
            .background(colors.surfaceSunken, InnerShape)
            .padding(12.dp),
    )
}

// --------------------------------------------------------------- helpers ----

private fun appLabel(context: Context, sourcePackage: String): String {
    val installedLabel = runCatching {
        val applicationInfo = context.packageManager.getApplicationInfo(sourcePackage, 0)
        context.packageManager.getApplicationLabel(applicationInfo).toString().trim()
    }.getOrNull()
    if (!installedLabel.isNullOrBlank()) return installedLabel
    val raw = sourcePackage.substringAfterLast('.').ifBlank { "App" }
    return raw
        .split(Regex("[-_\\s]+"))
        .filter(String::isNotBlank)
        .joinToString(" ") { word ->
            word.replaceFirstChar { character ->
                if (character.isLowerCase()) character.titlecase() else character.toString()
            }
        }
        .ifBlank { "App" }
}

private fun relativeTime(millis: Long): String {
    val minutes = ((System.currentTimeMillis() - millis).coerceAtLeast(0L) / 60_000L)
    return when {
        minutes < 1L -> "just now"
        minutes < 60L -> "${minutes}m ago"
        minutes < 1_440L -> "${minutes / 60L}h ago"
        else -> "${minutes / 1_440L}d ago"
    }
}

private fun savedTime(routedAtMillis: Long): String = "saved ${relativeTime(routedAtMillis)}"

private fun Route.label(): String = when (this) {
    Route.INTERRUPT -> "Show now"
    Route.LATER -> "Later"
    Route.ARCHIVE -> "Silence"
}

@Composable
private fun Route.color(): Color = when (this) {
    Route.INTERRUPT -> routerColors.now
    Route.LATER -> routerColors.later
    Route.ARCHIVE -> routerColors.silent
}

/**
 * Whole-percent confidence for the bars. The exact figures stay available in
 * [RouteProbabilityShiftUi.displayText] for the audit trail; two decimal places
 * on a screen a person reads is noise.
 */
private fun confidenceText(row: RouteProbabilityShiftUi): String {
    val current = String.format(
        Locale.US,
        "%.0f%%",
        row.currentProbability.coerceIn(0f, 1f) * 100f,
    )
    val shift = row.deltaPercentagePoints ?: return current
    val rounded = Math.round(shift).toInt()
    if (rounded == 0) return current
    return "$current ${String.format(Locale.US, "%+d", rounded)}"
}

private fun format(value: Double, digits: Int = 3): String =
    String.format(Locale.US, "%.${digits}f", value)

private fun percent(value: Double): String = String.format(Locale.US, "%.0f%%", value * 100)

private fun signedPoints(value: Double): String = String.format(Locale.US, "%+.0f pts", value)

private fun signedPercent(value: Double): String = String.format(Locale.US, "%+.0f%%", value * 100)

private fun signed(value: Double, digits: Int): String =
    String.format(Locale.US, "%+.${digits}f", value)
