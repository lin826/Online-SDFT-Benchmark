package ai.onlinesdft.router.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RouterTimelineUiModelTest {
    @Test
    fun `newest bottom event maps to its lazy column item`() {
        val target = routerAutoScrollTarget(listOf("first", "second", "third"))

        assertEquals("third", target?.newestEventId)
        assertEquals(ROUTER_ITEMS_BEFORE_EVENT_TIMELINE + 2, target?.itemIndex)
    }

    @Test
    fun `metric updates keep the same auto scroll identity`() {
        val beforeMetrics = routerAutoScrollTarget(listOf("first", "second"))
        val afterMetrics = routerAutoScrollTarget(listOf("first", "second"))

        assertEquals(beforeMetrics, afterMetrics)
    }

    @Test
    fun `appending an event moves auto scroll to the new bottom`() {
        val beforeAppend = routerAutoScrollTarget(listOf("first", "second"))
        val afterAppend = routerAutoScrollTarget(listOf("first", "second", "third"))

        assertEquals("second", beforeAppend?.newestEventId)
        assertEquals("third", afterAppend?.newestEventId)
        assertEquals(beforeAppend!!.itemIndex + 1, afterAppend!!.itemIndex)
    }

    @Test
    fun `event focus resolves the exact chronological card`() {
        val target = routerEventFocusTarget(
            eventIds = listOf("first", "second", "third", "last"),
            requestedEventId = "second",
        )

        assertEquals("second", target?.eventId)
        assertEquals(ROUTER_ITEMS_BEFORE_EVENT_TIMELINE + 1, target?.itemIndex)
        assertNull(
            routerEventFocusTarget(
                eventIds = listOf("first", "second"),
                requestedEventId = "unknown",
            ),
        )
    }

    @Test
    fun `plain router request returns the viewport to the top`() {
        val target = routerPageRequestTarget(
            eventIds = listOf("first", "second", "third"),
            highlightedEventId = null,
        )

        assertEquals(0, target?.itemIndex)
        assertNull(target?.highlightedEventId)
    }

    @Test
    fun `highlight router request is fail closed for a missing event`() {
        assertNull(
            routerPageRequestTarget(
                eventIds = listOf("first", "second"),
                highlightedEventId = "missing",
            ),
        )
    }

    @Test
    fun `saved alert permission status remains visible in either state`() {
        val allowed = savedAlertsPermissionUi(enabled = true)
        val required = savedAlertsPermissionUi(enabled = false)

        assertEquals("Allowed", allowed.value)
        assertTrue(allowed.healthy)
        assertEquals("Action required", required.value)
        assertFalse(required.healthy)
    }

    @Test
    fun `empty timeline has no auto scroll target`() {
        assertNull(routerAutoScrollTarget(emptyList()))
    }
}
