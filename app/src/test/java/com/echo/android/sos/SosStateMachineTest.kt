package com.echo.android.sos

import com.echo.android.ai.SosState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class SosStateMachineTest {

    private lateinit var stateMachine: SosStateMachine

    @Before
    fun setup() {
        stateMachine = SosStateMachine(SosState.MONITORING)
    }

    @Test
    fun testValidCompleteSosLifecycle() {
        assertEquals(SosState.MONITORING, stateMachine.currentState)

        // MONITORING -> COUNTDOWN
        assertTrue(stateMachine.transitionTo(SosState.COUNTDOWN, "Threat threshold crossed"))
        assertEquals(SosState.COUNTDOWN, stateMachine.currentState)

        // COUNTDOWN -> ACTIVE_SOS
        assertTrue(stateMachine.transitionTo(SosState.ACTIVE_SOS, "Countdown finished"))
        assertEquals(SosState.ACTIVE_SOS, stateMachine.currentState)

        // ACTIVE_SOS -> RESOLVED
        assertTrue(stateMachine.transitionTo(SosState.RESOLVED, "User resolved"))
        assertEquals(SosState.RESOLVED, stateMachine.currentState)

        // RESOLVED -> MONITORING
        assertTrue(stateMachine.transitionTo(SosState.MONITORING, "Reset"))
        assertEquals(SosState.MONITORING, stateMachine.currentState)
    }

    @Test
    fun testCancellationLifecycle() {
        assertTrue(stateMachine.transitionTo(SosState.COUNTDOWN, "Threat threshold crossed"))
        assertEquals(SosState.COUNTDOWN, stateMachine.currentState)

        // COUNTDOWN -> CANCELLED
        assertTrue(stateMachine.transitionTo(SosState.CANCELLED, "User cancelled"))
        assertEquals(SosState.CANCELLED, stateMachine.currentState)

        // CANCELLED -> MONITORING
        assertTrue(stateMachine.transitionTo(SosState.MONITORING, "Reset"))
        assertEquals(SosState.MONITORING, stateMachine.currentState)
    }

    @Test
    fun testActiveSosCannotSilentlyReturnToMonitoring() {
        stateMachine.transitionTo(SosState.COUNTDOWN)
        stateMachine.transitionTo(SosState.ACTIVE_SOS)
        assertEquals(SosState.ACTIVE_SOS, stateMachine.currentState)

        // Illegal direct transition to MONITORING or CANCELLED must be rejected
        assertFalse(stateMachine.transitionTo(SosState.MONITORING))
        assertFalse(stateMachine.transitionTo(SosState.CANCELLED))
        assertEquals(SosState.ACTIVE_SOS, stateMachine.currentState)
    }

    @Test
    fun testManualSosDirectTransition() {
        assertEquals(SosState.MONITORING, stateMachine.currentState)
        assertTrue(stateMachine.transitionTo(SosState.ACTIVE_SOS, "Manual trigger"))
        assertEquals(SosState.ACTIVE_SOS, stateMachine.currentState)
    }
}
