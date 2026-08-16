package com.echo.android.sos

import android.util.Log
import com.echo.android.ai.SosState
import java.util.concurrent.CopyOnWriteArrayList

/**
 * State machine managing the SOS lifecycle.
 *
 * Enforces valid state transitions and rejects illegal transitions.
 * Guaranteed invariant: ACTIVE_SOS cannot transition silently to MONITORING without RESOLVED.
 */
class SosStateMachine(initialState: SosState = SosState.MONITORING) {

    companion object {
        private const val TAG = "SosStateMachine"
    }

    @Volatile
    var currentState: SosState = initialState
        private set

    private val listeners = CopyOnWriteArrayList<(SosState, SosState) -> Unit>()

    /**
     * Attempts to transition to the target state.
     * Returns true if transition was valid and executed; false if rejected.
     */
    @Synchronized
    fun transitionTo(targetState: SosState, reason: String = ""): Boolean {
        val previous = currentState
        if (previous == targetState) return true

        val isValid = when (previous) {
            SosState.MONITORING -> {
                targetState in listOf(SosState.COUNTDOWN, SosState.ACTIVE_SOS, SosState.DEGRADED)
            }
            SosState.COUNTDOWN -> {
                targetState in listOf(SosState.ACTIVE_SOS, SosState.CANCELLED, SosState.DEGRADED)
            }
            SosState.CANCELLED -> {
                targetState == SosState.MONITORING
            }
            SosState.ACTIVE_SOS -> {
                // ACTIVE_SOS must only transition to RESOLVED
                targetState == SosState.RESOLVED
            }
            SosState.RESOLVED -> {
                targetState == SosState.MONITORING
            }
            SosState.DEGRADED -> {
                targetState in listOf(SosState.MONITORING, SosState.COUNTDOWN, SosState.ACTIVE_SOS)
            }
        }

        if (!isValid) {
            Log.e(TAG, "Illegal state transition rejected: $previous -> $targetState (Reason: $reason)")
            return false
        }

        currentState = targetState
        Log.i(TAG, "State Transition: $previous -> $targetState | Reason: '$reason'")
        notifyListeners(previous, targetState)
        return true
    }

    fun addStateListener(listener: (previousState: SosState, newState: SosState) -> Unit) {
        listeners.add(listener)
    }

    fun removeStateListener(listener: (previousState: SosState, newState: SosState) -> Unit) {
        listeners.remove(listener)
    }

    private fun notifyListeners(previous: SosState, current: SosState) {
        for (l in listeners) {
            try {
                l.invoke(previous, current)
            } catch (e: Exception) {
                Log.e(TAG, "Error in state listener", e)
            }
        }
    }
}
