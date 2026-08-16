package com.echo.android.sos

import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.util.Log
import com.echo.android.ai.SosState

/**
 * Controls the LOCAL EMERGENCY SIREN.
 *
 * CRITICAL RULE:
 * The siren may activate ONLY when:
 *   state == ACTIVE_SOS AND remote communication is unavailable.
 *
 * MUST NOT activate during:
 *   MONITORING, COUNTDOWN, CANCELLED, RESOLVED, DEGRADED, or network failure before SOS confirmation.
 */
class LocalSirenController(private val context: Context? = null) {

    companion object {
        private const val TAG = "LocalSirenController"
    }

    @Volatile
    var isSirenActive: Boolean = false
        private set

    private var toneGenerator: ToneGenerator? = null

    /**
     * Evaluates and updates siren status based on SOS state and network reachability.
     */
    @Synchronized
    fun updateSirenState(currentState: SosState, isRemoteAvailable: Boolean) {
        val shouldSoundSiren = (currentState == SosState.ACTIVE_SOS && !isRemoteAvailable)

        if (shouldSoundSiren && !isSirenActive) {
            startSiren()
        } else if (!shouldSoundSiren && isSirenActive) {
            stopSiren()
        }
    }

    @Synchronized
    fun startSiren() {
        if (isSirenActive) return
        isSirenActive = true
        Log.w(TAG, "LOCAL EMERGENCY SIREN ACTIVATED (ACTIVE_SOS + Offline)")
        try {
            toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, 100)
            toneGenerator?.startTone(ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK, 5000)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start tone generator for siren: ${e.message}")
        }
    }

    @Synchronized
    fun stopSiren() {
        if (!isSirenActive) return
        isSirenActive = false
        Log.i(TAG, "LOCAL EMERGENCY SIREN STOPPED")
        try {
            toneGenerator?.stopTone()
            toneGenerator?.release()
            toneGenerator = null
        } catch (e: Exception) {
            Log.w(TAG, "Failed to release tone generator: ${e.message}")
        }
    }
}
