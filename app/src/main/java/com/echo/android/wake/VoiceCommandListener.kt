package com.echo.android.wake

import android.util.Log

/**
 * Interface contract for voice cancellation commands during SOS countdown.
 *
 * Dedicated to recognizing: "Hey Echo, cancel SOS".
 *
 * Developer 1 owns this phrase detection primitive.
 * Developer 3's SOS State Machine subscribes to this interface to abort active countdowns.
 */
interface VoiceCommandListener {

    /**
     * Starts listening specifically for emergency cancellation phrases.
     *
     * @param onCancelDetected Callback executed when "Hey Echo, cancel SOS" is detected.
     */
    fun startListeningForCancel(onCancelDetected: () -> Unit)

    /**
     * Stops listening for cancellation commands.
     */
    fun stopListeningForCancel()

    /**
     * Ingests PCM frames during the active countdown window.
     */
    fun processAudio(pcmData: ShortArray, length: Int = pcmData.size)

    /**
     * Manually triggers cancellation detection (useful for tests and demo UI).
     */
    fun simulateCancelCommand()
}

/**
 * Default prototype implementation of [VoiceCommandListener].
 *
 * NOTE: Hackathon prototype implementation.
 */
class DefaultVoiceCommandDetector(
    private val sampleRateHz: Int = 16000
) : VoiceCommandListener {

    companion object {
        private const val TAG = "VoiceCommandDetector"
    }

    private var onCancelCallback: (() -> Unit)? = null
    private var isListening: Boolean = false

    override fun startListeningForCancel(onCancelDetected: () -> Unit) {
        this.onCancelCallback = onCancelDetected
        this.isListening = true
        Log.i(TAG, "VoiceCommandDetector started listening for \"Hey Echo, cancel SOS\"")
    }

    override fun stopListeningForCancel() {
        this.isListening = false
        this.onCancelCallback = null
        Log.i(TAG, "VoiceCommandDetector stopped")
    }

    override fun processAudio(pcmData: ShortArray, length: Int) {
        if (!isListening || length <= 0) return

        // In a full implementation, a focused keyword spotter or small speech command model processes this.
        // For the hackathon prototype, simulateCancelCommand provides immediate testability.
    }

    override fun simulateCancelCommand() {
        Log.i(TAG, "Voice cancellation command \"Hey Echo, cancel SOS\" detected/simulated!")
        onCancelCallback?.invoke()
    }
}
