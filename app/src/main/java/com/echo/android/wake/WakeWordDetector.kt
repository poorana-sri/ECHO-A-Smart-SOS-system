package com.echo.android.wake

import android.util.Log
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Interface contract for lightweight on-device keyword spotting.
 *
 * Requirements:
 * - Dedicated to spotting the wake phrase: "Hey Echo".
 * - Must NOT run heavy YAMNet continuously just to detect the wake phrase.
 * - Allows drop-in replacement with a fully-trained KWS TFLite model in production.
 */
interface WakeWordDetector {

    /**
     * Starts listening for the wake phrase.
     *
     * @param onWakeWordDetected Callback invoked when "Hey Echo" is detected.
     */
    fun startListening(onWakeWordDetected: () -> Unit)

    /**
     * Stops listening.
     */
    fun stopListening()

    /**
     * Ingests a short PCM audio slice from the continuous capture stream.
     *
     * @param pcmData 16-bit PCM mono samples at 16kHz.
     * @param length Number of valid samples in [pcmData].
     */
    fun processAudio(pcmData: ShortArray, length: Int = pcmData.size)

    /**
     * Manually triggers detection (useful for unit tests and UI demo simulations).
     */
    fun simulateDetection()
}

/**
 * Lightweight, energy/VAD and cadence-gated prototype detector for "Hey Echo".
 *
 * Algorithm (Hackathon Prototype):
 * - Maintains a rolling window of short-term frame energies (e.g. 50ms frames).
 * - Identifies two distinct acoustic syllable bursts matching the temporal rhythm of "Hey" -> pause -> "Echo".
 * - Designed to be CPU and battery efficient.
 *
 * NOTE: Hackathon prototype implementation.
 */
class DefaultWakeWordDetector(
    private val sampleRateHz: Int = 16000,
    private val energyThreshold: Float = 0.12f,
    private val minCadenceFrames: Int = 4, // ~200ms
    private val maxCadenceFrames: Int = 18 // ~900ms
) : WakeWordDetector {

    companion object {
        private const val TAG = "DefaultWakeWordDetector"
        private const val FRAME_SIZE_MS = 50 // 50ms analysis frames (800 samples @ 16kHz)
    }

    private val frameSamples = sampleRateHz * FRAME_SIZE_MS / 1000
    private var callback: (() -> Unit)? = null
    private var isListening: Boolean = false

    // Frame accumulation buffer
    private val frameBuffer = ShortArray(frameSamples)
    private var frameBufferCount = 0

    // Cadence state machine
    private var consecutiveActiveFrames = 0
    private var consecutiveSilenceFrames = 0
    private var burstCount = 0
    private var lastTriggerTimeMs = 0L

    override fun startListening(onWakeWordDetected: () -> Unit) {
        this.callback = onWakeWordDetected
        this.isListening = true
        resetCadence()
        Log.i(TAG, "WakeWordDetector started listening for \"Hey Echo\"")
    }

    override fun stopListening() {
        this.isListening = false
        this.callback = null
        resetCadence()
        Log.i(TAG, "WakeWordDetector stopped")
    }

    override fun processAudio(pcmData: ShortArray, length: Int) {
        if (!isListening || length <= 0) return

        var offset = 0
        while (offset < length) {
            val needed = frameSamples - frameBufferCount
            val toCopy = needed.coerceAtMost(length - offset)

            System.arraycopy(pcmData, offset, frameBuffer, frameBufferCount, toCopy)
            frameBufferCount += toCopy
            offset += toCopy

            if (frameBufferCount >= frameSamples) {
                analyzeFrame(frameBuffer, frameSamples)
                frameBufferCount = 0
            }
        }
    }

    private fun analyzeFrame(frame: ShortArray, length: Int) {
        val rms = computeFrameRms(frame, length)
        val isActive = rms >= energyThreshold

        if (isActive) {
            consecutiveActiveFrames++
            consecutiveSilenceFrames = 0

            // If we detected a prior burst and a brief dip, and now a second burst
            if (burstCount == 1 && consecutiveActiveFrames in 2..minCadenceFrames + 3) {
                burstCount = 2
            }
        } else {
            if (consecutiveActiveFrames >= minCadenceFrames / 2) {
                if (burstCount == 0) {
                    burstCount = 1 // First syllable ("Hey") completed
                }
            }
            consecutiveSilenceFrames++
            consecutiveActiveFrames = 0

            // If silence is too long, reset cadence
            if (consecutiveSilenceFrames > 8) {
                if (burstCount == 2) {
                    // Cadence completed ("Hey" + dip + "Echo")
                    triggerWakeWord()
                }
                resetCadence()
            }
        }
    }

    private fun triggerWakeWord() {
        val now = System.currentTimeMillis()
        // Debounce triggers (min 2 seconds between detections)
        if (now - lastTriggerTimeMs > 2000L) {
            lastTriggerTimeMs = now
            Log.i(TAG, "Wake phrase \"Hey Echo\" detected by cadence engine!")
            callback?.invoke()
        }
    }

    override fun simulateDetection() {
        Log.i(TAG, "Simulated \"Hey Echo\" wake word trigger.")
        callback?.invoke()
    }

    private fun resetCadence() {
        consecutiveActiveFrames = 0
        consecutiveSilenceFrames = 0
        burstCount = 0
    }

    private fun computeFrameRms(frame: ShortArray, length: Int): Float {
        var sum = 0.0
        for (i in 0 until length) {
            val norm = frame[i] / 32768.0
            sum += (norm * norm)
        }
        return sqrt(sum / length).toFloat()
    }
}
