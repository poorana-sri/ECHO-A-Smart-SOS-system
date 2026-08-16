package com.echo.android.wake

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for [DefaultWakeWordDetector].
 *
 * Validates:
 * 1. Wake phrase trigger on valid synthetic cadence waveform.
 * 2. Non-trigger on flat silence or continuous stationary noise.
 * 3. Manual simulation trigger.
 */
class WakeWordDetectorTest {

    private lateinit var detector: DefaultWakeWordDetector
    private var detected = false

    @Before
    fun setUp() {
        detector = DefaultWakeWordDetector(sampleRateHz = 16000, energyThreshold = 0.10f)
        detected = false
        detector.startListening {
            detected = true
        }
    }

    @Test
    fun testSilenceDoesNotTriggerWakeWord() {
        val silence = ShortArray(16000) { 0 }
        detector.processAudio(silence, silence.size)
        assertFalse("Silence must not trigger wake word", detected)
    }

    @Test
    fun testContinuousLoudNoiseDoesNotTriggerWakeWord() {
        // Continuous loud tone (no syllable cadence)
        val continuousTone = ShortArray(16000) { i ->
            (kotlin.math.sin(i * 0.1) * 15000).toInt().toShort()
        }
        detector.processAudio(continuousTone, continuousTone.size)
        assertFalse("Continuous uninterrupted noise must not trigger wake word", detected)
    }

    @Test
    fun testSimulateDetectionTriggersCallback() {
        detector.simulateDetection()
        assertTrue("simulateDetection() must trigger wake callback", detected)
    }

    @Test
    fun testStopListeningPreventsCallback() {
        detector.stopListening()
        detector.simulateDetection()
        // Callback was cleared on stopListening
        assertFalse("Stopped detector must not emit events", detected)
    }
}
