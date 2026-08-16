package com.echo.android.ai

import android.util.Log
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Standalone mock implementation of [ThreatEngineInterface].
 *
 * Used for Developer 1 prototype demoing before Developer 3's real sensor-fused
 * Threat Score Engine is integrated.
 *
 * Demonstrates:
 * - Threat score rise upon ACCIDENT, DISTRESS, or VIOLENT_INCIDENT classifications.
 * - Score decay back to 0 on NORMAL classifications.
 * - Threshold evaluation (SOS trigger threshold: 70/100).
 * - Thread-safe listener notification.
 */
class MockThreatEngine : ThreatEngineInterface {

    companion object {
        private const val TAG = "MockThreatEngine"
        const val SOS_THRESHOLD = 70

        // Shared singleton instance for standalone app demo
        val instance: MockThreatEngine by lazy { MockThreatEngine() }
    }

    private val listeners = CopyOnWriteArrayList<ThreatEngineInterface.ThreatUpdateListener>()

    @Volatile
    var currentThreatScore: Int = 0
        private set

    @Volatile
    var isArmed: Boolean = false
        private set

    @Volatile
    var lastClassifierResult: ClassifierResult? = null
        private set

    @Volatile
    var lastAudioEvent: AudioEvent? = null
        private set

    override fun onClassifierResult(result: ClassifierResult, sensorEvidence: SensorEvidence?) {
        lastClassifierResult = result

        if (!isArmed) {
            Log.d(TAG, "Ignoring inference result because Echo is disarmed.")
            return
        }

        // Mock Threat Score computation logic
        val rawDelta: Float = when (result.predictedClass) {
            EmergencyClass.NORMAL -> -15f // Linear decay towards 0
            EmergencyClass.ACCIDENT -> result.accidentProbability * 55f + 25f
            EmergencyClass.DISTRESS -> result.distressProbability * 50f + 20f
            EmergencyClass.VIOLENT_INCIDENT -> result.violentIncidentProbability * 65f + 25f
        }

        // Optional sensor fusion simulation if sensorEvidence passed
        val sensorBonus = if (sensorEvidence?.accelerationAnomaly == true) 25f else 0f

        val calculatedScore = if (result.predictedClass == EmergencyClass.NORMAL) {
            // Apply gradual decay
            max(0, (currentThreatScore * 0.75f - 5f).roundToInt())
        } else {
            min(100, (currentThreatScore * 0.4f + rawDelta + sensorBonus).roundToInt())
        }

        currentThreatScore = calculatedScore
        val thresholdCrossed = currentThreatScore >= SOS_THRESHOLD

        val evidenceStatus = when {
            thresholdCrossed -> "SOS THRESHOLD CROSSED ($currentThreatScore >= $SOS_THRESHOLD)"
            currentThreatScore > 35 -> "ELEVATED ACOUSTIC ACTIVITY ($currentThreatScore/100)"
            else -> "NOMINAL MONITORING ($currentThreatScore/100)"
        }

        val update = ThreatUpdate(
            timestampMs = result.timestampMs,
            acousticEvidence = result,
            sensorEvidence = sensorEvidence,
            threatScore = currentThreatScore,
            thresholdCrossed = thresholdCrossed,
            evidenceStatus = evidenceStatus
        )

        Log.i(
            TAG,
            "MockThreatUpdate: Class=${result.predictedClass} Prob=${String.format("%.2f", getProbability(result))} " +
                    "Score=$currentThreatScore/100 Status='$evidenceStatus'"
        )

        notifyListeners(update)
    }

    override fun onAudioEvent(event: AudioEvent) {
        lastAudioEvent = event
        Log.v(TAG, "YAMNet AudioEvent received with ${event.yamnetPredictions.size} predictions. Status=${event.modelStatus}")
    }

    override fun onMonitoringStateChanged(isArmed: Boolean) {
        this.isArmed = isArmed
        if (!isArmed) {
            currentThreatScore = 0
        }
        Log.i(TAG, "Monitoring state changed. isArmed=$isArmed")
    }

    override fun triggerManualSos() {
        Log.w(TAG, "Manual SOS triggered via ThreatEngineInterface!")
        currentThreatScore = 100
        val manualResult = ClassifierResult(
            timestampMs = System.currentTimeMillis(),
            normalProbability = 0.0f,
            accidentProbability = 0.0f,
            distressProbability = 1.0f,
            violentIncidentProbability = 0.0f,
            predictedClass = EmergencyClass.DISTRESS,
            modelStatus = ModelStatus.OK
        )
        val update = ThreatUpdate(
            timestampMs = System.currentTimeMillis(),
            acousticEvidence = manualResult,
            sensorEvidence = null,
            threatScore = 100,
            thresholdCrossed = true,
            evidenceStatus = "MANUAL SOS TRIGGERED"
        )
        notifyListeners(update)
    }

    override fun addThreatListener(listener: ThreatEngineInterface.ThreatUpdateListener) {
        listeners.add(listener)
    }

    override fun removeThreatListener(listener: ThreatEngineInterface.ThreatUpdateListener) {
        listeners.remove(listener)
    }

    private fun notifyListeners(update: ThreatUpdate) {
        for (listener in listeners) {
            try {
                listener.onThreatUpdate(update)
            } catch (e: Exception) {
                Log.e(TAG, "Error in threat update listener", e)
            }
        }
    }

    private fun getProbability(result: ClassifierResult): Float {
        return when (result.predictedClass) {
            EmergencyClass.NORMAL -> result.normalProbability
            EmergencyClass.ACCIDENT -> result.accidentProbability
            EmergencyClass.DISTRESS -> result.distressProbability
            EmergencyClass.VIOLENT_INCIDENT -> result.violentIncidentProbability
        }
    }
}
