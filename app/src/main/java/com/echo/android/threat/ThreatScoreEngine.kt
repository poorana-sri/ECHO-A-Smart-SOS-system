package com.echo.android.threat

import android.util.Log
import com.echo.android.ai.AudioEvent
import com.echo.android.ai.ClassifierResult
import com.echo.android.ai.EmergencyClass
import com.echo.android.ai.ModelStatus
import com.echo.android.ai.SensorEvidence
import com.echo.android.ai.ThreatEngineInterface
import com.echo.android.ai.ThreatUpdate
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Real implementation of [ThreatEngineInterface] fusing acoustic evidence from Developer 2
 * with physical accelerometer/gyroscope evidence from [SensorEvidenceCollector].
 *
 * Decoupled from [SosStateMachine].
 */
class ThreatScoreEngine(
    val config: ThreatScoreConfig = ThreatScoreConfig()
) : ThreatEngineInterface {

    companion object {
        private const val TAG = "ThreatScoreEngine"
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

    @Volatile
    var lastSensorEvidence: SensorEvidence? = null
        private set

    override fun onClassifierResult(result: ClassifierResult, sensorEvidence: SensorEvidence?) {
        lastClassifierResult = result
        lastSensorEvidence = sensorEvidence

        if (!isArmed) {
            Log.d(TAG, "Monitoring disarmed. Ignoring inference result.")
            return
        }

        // 1. Acoustic Component Score (0 - 100)
        val acousticScore: Float = when (result.predictedClass) {
            EmergencyClass.NORMAL -> 0.0f
            EmergencyClass.ACCIDENT -> config.baseAccidentScore * result.accidentProbability + 15f
            EmergencyClass.DISTRESS -> config.baseDistressScore * result.distressProbability + 10f
            EmergencyClass.VIOLENT_INCIDENT -> config.baseViolentScore * result.violentIncidentProbability + 20f
        }

        // 2. Physical Sensor Component Score (0 - 100)
        val accelScore = (sensorEvidence?.accelerationScore ?: 0.0f) * 100.0f
        val gyroScore = (sensorEvidence?.gyroScore ?: 0.0f) * 100.0f
        val motionBonus = if (sensorEvidence?.accelerationAnomaly == true || sensorEvidence?.erraticMovement == true) 20.0f else 0.0f

        // 3. Fused Calculation
        val rawFused = if (result.predictedClass == EmergencyClass.NORMAL) {
            // Decay previous score towards 0
            max(0.0f, (currentThreatScore * config.decayFactor - config.decaySubtrahend))
        } else {
            val fusedIncoming = (acousticScore * config.acousticWeight) +
                    (accelScore * config.accelWeight) +
                    (gyroScore * config.gyroWeight) +
                    motionBonus
            // Reflect highest immediate fused evidence while retaining historical memory
            min(100.0f, max(currentThreatScore * 0.40f + fusedIncoming * 0.60f, fusedIncoming))
        }

        currentThreatScore = rawFused.roundToInt().coerceIn(0, 100)
        val thresholdCrossed = currentThreatScore >= config.sosThreshold

        val evidenceStatus = when {
            thresholdCrossed -> "SOS THRESHOLD REACHED ($currentThreatScore >= ${config.sosThreshold})"
            currentThreatScore > 35 -> "ELEVATED THREAT ($currentThreatScore/100)"
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

        Log.i(TAG, "ThreatUpdate: Score=$currentThreatScore/100 Class=${result.predictedClass} Sensor=$sensorEvidence Status='$evidenceStatus'")
        notifyListeners(update)
    }

    override fun onAudioEvent(event: AudioEvent) {
        lastAudioEvent = event
    }

    override fun onMonitoringStateChanged(isArmed: Boolean) {
        this.isArmed = isArmed
        if (!isArmed) {
            currentThreatScore = 0
        }
        Log.i(TAG, "ThreatScoreEngine armed state changed to $isArmed")
    }

    override fun triggerManualSos() {
        Log.w(TAG, "Manual SOS triggered directly via ThreatScoreEngine.")
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
            evidenceStatus = "MANUAL SOS ACTIVATED"
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
        for (l in listeners) {
            try {
                l.onThreatUpdate(update)
            } catch (e: Exception) {
                Log.e(TAG, "Error notifying threat update listener", e)
            }
        }
    }
}
