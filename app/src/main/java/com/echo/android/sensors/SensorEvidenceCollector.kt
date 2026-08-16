package com.echo.android.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import com.echo.android.ai.SensorEvidence
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Collects physical motion evidence from Accelerometer and Gyroscope sensors.
 *
 * Requirements & Philosophy:
 * - Accelerometer detects sudden impact-like changes and anomalies (e.g. vehicle crash, hard drop).
 * - Gyroscope detects abnormal angular velocity and erratic rotation.
 * - Sensors act strictly as supporting evidence; sensors ALONE never trigger an emergency.
 * - Robust error handling for missing sensors or permission issues.
 */
class SensorEvidenceCollector(private val context: Context) : SensorEventListener {

    companion object {
        private const val TAG = "SensorEvidenceCollector"

        // Default prototype thresholds (configurable)
        const val DEFAULT_GRAVITY = 9.80665f
        const val DEFAULT_ACCEL_ANOMALY_THRESHOLD = 12.0f // m/s^2 deviation from 1G
        const val DEFAULT_GYRO_ANOMALY_THRESHOLD = 3.5f   // rad/s angular speed
        const val SENSOR_WINDOW_EXPIRY_MS = 1500L         // Evidence remains active for 1.5s
    }

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
    private val accelerometer: Sensor? = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope: Sensor? = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    var accelAnomalyThreshold: Float = DEFAULT_ACCEL_ANOMALY_THRESHOLD
    var gyroAnomalyThreshold: Float = DEFAULT_GYRO_ANOMALY_THRESHOLD

    @Volatile
    var isCollecting: Boolean = false
        private set

    // Sensor availability status
    val isAccelerometerAvailable: Boolean = accelerometer != null
    val isGyroscopeAvailable: Boolean = gyroscope != null

    // Real-time sensor state
    @Volatile
    private var lastAccelMagnitude: Float = DEFAULT_GRAVITY

    @Volatile
    private var lastGyroMagnitude: Float = 0f

    @Volatile
    private var lastAccelAnomalyTimeMs: Long = 0L

    @Volatile
    private var lastGyroAnomalyTimeMs: Long = 0L

    @Volatile
    private var maxRecentAccelScore: Float = 0f

    @Volatile
    private var maxRecentGyroScore: Float = 0f

    // Synthetic override for demo scenarios
    @Volatile
    private var syntheticEvidence: SensorEvidence? = null

    /**
     * Starts listening to sensor updates.
     */
    fun startCollecting() {
        if (isCollecting || sensorManager == null) return

        isCollecting = true
        maxRecentAccelScore = 0f
        maxRecentGyroScore = 0f

        if (accelerometer != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME)
            Log.i(TAG, "Registered Accelerometer listener.")
        } else {
            Log.w(TAG, "Accelerometer sensor not present on this device.")
        }

        if (gyroscope != null) {
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME)
            Log.i(TAG, "Registered Gyroscope listener.")
        } else {
            Log.w(TAG, "Gyroscope sensor not present on this device.")
        }
    }

    /**
     * Stops listening and clears sensor state.
     */
    fun stopCollecting() {
        if (!isCollecting) return
        isCollecting = false
        sensorManager?.unregisterListener(this)
        syntheticEvidence = null
        Log.i(TAG, "Stopped sensor collection and unregistered listeners.")
    }

    /**
     * Returns the latest fused [SensorEvidence] frame.
     */
    fun getLatestEvidence(nowMs: Long = System.currentTimeMillis()): SensorEvidence {
        // Return synthetic override if active
        syntheticEvidence?.let {
            if (nowMs - it.timestampMs < SENSOR_WINDOW_EXPIRY_MS) {
                return it
            } else {
                syntheticEvidence = null
            }
        }

        val accelActive = (nowMs - lastAccelAnomalyTimeMs) < SENSOR_WINDOW_EXPIRY_MS
        val gyroActive = (nowMs - lastGyroAnomalyTimeMs) < SENSOR_WINDOW_EXPIRY_MS

        val accelScore = if (accelActive) maxRecentAccelScore else 0f
        val gyroScore = if (gyroActive) maxRecentGyroScore else 0f

        return SensorEvidence(
            timestampMs = nowMs,
            accelerationAnomaly = accelActive && accelScore > 0.35f,
            accelerationScore = accelScore,
            erraticMovement = gyroActive && gyroScore > 0.30f,
            gyroScore = gyroScore
        )
    }

    /**
     * Injects synthetic sensor evidence for testing & demo scenarios.
     */
    fun injectSyntheticEvidence(
        accelerationAnomaly: Boolean,
        accelerationScore: Float,
        erraticMovement: Boolean,
        gyroScore: Float,
        timestampMs: Long = System.currentTimeMillis()
    ) {
        syntheticEvidence = SensorEvidence(
            timestampMs = timestampMs,
            accelerationAnomaly = accelerationAnomaly,
            accelerationScore = accelerationScore.coerceIn(0f, 1f),
            erraticMovement = erraticMovement,
            gyroScore = gyroScore.coerceIn(0f, 1f)
        )
        Log.i(TAG, "Injected synthetic sensor evidence: AccelAnomaly=$accelerationAnomaly ($accelerationScore), GyroErratic=$erraticMovement ($gyroScore)")
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (!isCollecting || event == null) return

        val now = System.currentTimeMillis()

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                val x = event.values[0]
                val y = event.values[1]
                val z = event.values[2]
                val magnitude = sqrt((x * x + y * y + z * z).toDouble()).toFloat()
                lastAccelMagnitude = magnitude

                val deviation = abs(magnitude - DEFAULT_GRAVITY)
                if (deviation >= accelAnomalyThreshold) {
                    lastAccelAnomalyTimeMs = now
                    // Scale score from 0.0 to 1.0 (normalized over 12 m/s^2 to 35 m/s^2)
                    val rawScore = (deviation - accelAnomalyThreshold) / 20.0f + 0.4f
                    maxRecentAccelScore = min(1.0f, max(maxRecentAccelScore, rawScore))
                    Log.d(TAG, "Accelerometer anomaly detected! Mag=$magnitude, Dev=$deviation, Score=$maxRecentAccelScore")
                } else if (now - lastAccelAnomalyTimeMs > SENSOR_WINDOW_EXPIRY_MS) {
                    maxRecentAccelScore = max(0f, maxRecentAccelScore * 0.9f)
                }
            }

            Sensor.TYPE_GYROSCOPE -> {
                val gx = event.values[0]
                val gy = event.values[1]
                val gz = event.values[2]
                val angularSpeed = sqrt((gx * gx + gy * gy + gz * gz).toDouble()).toFloat()
                lastGyroMagnitude = angularSpeed

                if (angularSpeed >= gyroAnomalyThreshold) {
                    lastGyroAnomalyTimeMs = now
                    val rawScore = (angularSpeed - gyroAnomalyThreshold) / 5.0f + 0.35f
                    maxRecentGyroScore = min(1.0f, max(maxRecentGyroScore, rawScore))
                    Log.d(TAG, "Gyroscope erratic rotation detected! Speed=$angularSpeed rad/s, Score=$maxRecentGyroScore")
                } else if (now - lastGyroAnomalyTimeMs > SENSOR_WINDOW_EXPIRY_MS) {
                    maxRecentGyroScore = max(0f, maxRecentGyroScore * 0.9f)
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // No action needed for basic anomaly detection
    }
}
