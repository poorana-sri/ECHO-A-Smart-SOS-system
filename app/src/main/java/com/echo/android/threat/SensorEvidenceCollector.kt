package com.echo.android.threat

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import com.echo.android.ai.SensorEvidence
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Collects physical motion evidence from Android Accelerometer and Gyroscope.
 * Handles missing/unavailable sensors gracefully by returning neutral evidence.
 */
class SensorEvidenceCollector(context: Context) : SensorEventListener {

    companion object {
        private const val TAG = "SensorEvidenceCollector"
        private const val GRAVITY_STANDARD = 9.80665f
        private const val IMPACT_THRESHOLD_G = 2.5f // ~24.5 m/s^2
        private const val JERK_THRESHOLD_DPS = 4.0f // ~230 deg/s
    }

    private val sensorManager: SensorManager? =
        context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager

    private val accelerometer: Sensor? =
        sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private val gyroscope: Sensor? =
        sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    @Volatile
    private var isListening: Boolean = false

    @Volatile
    private var lastAccelMagnitude: Float = GRAVITY_STANDARD

    @Volatile
    private var peakAccelDelta: Float = 0.0f

    @Volatile
    private var lastGyroMagnitude: Float = 0.0f

    @Volatile
    private var peakGyroRate: Float = 0.0f

    /**
     * Starts listening to accelerometer and gyroscope if available.
     */
    fun startListening() {
        if (isListening || sensorManager == null) return

        try {
            accelerometer?.let {
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
            }
            gyroscope?.let {
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
            }
            isListening = true
            Log.i(TAG, "Sensor listeners registered. Accel=${accelerometer != null}, Gyro=${gyroscope != null}")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to register sensor listeners: ${e.message}")
        }
    }

    /**
     * Stops listening and clears peak values.
     */
    fun stopListening() {
        if (!isListening) return
        try {
            sensorManager?.unregisterListener(this)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to unregister sensor listeners: ${e.message}")
        }
        isListening = false
        peakAccelDelta = 0.0f
        peakGyroRate = 0.0f
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                val x = event.values[0]
                val y = event.values[1]
                val z = event.values[2]
                val mag = sqrt(x * x + y * y + z * z)
                val delta = abs(mag - GRAVITY_STANDARD)
                if (delta > peakAccelDelta) {
                    peakAccelDelta = delta
                }
                lastAccelMagnitude = mag
            }
            Sensor.TYPE_GYROSCOPE -> {
                val gx = event.values[0]
                val gy = event.values[1]
                val gz = event.values[2]
                val rotRate = sqrt(gx * gx + gy * gy + gz * gz)
                if (rotRate > peakGyroRate) {
                    peakGyroRate = rotRate
                }
                lastGyroMagnitude = rotRate
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // No-op
    }

    /**
     * Computes the current SensorEvidence snapshot and resets peak readings for the next window.
     */
    fun extractCurrentEvidence(): SensorEvidence {
        val accelDelta = peakAccelDelta
        val gyroRate = peakGyroRate

        // Reset peaks for subsequent measurement window
        peakAccelDelta = 0.0f
        peakGyroRate = 0.0f

        val accelG = accelDelta / GRAVITY_STANDARD
        val isImpact = accelG >= IMPACT_THRESHOLD_G
        val accelScore = (accelG / 4.0f).coerceIn(0.0f, 1.0f)

        val isErratic = gyroRate >= JERK_THRESHOLD_DPS
        val gyroScore = (gyroRate / 8.0f).coerceIn(0.0f, 1.0f)

        return SensorEvidence(
            timestampMs = System.currentTimeMillis(),
            accelerationAnomaly = isImpact,
            accelerationScore = accelScore,
            erraticMovement = isErratic,
            gyroScore = gyroScore
        )
    }

    val isAvailable: Boolean
        get() = accelerometer != null || gyroscope != null
}
