package com.echo.android.threat

/**
 * Configuration parameters for the Threat Score calculation.
 *
 * NOTE: 70 is a hackathon prototype calibration value only.
 * Weights and thresholds are configurable.
 */
data class ThreatScoreConfig(
    val acousticWeight: Float = 0.55f,
    val accelWeight: Float = 0.25f,
    val gyroWeight: Float = 0.20f,
    val sosThreshold: Int = 70,
    val decayFactor: Float = 0.80f,
    val decaySubtrahend: Float = 5.0f,
    val baseAccidentScore: Float = 60.0f,
    val baseDistressScore: Float = 55.0f,
    val baseViolentScore: Float = 65.0f
)
