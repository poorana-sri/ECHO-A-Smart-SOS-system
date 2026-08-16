package com.echo.android.ai

/**
 * Shared data contracts for ECHO hackathon prototype.
 *
 * NOTE: This is a hackathon prototype, NOT a production system.
 * These contracts maintain exact compatibility across all 4 developer modules:
 * - Developer 1: Audio Capture, Foreground Service, Wake Layer, AI Runtime Shell
 * - Developer 2: YAMNet Feature Extraction, LSTM/GRU Sequence Classifier
 * - Developer 3: Threat Score, Sensor Fusion, SOS State Machine
 * - Developer 4: Firebase, Encryption, Auth, Twilio Backend, Dashboard
 */

/**
 * Acoustic event output emitted by YAMNet.
 * Answers: "What acoustic events are present in this audio slice?" (e.g. Speech, Scream, Siren, Crash).
 * NOTE: YAMNet outputs 521-class AudioSet labels/probabilities, NOT the 4 Echo emergency classes.
 */
data class AudioEvent(
    val timestampMs: Long,
    val yamnetPredictions: List<Pair<String, Float>>, // label -> probability
    val modelStatus: ModelStatus // OK, DEGRADED, UNAVAILABLE
)

/**
 * Sequence classifier result mapping temporal patterns across audio windows.
 * Answers: "What does this acoustic sequence resemble over time?"
 * This 4-class output is produced exclusively by the Echo Sequence Classifier (LSTM/GRU), NOT YAMNet.
 */
data class ClassifierResult(
    val timestampMs: Long,
    val normalProbability: Float,
    val accidentProbability: Float,
    val distressProbability: Float,
    val violentIncidentProbability: Float,
    val predictedClass: EmergencyClass, // NORMAL, ACCIDENT, DISTRESS, VIOLENT_INCIDENT
    val modelStatus: ModelStatus
)

/**
 * Four primary acoustic categories defined in the team specification.
 */
enum class EmergencyClass {
    NORMAL,
    ACCIDENT,
    DISTRESS,
    VIOLENT_INCIDENT
}

/**
 * Operating status of AI models and acoustic inference pipeline.
 * - OK: Real TensorFlow Lite model loaded and executing inference.
 * - DEGRADED: Model unavailable or error during inference; running in fallback mode.
 * - UNAVAILABLE: Audio capture or AI runtime not initialized.
 */
enum class ModelStatus {
    OK,
    DEGRADED,
    UNAVAILABLE
}

/**
 * Physical sensor evidence computed by Developer 3 from Accelerometer and Gyroscope.
 */
data class SensorEvidence(
    val timestampMs: Long,
    val accelerationAnomaly: Boolean,
    val accelerationScore: Float = 0.0f,
    val erraticMovement: Boolean = false,
    val gyroScore: Float = 0.0f
)

/**
 * Fused threat score update produced by Developer 3's Threat Engine.
 */
data class ThreatUpdate(
    val timestampMs: Long,
    val acousticEvidence: ClassifierResult,
    val sensorEvidence: SensorEvidence?,
    val threatScore: Int, // 0 - 100
    val thresholdCrossed: Boolean,
    val evidenceStatus: String
)

/**
 * SOS Event contract shared with Developer 3 (State Machine) and Developer 4 (Backend/Storage).
 */
data class SosEvent(
    val incidentId: String,
    val userId: String,
    val emergencyType: String,
    val threatScore: Int,
    val timestamp: Long,
    val status: String // MONITORING, COUNTDOWN, CANCELLED, ACTIVE_SOS, RESOLVED, DEGRADED
)

/**
 * Standard SOS State Machine states.
 */
enum class SosState {
    MONITORING,
    COUNTDOWN,
    CANCELLED,
    ACTIVE_SOS,
    RESOLVED,
    DEGRADED
}

/**
 * Location data source status.
 */
enum class LocationSource {
    LIVE,
    LAST_KNOWN,
    UNAVAILABLE
}

/**
 * Real-time location update contract for Developer 3 & Developer 4.
 */
data class LocationUpdate(
    val incidentId: String,
    val latitude: Double,
    val longitude: Double,
    val timestamp: Long,
    val accuracy: Float,
    val source: LocationSource = LocationSource.LIVE
)
