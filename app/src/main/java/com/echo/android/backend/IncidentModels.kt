package com.echo.android.backend

import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/**
 * Helper object for deserializing Incident and LocationUpdate from JSON strings.
 */
object IncidentModels {
    fun incidentFromJson(json: String): Incident = Incident.fromJson(JSONObject(json))
    fun locationUpdateFromJson(json: String, incidentId: String): LocationUpdate {
        return try { LocationUpdate.fromJson(JSONObject(json)) }
        catch (_: Exception) { LocationUpdate(incidentId = incidentId, latitude = 0.0, longitude = 0.0, accuracy = 0f) }
    }
}

/**
 * Geographic location update for emergency live tracking.
 */
data class LocationUpdate(
    val incidentId: String,
    val latitude: Double,
    val longitude: Double,
    val accuracy: Float,
    val timestampMs: Long = System.currentTimeMillis(),
    val isMock: Boolean = false,
    val status: String = "ACTIVE"
) {
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("incidentId", incidentId)
            put("latitude", latitude)
            put("longitude", longitude)
            put("accuracy", accuracy.toDouble())
            put("timestampMs", timestampMs)
            put("isMock", isMock)
            put("status", status)
        }
    }

    companion object {
        fun fromJson(json: JSONObject): LocationUpdate {
            return LocationUpdate(
                incidentId = json.getString("incidentId"),
                latitude = json.getDouble("latitude"),
                longitude = json.getDouble("longitude"),
                accuracy = json.getDouble("accuracy").toFloat(),
                timestampMs = json.optLong("timestampMs", System.currentTimeMillis()),
                isMock = json.optBoolean("isMock", false),
                status = json.optString("status", "ACTIVE")
            )
        }
    }
}

/**
 * Complete emergency incident metadata document.
 */
data class Incident(
    val incidentId: String = UUID.randomUUID().toString(),
    val userId: String = "usr_demo_device",
    val emergencyType: String, // ACCIDENT, DISTRESS, VIOLENT_INCIDENT, MANUAL
    val threatScore: Int,
    val createdAtMs: Long = System.currentTimeMillis(),
    val status: String = "ACTIVE_SOS", // ACTIVE_SOS, RESOLVED, CANCELLED
    val initialLocation: LocationUpdate? = null,
    val contactsAlerted: List<String> = emptyList(),
    val encryptedAudioRef: String? = null
) {
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("incidentId", incidentId)
            put("userId", userId)
            put("emergencyType", emergencyType)
            put("threatScore", threatScore)
            put("createdAtMs", createdAtMs)
            put("status", status)
            initialLocation?.let { put("initialLocation", it.toJson()) }
            put("contactsAlerted", JSONArray(contactsAlerted))
            encryptedAudioRef?.let { put("encryptedAudioRef", it) }
        }
    }

    companion object {
        fun fromJson(json: JSONObject): Incident {
            return Incident(
                incidentId = json.optString("incidentId", UUID.randomUUID().toString()),
                userId = json.optString("userId", "usr_demo_device"),
                emergencyType = json.optString("emergencyType", "UNKNOWN"),
                threatScore = json.optInt("threatScore", 0),
                createdAtMs = json.optLong("createdAtMs", System.currentTimeMillis()),
                status = json.optString("status", "ACTIVE_SOS")
            )
        }
    }
}

/**
 * Types of operations that can be queued during offline periods.
 */
enum class OperationType {
    CREATE_INCIDENT,
    ALERT_CONTACTS,
    LOCATION_UPDATE,
    UPLOAD_AUDIO,
    RESOLVE_INCIDENT
}

/**
 * Persistent queued operation record.
 */
data class QueuedOperation(
    val id: String = UUID.randomUUID().toString(),
    val type: OperationType,
    val incidentId: String,
    val payloadJson: String,
    val timestampMs: Long = System.currentTimeMillis(),
    var retryCount: Int = 0
) {
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("id", id)
            put("type", type.name)
            put("incidentId", incidentId)
            put("payloadJson", payloadJson)
            put("timestampMs", timestampMs)
            put("retryCount", retryCount)
        }
    }

    companion object {
        fun fromJson(json: JSONObject): QueuedOperation {
            return QueuedOperation(
                id = json.getString("id"),
                type = OperationType.valueOf(json.getString("type")),
                incidentId = json.getString("incidentId"),
                payloadJson = json.getString("payloadJson"),
                timestampMs = json.getLong("timestampMs"),
                retryCount = json.optInt("retryCount", 0)
            )
        }
    }
}
