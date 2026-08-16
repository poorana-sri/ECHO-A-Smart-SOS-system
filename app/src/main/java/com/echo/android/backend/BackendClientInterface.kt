package com.echo.android.backend

import com.echo.android.ai.LocationUpdate
import com.echo.android.ai.SosEvent

/**
 * Contract between Developer 3's SOS system and Developer 4's Backend / Cloud Functions.
 */
interface BackendClientInterface {

    /**
     * Creates a new incident document in the backend database.
     * @return Result containing the unique incidentId
     */
    suspend fun createIncident(sosEvent: SosEvent): Result<String>

    /**
     * Uploads AES-256-GCM encrypted audio ciphertext + IV to backend storage.
     */
    suspend fun uploadEncryptedAudio(incidentId: String, ciphertext: ByteArray, iv: ByteArray): Result<Unit>

    /**
     * Triggers Twilio automated emergency call + SMS alerts to all configured contacts.
     */
    suspend fun triggerTwilioAlert(incidentId: String): Result<Unit>

    /**
     * Begins live location sharing session for the given incident.
     */
    suspend fun startLiveLocation(incidentId: String): Result<Unit>

    /**
     * Relays a new GPS/network location update to the backend live dashboard.
     */
    suspend fun sendLocationUpdate(update: LocationUpdate): Result<Unit>

    /**
     * Stops live location streaming upon SOS resolution.
     */
    suspend fun stopLiveLocation(incidentId: String): Result<Unit>

    /**
     * Sends the resolution notification SMS to emergency contacts.
     */
    suspend fun sendResolutionSms(incidentId: String): Result<Unit>
}
