package com.echo.android.backend

import android.util.Log
import com.echo.android.ai.LocationUpdate
import com.echo.android.ai.SosEvent
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Mock implementation of [BackendClientInterface] for Developer 3 standalone testing,
 * offline simulation, and hackathon demonstration.
 */
class MockBackendClient : BackendClientInterface {

    companion object {
        private const val TAG = "MockBackendClient"
    }

    @Volatile
    var isNetworkAvailable: Boolean = true

    @Volatile
    var shouldSimulateErrors: Boolean = false

    val recordedIncidents = CopyOnWriteArrayList<SosEvent>()
    val recordedAudioUploads = CopyOnWriteArrayList<Pair<String, Int>>() // incidentId -> byteSize
    val recordedAlertTriggers = CopyOnWriteArrayList<String>() // incidentId
    val recordedLocationUpdates = CopyOnWriteArrayList<LocationUpdate>()
    val recordedResolutions = CopyOnWriteArrayList<String>() // incidentId

    override suspend fun createIncident(sosEvent: SosEvent): Result<String> {
        Log.i(TAG, "createIncident called: $sosEvent (network=$isNetworkAvailable)")
        if (!isNetworkAvailable) {
            return Result.failure(IllegalStateException("Network unavailable (offline mode)."))
        }
        if (shouldSimulateErrors) {
            return Result.failure(RuntimeException("Simulated backend 500 error."))
        }
        recordedIncidents.add(sosEvent)
        return Result.success(sosEvent.incidentId)
    }

    override suspend fun uploadEncryptedAudio(
        incidentId: String,
        ciphertext: ByteArray,
        iv: ByteArray
    ): Result<Unit> {
        Log.i(TAG, "uploadEncryptedAudio called for $incidentId: size=${ciphertext.size} bytes, iv=${iv.size} bytes")
        if (!isNetworkAvailable) {
            return Result.failure(IllegalStateException("Network unavailable (offline mode)."))
        }
        recordedAudioUploads.add(incidentId to ciphertext.size)
        return Result.success(Unit)
    }

    override suspend fun triggerTwilioAlert(incidentId: String): Result<Unit> {
        Log.i(TAG, "triggerTwilioAlert called for incident: $incidentId")
        if (!isNetworkAvailable) {
            return Result.failure(IllegalStateException("Network unavailable (offline mode)."))
        }
        recordedAlertTriggers.add(incidentId)
        return Result.success(Unit)
    }

    override suspend fun startLiveLocation(incidentId: String): Result<Unit> {
        Log.i(TAG, "startLiveLocation called for incident: $incidentId")
        if (!isNetworkAvailable) {
            return Result.failure(IllegalStateException("Network unavailable (offline mode)."))
        }
        return Result.success(Unit)
    }

    override suspend fun sendLocationUpdate(update: LocationUpdate): Result<Unit> {
        Log.i(TAG, "sendLocationUpdate called: lat=${update.latitude}, lng=${update.longitude}, src=${update.source}")
        if (!isNetworkAvailable) {
            return Result.failure(IllegalStateException("Network unavailable (offline mode)."))
        }
        recordedLocationUpdates.add(update)
        return Result.success(Unit)
    }

    override suspend fun stopLiveLocation(incidentId: String): Result<Unit> {
        Log.i(TAG, "stopLiveLocation called for incident: $incidentId")
        return Result.success(Unit)
    }

    override suspend fun sendResolutionSms(incidentId: String): Result<Unit> {
        Log.i(TAG, "sendResolutionSms called for incident: $incidentId")
        if (!isNetworkAvailable) {
            return Result.failure(IllegalStateException("Network unavailable (offline mode)."))
        }
        recordedResolutions.add(incidentId)
        return Result.success(Unit)
    }

    fun clear() {
        recordedIncidents.clear()
        recordedAudioUploads.clear()
        recordedAlertTriggers.clear()
        recordedLocationUpdates.clear()
        recordedResolutions.clear()
    }
}
