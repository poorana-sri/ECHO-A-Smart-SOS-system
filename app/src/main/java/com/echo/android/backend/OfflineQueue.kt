package com.echo.android.backend

import android.util.Log
import com.echo.android.ai.LocationUpdate
import com.echo.android.ai.SosEvent
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * Lightweight queue for operations deferred due to offline / connectivity loss during ACTIVE_SOS.
 */
class OfflineQueue(private val backendClient: BackendClientInterface) {

    companion object {
        private const val TAG = "OfflineQueue"
    }

    enum class OperationType {
        CREATE_INCIDENT,
        ALERT_CONTACTS,
        UPLOAD_AUDIO,
        LOCATION_UPDATE,
        RESOLUTION
    }

    data class QueuedOperation(
        val id: String,
        val type: OperationType,
        val timestamp: Long,
        val incidentId: String,
        val sosEvent: SosEvent? = null,
        val locationUpdate: LocationUpdate? = null,
        val audioCiphertext: ByteArray? = null,
        val audioIv: ByteArray? = null
    )

    private val queue = ConcurrentLinkedQueue<QueuedOperation>()

    fun enqueue(operation: QueuedOperation) {
        Log.i(TAG, "Enqueued offline operation: ${operation.type} for incident ${operation.incidentId}")
        queue.add(operation)
    }

    fun size(): Int = queue.size

    fun isEmpty(): Boolean = queue.isEmpty()

    /**
     * Attempts to drain and execute all queued operations in order.
     * Keeps failed operations in queue if network remains offline.
     */
    suspend fun drainQueue(): Int {
        if (queue.isEmpty()) return 0

        var successfulDrains = 0
        val iterator = queue.iterator()

        while (iterator.hasNext()) {
            val op = iterator.next()
            val result = try {
                when (op.type) {
                    OperationType.CREATE_INCIDENT -> {
                        op.sosEvent?.let { backendClient.createIncident(it) } ?: Result.success(op.incidentId)
                    }
                    OperationType.ALERT_CONTACTS -> {
                        backendClient.triggerTwilioAlert(op.incidentId)
                    }
                    OperationType.UPLOAD_AUDIO -> {
                        if (op.audioCiphertext != null && op.audioIv != null) {
                            backendClient.uploadEncryptedAudio(op.incidentId, op.audioCiphertext, op.audioIv)
                        } else {
                            Result.success(Unit)
                        }
                    }
                    OperationType.LOCATION_UPDATE -> {
                        op.locationUpdate?.let { backendClient.sendLocationUpdate(it) } ?: Result.success(Unit)
                    }
                    OperationType.RESOLUTION -> {
                        backendClient.sendResolutionSms(op.incidentId)
                    }
                }
            } catch (e: Exception) {
                Result.failure(e)
            }

            if (result.isSuccess) {
                iterator.remove()
                successfulDrains++
                Log.d(TAG, "Successfully drained operation ${op.type} (${op.id})")
            } else {
                Log.w(TAG, "Operation ${op.type} (${op.id}) failed to drain: ${result.exceptionOrNull()?.message}")
                break // Stop draining if network still failing
            }
        }
        return successfulDrains
    }

    fun clear() {
        queue.clear()
    }
}
