package com.echo.android.sos

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.echo.android.backend.BackendClientInterface
import com.echo.android.backend.IncidentModels
import com.echo.android.backend.OperationType
import com.echo.android.backend.QueuedOperation
import com.echo.android.contacts.EmergencyContact
import org.json.JSONArray
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Lightweight persistent queue for failed backend operations during ACTIVE_SOS.
 *
 * Survives transient connectivity loss by persisting to SharedPreferences.
 * Operations are retried when connectivity is restored.
 *
 * NOTE: This is a hackathon MVP queue — not a distributed reliable queue.
 */
class OfflineQueue(private val context: Context) {

    companion object {
        private const val TAG = "OfflineQueue"
        private const val PREFS_NAME = "echo_offline_queue"
        private const val KEY_QUEUE = "queued_operations"
        private const val MAX_RETRY_ATTEMPTS = 5
    }

    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val memoryQueue = CopyOnWriteArrayList<QueuedOperation>()

    init {
        loadFromDisk()
    }

    fun enqueue(operation: QueuedOperation) {
        if (memoryQueue.none { it.id == operation.id }) {
            memoryQueue.add(operation)
            persistToDisk()
            Log.i(TAG, "Queued operation: ${operation.type} for incident ${operation.incidentId} (id=${operation.id})")
        }
    }

    fun dequeueAll(): List<QueuedOperation> = memoryQueue.toList()

    fun remove(id: String) {
        val removed = memoryQueue.removeIf { it.id == id }
        if (removed) {
            persistToDisk()
            Log.d(TAG, "Removed queued operation: $id")
        }
    }

    fun size(): Int = memoryQueue.size

    fun clear() {
        memoryQueue.clear()
        persistToDisk()
    }

    /**
     * Attempts to retry all queued operations when connectivity is restored.
     * Returns the count of successfully replayed operations.
     * Failed ops that exceed MAX_RETRY_ATTEMPTS are discarded.
     */
    fun retryAll(backend: BackendClientInterface, contacts: List<EmergencyContact>): Int {
        var successCount = 0
        val toRetry = memoryQueue.toList()

        for (op in toRetry) {
            if (op.retryCount >= MAX_RETRY_ATTEMPTS) {
                Log.w(TAG, "Dropping operation ${op.id} (${op.type}) after ${op.retryCount} retries.")
                remove(op.id)
                continue
            }

            val success = try {
                when (op.type) {
                    OperationType.CREATE_INCIDENT -> {
                        val incident = IncidentModels.incidentFromJson(op.payloadJson)
                        backend.createIncident(incident)
                    }
                    OperationType.ALERT_CONTACTS -> {
                        backend.triggerEmergencyAlerts(op.incidentId, contacts, "ECHO SOS ACTIVE — Emergency response required.")
                    }
                    OperationType.LOCATION_UPDATE -> {
                        val location = IncidentModels.locationUpdateFromJson(op.payloadJson, op.incidentId)
                        backend.updateLiveLocation(op.incidentId, location)
                    }
                    OperationType.UPLOAD_AUDIO -> {
                        // Audio bytes not persisted to disk for privacy; skip if missing
                        Log.w(TAG, "Cannot retry UPLOAD_AUDIO — audio bytes not persisted for privacy.")
                        true
                    }
                    OperationType.RESOLVE_INCIDENT -> {
                        backend.resolveIncident(op.incidentId) &&
                                backend.sendResolutionAlerts(op.incidentId, contacts, "ECHO SOS has been RESOLVED by the user.")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error during retry of operation ${op.id}", e)
                false
            }

            if (success) {
                remove(op.id)
                successCount++
                Log.i(TAG, "Successfully replayed queued operation: ${op.type} (${op.incidentId})")
            } else {
                op.retryCount++
                persistToDisk()
                Log.w(TAG, "Retry ${op.retryCount}/$MAX_RETRY_ATTEMPTS failed for ${op.type} (${op.incidentId})")
            }
        }

        return successCount
    }

    private fun loadFromDisk() {
        memoryQueue.clear()
        try {
            val jsonStr = prefs.getString(KEY_QUEUE, null) ?: return
            val arr = JSONArray(jsonStr)
            for (i in 0 until arr.length()) {
                memoryQueue.add(QueuedOperation.fromJson(arr.getJSONObject(i)))
            }
            if (memoryQueue.isNotEmpty()) {
                Log.i(TAG, "Loaded ${memoryQueue.size} pending queued operations from disk.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading offline queue from disk", e)
        }
    }

    private fun persistToDisk() {
        try {
            val arr = JSONArray()
            for (op in memoryQueue) arr.put(op.toJson())
            prefs.edit().putString(KEY_QUEUE, arr.toString()).apply()
        } catch (e: Exception) {
            Log.e(TAG, "Error persisting offline queue to disk", e)
        }
    }
}
