package com.echo.android.sos

import android.content.Context
import android.util.Log
import com.echo.android.ai.ClassifierResult
import com.echo.android.ai.EmergencyClass
import com.echo.android.ai.LocationUpdate
import com.echo.android.ai.ModelStatus
import com.echo.android.ai.SosEvent
import com.echo.android.ai.SosState
import com.echo.android.ai.ThreatEngineInterface
import com.echo.android.ai.ThreatUpdate
import com.echo.android.backend.BackendClientInterface
import com.echo.android.backend.MockBackendClient
import com.echo.android.backend.OfflineQueue
import com.echo.android.contacts.EmergencyContactManager
import com.echo.android.threat.SensorEvidenceCollector
import com.echo.android.threat.ThreatScoreEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.UUID

/**
 * Master orchestrator connecting Threat Engine, Sensor Fusion, SOS State Machine,
 * Countdown, Emergency Contacts, Location Tracking, Offline Queue, and Local Siren.
 */
class EchoSosOrchestrator(
    val context: Context,
    val threatEngine: ThreatScoreEngine = ThreatScoreEngine(),
    val sensorCollector: SensorEvidenceCollector = SensorEvidenceCollector(context),
    val stateMachine: SosStateMachine = SosStateMachine(),
    val countdownController: CountdownController = CountdownController(),
    val contactManager: EmergencyContactManager = EmergencyContactManager(context),
    val locationController: LocationController = LocationController(context),
    val backendClient: BackendClientInterface = MockBackendClient(),
    val localSirenController: LocalSirenController = LocalSirenController(context),
    private val coroutineScope: CoroutineScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
) : ThreatEngineInterface.ThreatUpdateListener, CountdownController.CountdownListener {

    companion object {
        private const val TAG = "EchoSosOrchestrator"
    }

    val offlineQueue: OfflineQueue = OfflineQueue(backendClient)

    @Volatile
    var activeIncidentId: String? = null
        private set

    @Volatile
    var isRemoteAvailable: Boolean = true
        private set

    init {
        threatEngine.addThreatListener(this)
        countdownController.setListener(this)

        locationController.setLocationListener { update ->
            onLocationUpdated(update)
        }

        stateMachine.addStateListener { previous, current ->
            Log.i(TAG, "Orchestrator observed state transition: $previous -> $current")
            localSirenController.updateSirenState(current, isRemoteAvailable)
        }
    }

    /**
     * Arms acoustic monitoring and sensor evidence collection.
     */
    fun arm() {
        threatEngine.onMonitoringStateChanged(true)
        sensorCollector.startListening()
        Log.i(TAG, "EchoSosOrchestrator ARMED.")
    }

    /**
     * Disarms monitoring and resets active threat state.
     */
    fun disarm() {
        threatEngine.onMonitoringStateChanged(false)
        sensorCollector.stopListening()
        if (stateMachine.currentState == SosState.COUNTDOWN) {
            cancelCountdown("Disarmed by user")
        }
        Log.i(TAG, "EchoSosOrchestrator DISARMED.")
    }

    /**
     * Receives acoustic inference result from Developer 1 / Developer 2, attaches physical sensor evidence,
     * and forwards to ThreatScoreEngine.
     */
    fun onAcousticInference(result: ClassifierResult) {
        val sensorEvidence = sensorCollector.extractCurrentEvidence()
        threatEngine.onClassifierResult(result, sensorEvidence)
    }

    override fun onThreatUpdate(update: ThreatUpdate) {
        if (update.thresholdCrossed && stateMachine.currentState == SosState.MONITORING) {
            Log.w(TAG, "Threat score crossed threshold (${update.threatScore} >= ${threatEngine.config.sosThreshold}). Triggering 10s countdown!")
            if (stateMachine.transitionTo(SosState.COUNTDOWN, "Threat Score ${update.threatScore} >= Threshold")) {
                countdownController.startCountdown()
            }
        }
    }

    override fun onTick(secondsRemaining: Int) {
        Log.d(TAG, "Pre-SOS countdown tick: $secondsRemaining s")
    }

    override fun onCountdownFinished() {
        Log.w(TAG, "Countdown completed. Promoting to ACTIVE_SOS.")
        confirmActiveSos(emergencyType = threatEngine.lastClassifierResult?.predictedClass?.name ?: "EMERGENCY")
    }

    override fun onCountdownCancelled() {
        Log.i(TAG, "Countdown cancelled. Returning to MONITORING.")
        stateMachine.transitionTo(SosState.CANCELLED, "User cancelled countdown")
        stateMachine.transitionTo(SosState.MONITORING, "Reset after cancellation")
    }

    fun cancelCountdown(reason: String = "User action") {
        if (stateMachine.currentState == SosState.COUNTDOWN) {
            countdownController.cancelCountdown()
        }
    }

    /**
     * Triggers manual SOS directly into ACTIVE_SOS lifecycle.
     */
    fun triggerManualSos() {
        Log.w(TAG, "Manual SOS triggered.")
        if (stateMachine.currentState == SosState.COUNTDOWN) {
            countdownController.cancelCountdown()
        }
        confirmActiveSos(emergencyType = "MANUAL_SOS")
    }

    private fun confirmActiveSos(emergencyType: String) {
        if (!stateMachine.transitionTo(SosState.ACTIVE_SOS, "SOS Confirmed ($emergencyType)")) {
            Log.e(TAG, "Failed to transition to ACTIVE_SOS from ${stateMachine.currentState}")
            return
        }

        val incidentId = "echo-inc-${UUID.randomUUID().toString().take(8)}"
        activeIncidentId = incidentId

        val sosEvent = SosEvent(
            incidentId = incidentId,
            userId = "echo-user-local",
            emergencyType = emergencyType,
            threatScore = threatEngine.currentThreatScore,
            timestamp = System.currentTimeMillis(),
            status = SosState.ACTIVE_SOS.name
        )

        // Start location tracking immediately
        locationController.startTracking(incidentId)

        // Execute backend dispatch asynchronously
        coroutineScope.launch {
            dispatchActiveSos(sosEvent)
        }
    }

    private suspend fun dispatchActiveSos(sosEvent: SosEvent) {
        val incidentId = sosEvent.incidentId
        Log.i(TAG, "Dispatching ACTIVE_SOS for incident $incidentId...")

        // 1. Create Incident Record
        val createResult = backendClient.createIncident(sosEvent)
        if (createResult.isFailure) {
            Log.w(TAG, "Backend createIncident failed (Offline/Error). Queuing...")
            isRemoteAvailable = false
            offlineQueue.enqueue(
                OfflineQueue.QueuedOperation(
                    id = UUID.randomUUID().toString(),
                    type = OfflineQueue.OperationType.CREATE_INCIDENT,
                    timestamp = System.currentTimeMillis(),
                    incidentId = incidentId,
                    sosEvent = sosEvent
                )
            )
            // Offline condition on ACTIVE_SOS -> Sound Siren
            localSirenController.updateSirenState(SosState.ACTIVE_SOS, isRemoteAvailable = false)
        } else {
            isRemoteAvailable = true
            localSirenController.updateSirenState(SosState.ACTIVE_SOS, isRemoteAvailable = true)
        }

        // 2. Trigger Twilio Call + SMS to Emergency Contacts
        val alertResult = backendClient.triggerTwilioAlert(incidentId)
        if (alertResult.isFailure) {
            Log.w(TAG, "Backend triggerTwilioAlert failed. Queuing...")
            offlineQueue.enqueue(
                OfflineQueue.QueuedOperation(
                    id = UUID.randomUUID().toString(),
                    type = OfflineQueue.OperationType.ALERT_CONTACTS,
                    timestamp = System.currentTimeMillis(),
                    incidentId = incidentId
                )
            )
        }

        // 3. Start Live Location Stream
        backendClient.startLiveLocation(incidentId)
    }

    private fun onLocationUpdated(update: LocationUpdate) {
        if (stateMachine.currentState != SosState.ACTIVE_SOS) return

        coroutineScope.launch {
            val result = backendClient.sendLocationUpdate(update)
            if (result.isFailure) {
                offlineQueue.enqueue(
                    OfflineQueue.QueuedOperation(
                        id = UUID.randomUUID().toString(),
                        type = OfflineQueue.OperationType.LOCATION_UPDATE,
                        timestamp = System.currentTimeMillis(),
                        incidentId = update.incidentId,
                        locationUpdate = update
                    )
                )
            }
        }
    }

    /**
     * Resolves the active SOS, stopping siren, location, and notifying backend.
     */
    fun resolveSos(): Boolean {
        if (stateMachine.currentState != SosState.ACTIVE_SOS) {
            Log.w(TAG, "Cannot resolve SOS: current state is ${stateMachine.currentState}")
            return false
        }

        val incidentId = activeIncidentId ?: "unknown"
        Log.i(TAG, "Resolving ACTIVE_SOS for incident $incidentId...")

        // 1. Transition to RESOLVED
        stateMachine.transitionTo(SosState.RESOLVED, "User resolved emergency")

        // 2. Stop Location and Siren
        locationController.stopTracking()
        localSirenController.stopSiren()

        // 3. Send Resolution Notification
        coroutineScope.launch {
            val resResult = backendClient.sendResolutionSms(incidentId)
            if (resResult.isFailure) {
                offlineQueue.enqueue(
                    OfflineQueue.QueuedOperation(
                        id = UUID.randomUUID().toString(),
                        type = OfflineQueue.OperationType.RESOLUTION,
                        timestamp = System.currentTimeMillis(),
                        incidentId = incidentId
                    )
                )
            }
        }

        // 4. Return to nominal MONITORING
        activeIncidentId = null
        stateMachine.transitionTo(SosState.MONITORING, "Reset after resolution")
        return true
    }

    /**
     * Simulates network connectivity change.
     */
    fun setNetworkConnectivity(available: Boolean) {
        this.isRemoteAvailable = available
        if (backendClient is MockBackendClient) {
            backendClient.isNetworkAvailable = available
        }
        localSirenController.updateSirenState(stateMachine.currentState, isRemoteAvailable = available)

        if (available && !offlineQueue.isEmpty()) {
            coroutineScope.launch {
                val drained = offlineQueue.drainQueue()
                Log.i(TAG, "Connectivity restored: Drained $drained offline operations.")
            }
        }
    }
}
