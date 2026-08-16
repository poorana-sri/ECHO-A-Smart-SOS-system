package com.echo.android.ai

/**
 * Outbound interface contract between Developer 1's AI Runtime / Audio pipeline
 * and Developer 3's Threat Score Engine & SOS State Machine.
 *
 * Developer 1 provides a [MockThreatEngine] during standalone testing.
 * Developer 3 will provide the real implementation fusing sensor evidence and executing the SOS state machine.
 */
interface ThreatEngineInterface {

    /**
     * Called whenever the AI sequence classifier produces a new inference result.
     *
     * @param result The latest acoustic classification probabilities and predicted class.
     * @param sensorEvidence Optional sensor evidence if available at inference time.
     */
    fun onClassifierResult(result: ClassifierResult, sensorEvidence: SensorEvidence? = null)

    /**
     * Called when YAMNet produces an acoustic event detection frame.
     *
     * @param event The intermediate YAMNet predictions and model health status.
     */
    fun onAudioEvent(event: AudioEvent)

    /**
     * Called when the user or wake layer arms or disarms acoustic monitoring.
     *
     * @param isArmed True if monitoring is active; false if disarmed.
     */
    fun onMonitoringStateChanged(isArmed: Boolean)

    /**
     * Direct manual SOS trigger pathway (e.g. from UI panic button or fallback during mic failure).
     */
    fun triggerManualSos()

    /**
     * Registers a listener for threat score updates and state transitions.
     */
    fun addThreatListener(listener: ThreatUpdateListener)

    /**
     * Unregisters a listener for threat score updates.
     */
    fun removeThreatListener(listener: ThreatUpdateListener)

    /**
     * Listener interface to observe threat score calculation and state changes.
     */
    fun interface ThreatUpdateListener {
        fun onThreatUpdate(update: ThreatUpdate)
    }
}
