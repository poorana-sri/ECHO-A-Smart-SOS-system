package com.echo.android.wake

/**
 * Interface for listening to manual or voice activation events across the ECHO app.
 *
 * Developer 3's SOS state machine can subscribe to these events to synchronize monitoring state.
 */
interface ActivationListener {

    /**
     * Triggered when Echo is armed (either via manual UI button or "Hey Echo" wake word).
     *
     * @param source Description of the trigger source ("MANUAL" or "VOICE_WAKE").
     */
    fun onEchoArmed(source: String)

    /**
     * Triggered when Echo is disarmed.
     *
     * @param source Description of the disarm source ("MANUAL").
     */
    fun onEchoDisarmed(source: String)
}
