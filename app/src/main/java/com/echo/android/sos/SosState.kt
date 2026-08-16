package com.echo.android.sos

/**
 * Primary lifecycle states for the ECHO SOS State Machine.
 */
enum class SosState {
    /** Normal acoustic and sensor monitoring active. */
    MONITORING,

    /** 10-second warning countdown with vibration and cancellation options. */
    COUNTDOWN,

    /** Temporary state when countdown was explicitly cancelled by the user. */
    CANCELLED,

    /** SOS confirmed: contacts alerted, live location sharing, encrypted audio preserved. */
    ACTIVE_SOS,

    /** User has resolved the active emergency. */
    RESOLVED,

    /** Critical sensor, microphone, or AI degradation. */
    DEGRADED
}
