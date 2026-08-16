package com.echo.android.sos

import com.echo.android.ai.SosState
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class OfflineSirenTest {

    private lateinit var sirenController: LocalSirenController

    @Before
    fun setup() {
        sirenController = LocalSirenController(null) // null context runs gracefully
    }

    @Test
    fun testPreConfirmationNetworkFailureDoesNotSoundSiren() {
        // Network failure during MONITORING
        sirenController.updateSirenState(SosState.MONITORING, isRemoteAvailable = false)
        assertFalse(sirenController.isSirenActive)

        // Network failure during COUNTDOWN
        sirenController.updateSirenState(SosState.COUNTDOWN, isRemoteAvailable = false)
        assertFalse(sirenController.isSirenActive)

        // Network failure during CANCELLED
        sirenController.updateSirenState(SosState.CANCELLED, isRemoteAvailable = false)
        assertFalse(sirenController.isSirenActive)

        // Network failure during DEGRADED
        sirenController.updateSirenState(SosState.DEGRADED, isRemoteAvailable = false)
        assertFalse(sirenController.isSirenActive)
    }

    @Test
    fun testConfirmedActiveSosWithOfflineSoundsSiren() {
        // ACTIVE_SOS + Online -> No siren
        sirenController.updateSirenState(SosState.ACTIVE_SOS, isRemoteAvailable = true)
        assertFalse(sirenController.isSirenActive)

        // ACTIVE_SOS + Offline -> Sounds Siren!
        sirenController.updateSirenState(SosState.ACTIVE_SOS, isRemoteAvailable = false)
        assertTrue(sirenController.isSirenActive)

        // Connectivity restored -> Siren stops
        sirenController.updateSirenState(SosState.ACTIVE_SOS, isRemoteAvailable = true)
        assertFalse(sirenController.isSirenActive)
    }

    @Test
    fun testResolutionStopsSiren() {
        sirenController.updateSirenState(SosState.ACTIVE_SOS, isRemoteAvailable = false)
        assertTrue(sirenController.isSirenActive)

        // Resolution stops siren even if still offline
        sirenController.updateSirenState(SosState.RESOLVED, isRemoteAvailable = false)
        assertFalse(sirenController.isSirenActive)
    }
}
