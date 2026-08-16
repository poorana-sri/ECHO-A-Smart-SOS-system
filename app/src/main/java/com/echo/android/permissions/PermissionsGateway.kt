package com.echo.android.permissions

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.result.ActivityResultLauncher
import androidx.core.content.ContextCompat

/**
 * Gateway and abstraction for Android runtime permissions required by ECHO.
 *
 * Responsibilities:
 * - Checks & requests Microphone (`RECORD_AUDIO`) and Notifications (`POST_NOTIFICATIONS`).
 * - Provides stub/gateway checks for Location (`ACCESS_FINE_LOCATION`) used by Developers 3 & 4.
 */
class PermissionsGateway(private val context: Context) {

    /**
     * Checks if microphone capture permission is granted.
     */
    fun hasMicrophonePermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * Checks if notification permission is granted (Android 13+).
     */
    fun hasNotificationPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
        } else {
            true // Granted automatically on Android 12 and below
        }
    }

    /**
     * Checks if fine location permission is granted (used by Dev 3/4 location pipeline).
     */
    fun hasLocationPermission(): Boolean {
        val fineLocation = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED

        val coarseLocation = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_COARSE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED

        return fineLocation || coarseLocation
    }

    /**
     * Returns list of missing essential permissions required for Developer 1's audio monitoring.
     */
    fun getMissingEssentialPermissions(): List<String> {
        val missing = mutableListOf<String>()

        if (!hasMicrophonePermission()) {
            missing.add(Manifest.permission.RECORD_AUDIO)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && !hasNotificationPermission()) {
            missing.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        return missing
    }

    /**
     * Helper to launch standard permission request.
     */
    fun requestEssentialPermissions(launcher: ActivityResultLauncher<Array<String>>) {
        val missing = getMissingEssentialPermissions()
        if (missing.isNotEmpty()) {
            launcher.launch(missing.toTypedArray())
        }
    }
}
