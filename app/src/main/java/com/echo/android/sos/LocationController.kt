package com.echo.android.sos

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.util.Log
import com.echo.android.ai.LocationSource
import com.echo.android.ai.LocationUpdate

/**
 * Manages device GPS and network location tracking during ACTIVE_SOS.
 * Handles missing permissions, unavailable GPS, and caches last known locations.
 */
class LocationController(
    private val context: Context,
    private val locationManager: LocationManager? = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
) : LocationListener {

    companion object {
        private const val TAG = "LocationController"
        private const val MIN_TIME_MS = 3000L
        private const val MIN_DISTANCE_M = 2.0f
    }

    @Volatile
    private var isTracking: Boolean = false

    @Volatile
    var lastLocation: Location? = null
        private set

    @Volatile
    var activeIncidentId: String? = null

    private var onLocationChangedListener: ((LocationUpdate) -> Unit)? = null

    fun setLocationListener(listener: ((LocationUpdate) -> Unit)?) {
        this.onLocationChangedListener = listener
    }

    @SuppressLint("MissingPermission")
    fun startTracking(incidentId: String) {
        this.activeIncidentId = incidentId
        if (isTracking || locationManager == null) return

        try {
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    MIN_TIME_MS,
                    MIN_DISTANCE_M,
                    this
                )
            }
            if (locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.NETWORK_PROVIDER,
                    MIN_TIME_MS,
                    MIN_DISTANCE_M,
                    this
                )
            }

            // Obtain immediate last known location if present
            val lastGps = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            val lastNet = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            val best = lastGps ?: lastNet
            if (best != null) {
                lastLocation = best
            }

            isTracking = true
            Log.i(TAG, "Location tracking started for incident: $incidentId")
        } catch (e: SecurityException) {
            Log.w(TAG, "Location permission missing: ${e.message}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start location updates: ${e.message}")
        }
    }

    fun stopTracking() {
        if (!isTracking) return
        try {
            locationManager?.removeUpdates(this)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to stop location updates: ${e.message}")
        }
        isTracking = false
        Log.i(TAG, "Location tracking stopped for incident: $activeIncidentId")
    }

    fun getCurrentLocationUpdate(): LocationUpdate {
        val incId = activeIncidentId ?: "unknown_incident"
        val loc = lastLocation
        return if (loc != null) {
            val isStale = (System.currentTimeMillis() - loc.time) > 60_000L
            LocationUpdate(
                incidentId = incId,
                latitude = loc.latitude,
                longitude = loc.longitude,
                timestamp = loc.time,
                accuracy = loc.accuracy,
                source = if (isStale) LocationSource.LAST_KNOWN else LocationSource.LIVE
            )
        } else {
            LocationUpdate(
                incidentId = incId,
                latitude = 0.0,
                longitude = 0.0,
                timestamp = System.currentTimeMillis(),
                accuracy = 0.0f,
                source = LocationSource.UNAVAILABLE
            )
        }
    }

    override fun onLocationChanged(location: Location) {
        lastLocation = location
        val update = getCurrentLocationUpdate()
        Log.d(TAG, "Location updated: lat=${location.latitude}, lng=${location.longitude}, acc=${location.accuracy}")
        onLocationChangedListener?.invoke(update)
    }

    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}
}
