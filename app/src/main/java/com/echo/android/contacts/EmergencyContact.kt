package com.echo.android.contacts

/**
 * Represents an emergency contact configured for ECHO automated alerts.
 */
data class EmergencyContact(
    val id: String,
    val name: String,
    val phoneNumber: String
) {
    init {
        require(name.isNotBlank()) { "Contact name cannot be blank." }
        require(phoneNumber.isNotBlank() && phoneNumber.replace(Regex("[^0-9+]"), "").length >= 7) {
            "Contact phone number must contain at least 7 digits."
        }
    }
}
