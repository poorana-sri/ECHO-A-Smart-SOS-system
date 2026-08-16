package com.echo.android.contacts

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONArray
import org.json.JSONObject

/**
 * Manages persistence and validation of emergency contacts (2 to 5 contacts required).
 */
class EmergencyContactManager(context: Context) {

    companion object {
        private const val PREFS_NAME = "echo_emergency_contacts"
        private const val KEY_CONTACTS_JSON = "contacts_json"
        const val MIN_CONTACTS = 2
        const val MAX_CONTACTS = 5
    }

    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /**
     * Loads the list of saved emergency contacts.
     */
    fun getContacts(): List<EmergencyContact> {
        val jsonStr = prefs.getString(KEY_CONTACTS_JSON, null) ?: return emptyList()
        val list = mutableListOf<EmergencyContact>()
        try {
            val jsonArray = JSONArray(jsonStr)
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val id = obj.getString("id")
                val name = obj.getString("name")
                val phone = obj.getString("phone")
                list.add(EmergencyContact(id, name, phone))
            }
        } catch (e: Exception) {
            // Return whatever valid parsed so far
        }
        return list
    }

    /**
     * Saves the list of emergency contacts, enforcing 2 to 5 contacts constraint.
     */
    fun saveContacts(contacts: List<EmergencyContact>): Result<Unit> {
        if (contacts.size < MIN_CONTACTS) {
            return Result.failure(IllegalArgumentException("Minimum $MIN_CONTACTS emergency contacts required (provided ${contacts.size})."))
        }
        if (contacts.size > MAX_CONTACTS) {
            return Result.failure(IllegalArgumentException("Maximum $MAX_CONTACTS emergency contacts allowed (provided ${contacts.size})."))
        }

        return try {
            val jsonArray = JSONArray()
            for (c in contacts) {
                val obj = JSONObject().apply {
                    put("id", c.id)
                    put("name", c.name)
                    put("phone", c.phoneNumber)
                }
                jsonArray.put(obj)
            }
            prefs.edit().putString(KEY_CONTACTS_JSON, jsonArray.toString()).apply()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Adds a contact if within max limit.
     */
    fun addContact(contact: EmergencyContact): Result<Unit> {
        val current = getContacts().toMutableList()
        if (current.size >= MAX_CONTACTS) {
            return Result.failure(IllegalStateException("Cannot exceed maximum of $MAX_CONTACTS contacts."))
        }
        current.removeAll { it.id == contact.id || it.phoneNumber == contact.phoneNumber }
        current.add(contact)
        return if (current.size >= MIN_CONTACTS) {
            saveContacts(current)
        } else {
            // Save draft if < MIN_CONTACTS without throwing error
            saveDraft(current)
        }
    }

    /**
     * Removes a contact by ID.
     */
    fun removeContact(contactId: String): Result<Unit> {
        val current = getContacts().toMutableList()
        current.removeAll { it.id == contactId }
        return saveDraft(current)
    }

    private fun saveDraft(contacts: List<EmergencyContact>): Result<Unit> {
        return try {
            val jsonArray = JSONArray()
            for (c in contacts) {
                val obj = JSONObject().apply {
                    put("id", c.id)
                    put("name", c.name)
                    put("phone", c.phoneNumber)
                }
                jsonArray.put(obj)
            }
            prefs.edit().putString(KEY_CONTACTS_JSON, jsonArray.toString()).apply()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Returns true if there are between 2 and 5 valid configured contacts.
     */
    fun hasValidEmergencyContacts(): Boolean {
        val count = getContacts().size
        return count in MIN_CONTACTS..MAX_CONTACTS
    }

    /**
     * Clears all contacts (useful for tests).
     */
    fun clear() {
        prefs.edit().clear().apply()
    }
}
