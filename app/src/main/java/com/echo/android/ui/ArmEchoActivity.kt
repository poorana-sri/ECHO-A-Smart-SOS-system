package com.echo.android.ui

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.res.ColorStateList
import android.os.Bundle
import android.os.IBinder
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.echo.android.R
import com.echo.android.ai.AudioEvent
import com.echo.android.ai.ClassifierResult
import com.echo.android.ai.EmergencyClass
import com.echo.android.ai.MockThreatEngine
import com.echo.android.ai.ModelStatus
import com.echo.android.ai.ThreatEngineInterface
import com.echo.android.ai.ThreatUpdate
import com.echo.android.audio.ForegroundAudioService
import com.echo.android.databinding.ActivityArmEchoBinding
import com.echo.android.permissions.PermissionsGateway
import com.echo.android.wake.DefaultVoiceCommandDetector
import com.echo.android.wake.DefaultWakeWordDetector
import com.echo.android.wake.VoiceCommandListener
import com.echo.android.wake.WakeWordDetector
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Main Activity for ECHO Prototype: Developer 1 Foundation & AI Runtime.
 *
 * Provides:
 * - Explicit user control to Arm/Disarm Echo.
 * - Start/Stop of microphone Foreground Service.
 * - Live visualization of volatile RAM buffer RMS energy.
 * - Live AI classification probabilities and Mock Threat Score.
 * - Interactive simulation controls for "Hey Echo", voice cancel, and emergency handoffs.
 *
 * NOTE: Hackathon prototype implementation.
 */
class ArmEchoActivity : AppCompatActivity(),
    ForegroundAudioService.ServiceStateListener,
    ThreatEngineInterface.ThreatUpdateListener {

    companion object {
        private const val TAG = "ArmEchoActivity"
        private const val MAX_LOG_LINES = 100
    }

    private lateinit var binding: ActivityArmEchoBinding
    private lateinit var permissionsGateway: PermissionsGateway

    private var audioService: ForegroundAudioService? = null
    private var isServiceBound = false
    private var isArmed = false

    // Local wake word detector and voice command listener for prototype controls
    private lateinit var wakeWordDetector: WakeWordDetector
    private lateinit var voiceCommandListener: VoiceCommandListener
    private val threatEngine: ThreatEngineInterface = MockThreatEngine.instance

    private val timeFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)
    private val logBuffer = StringBuilder()

    // Permission request launcher
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val micGranted = permissions[android.Manifest.permission.RECORD_AUDIO] == true
        if (micGranted) {
            appendLog("[Permission] Microphone permission granted.")
            updatePermissionUi(true)
        } else {
            appendLog("[Warning] Microphone permission denied. Acoustic detection degraded.")
            updatePermissionUi(false)
            Toast.makeText(this, "Microphone permission is required for acoustic monitoring.", Toast.LENGTH_LONG).show()
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as ForegroundAudioService.LocalBinder
            audioService = binder.getService()
            audioService?.setServiceStateListener(this@ArmEchoActivity)
            isServiceBound = true
            isArmed = true
            updateArmUi(true)
            appendLog("[Service] Connected to ForegroundAudioService (mic monitoring active).")
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            audioService?.setServiceStateListener(null)
            audioService = null
            isServiceBound = false
            isArmed = false
            updateArmUi(false)
            appendLog("[Service] Disconnected from ForegroundAudioService.")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityArmEchoBinding.inflate(layoutInflater)
        setContentView(binding.root)

        permissionsGateway = PermissionsGateway(this)
        wakeWordDetector = DefaultWakeWordDetector()
        voiceCommandListener = DefaultVoiceCommandDetector()

        setupViews()
        setupListeners()
        threatEngine.addThreatListener(this)

        // Check initial permissions
        val hasMic = permissionsGateway.hasMicrophonePermission()
        updatePermissionUi(hasMic)
        if (!hasMic) {
            permissionsGateway.requestEssentialPermissions(permissionLauncher)
        }
    }

    private fun setupViews() {
        binding.tvLogConsole.movementMethod = ScrollingMovementMethod()
        updateArmUi(false)
    }

    private fun setupListeners() {
        // Arm / Disarm toggle button
        binding.btnToggleArm.setOnClickListener {
            if (isArmed) {
                disarmEcho()
            } else {
                armEcho()
            }
        }

        // Permission retry button
        binding.btnPermissions.setOnClickListener {
            permissionsGateway.requestEssentialPermissions(permissionLauncher)
        }

        // Simulate "Hey Echo" wake word
        binding.btnSimulateWakeWord.setOnClickListener {
            appendLog("[DEMO / MOCK] User triggered \"Hey Echo\" voice wake simulation.")
            if (!isArmed) {
                armEcho()
            } else {
                Toast.makeText(this, "Echo is already armed and monitoring.", Toast.LENGTH_SHORT).show()
            }
        }

        // Simulate "Hey Echo, cancel SOS" voice cancellation
        binding.btnSimulateVoiceCancel.setOnClickListener {
            appendLog("[DEMO / MOCK] Voice cancel triggered: \"Hey Echo, cancel SOS\".")
            voiceCommandListener.simulateCancelCommand()
            Toast.makeText(this, "Voice cancel event emitted to VoiceCommandListener.", Toast.LENGTH_SHORT).show()
        }

        // Simulate ACCIDENT acoustic event handoff
        binding.btnSimulateAccident.setOnClickListener {
            appendLog("[DEMO / MOCK] Injecting synthetic ACCIDENT classification...")
            val simulatedResult = ClassifierResult(
                timestampMs = System.currentTimeMillis(),
                normalProbability = 0.05f,
                accidentProbability = 0.88f,
                distressProbability = 0.05f,
                violentIncidentProbability = 0.02f,
                predictedClass = EmergencyClass.ACCIDENT,
                modelStatus = ModelStatus.DEGRADED // Correctly marked as mock/degraded
            )
            threatEngine.onClassifierResult(simulatedResult)
        }

        // Clear console logs
        binding.btnClearLogs.setOnClickListener {
            logBuffer.clear()
            binding.tvLogConsole.text = ""
        }
    }

    private fun armEcho() {
        if (!permissionsGateway.hasMicrophonePermission()) {
            permissionsGateway.requestEssentialPermissions(permissionLauncher)
            return
        }

        appendLog("[User] Arm Echo action initiated.")
        val serviceIntent = Intent(this, ForegroundAudioService::class.java).apply {
            action = ForegroundAudioService.ACTION_START_ARMED
        }

        // Start Foreground Service explicitly
        ContextCompat.startForegroundService(this, serviceIntent)
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    private fun disarmEcho() {
        appendLog("[User] Disarm Echo action initiated.")
        if (isServiceBound) {
            audioService?.stopMonitoring()
            unbindService(serviceConnection)
            isServiceBound = false
        }

        val serviceIntent = Intent(this, ForegroundAudioService::class.java).apply {
            action = ForegroundAudioService.ACTION_STOP_DISARMED
        }
        stopService(serviceIntent)

        isArmed = false
        updateArmUi(false)
    }

    private fun updateArmUi(armed: Boolean) {
        if (armed) {
            binding.tvStatusValue.text = getString(R.string.status_armed)
            binding.tvStatusValue.setTextColor(ContextCompat.getColor(this, R.color.status_armed))
            binding.tvStatusDetail.text = "Microphone Foreground Service active. 3s rolling RAM buffer."

            binding.btnToggleArm.text = getString(R.string.btn_disarm)
            binding.btnToggleArm.backgroundTintList = ColorStateList.valueOf(
                ContextCompat.getColor(this, R.color.btn_disarm_bg)
            )
            binding.ivShieldIcon.imageTintList = ColorStateList.valueOf(
                ContextCompat.getColor(this, R.color.status_armed)
            )
        } else {
            binding.tvStatusValue.text = getString(R.string.status_disarmed)
            binding.tvStatusValue.setTextColor(ContextCompat.getColor(this, R.color.status_disarmed))
            binding.tvStatusDetail.text = "Tap below or say \"Hey Echo\" when active"

            binding.btnToggleArm.text = getString(R.string.btn_arm)
            binding.btnToggleArm.backgroundTintList = ColorStateList.valueOf(
                ContextCompat.getColor(this, R.color.btn_arm_bg)
            )
            binding.ivShieldIcon.imageTintList = ColorStateList.valueOf(
                ContextCompat.getColor(this, R.color.status_disarmed)
            )

            // Reset meters
            binding.pbAudioRms.progress = 0
            binding.tvRmsValue.text = "0.00 (-∞ dBFS)"
            binding.pbThreatScore.progress = 0
            binding.tvThreatScoreValue.text = "0 / 100"
            binding.tvClassValue.text = "NORMAL (Standby)"
        }
    }

    private fun updatePermissionUi(hasMic: Boolean) {
        if (hasMic) {
            binding.btnPermissions.visibility = View.GONE
        } else {
            binding.btnPermissions.visibility = View.VISIBLE
            binding.tvStatusValue.text = getString(R.string.status_degraded)
            binding.tvStatusValue.setTextColor(ContextCompat.getColor(this, R.color.status_degraded))
        }
    }

    // --- ForegroundAudioService Callbacks ---

    override fun onRmsLevelUpdated(rms: Float, dBFS: Float) {
        runOnUiThread {
            val progress = (rms * 150).coerceIn(0f, 100f).toInt()
            binding.pbAudioRms.progress = progress
            binding.tvRmsValue.text = String.format(Locale.US, "%.3f (%.1f dBFS)", rms, dBFS)
        }
    }

    override fun onAudioEventReceived(event: AudioEvent) {
        runOnUiThread {
            val topPred = event.yamnetPredictions.firstOrNull()
            val predText = if (topPred != null) "${topPred.first} (${(topPred.second * 100).toInt()}%)" else "None"
            binding.tvAiStatusValue.text = "${event.modelStatus} (YAMNet: $predText)"
        }
    }

    override fun onClassifierResultReceived(result: ClassifierResult) {
        runOnUiThread {
            val prob = when (result.predictedClass) {
                EmergencyClass.NORMAL -> result.normalProbability
                EmergencyClass.ACCIDENT -> result.accidentProbability
                EmergencyClass.DISTRESS -> result.distressProbability
                EmergencyClass.VIOLENT_INCIDENT -> result.violentIncidentProbability
            }
            val percent = (prob * 100).toInt()
            binding.tvClassValue.text = "${result.predictedClass} ($percent%)"

            val color = when (result.predictedClass) {
                EmergencyClass.NORMAL -> ContextCompat.getColor(this, R.color.status_armed)
                EmergencyClass.ACCIDENT -> ContextCompat.getColor(this, R.color.status_warning)
                EmergencyClass.DISTRESS, EmergencyClass.VIOLENT_INCIDENT -> ContextCompat.getColor(this, R.color.status_emergency)
            }
            binding.tvClassValue.setTextColor(color)
        }
    }

    override fun onMonitoringStateChanged(isArmed: Boolean, status: ModelStatus) {
        runOnUiThread {
            this.isArmed = isArmed
            updateArmUi(isArmed)
            binding.tvAiStatusValue.text = "$status"
        }
    }

    // --- ThreatEngineInterface Callbacks ---

    override fun onThreatUpdate(update: ThreatUpdate) {
        runOnUiThread {
            binding.pbThreatScore.progress = update.threatScore
            binding.tvThreatScoreValue.text = "${update.threatScore} / 100"

            val scoreColor = when {
                update.threatScore >= MockThreatEngine.SOS_THRESHOLD -> ContextCompat.getColor(this, R.color.status_emergency)
                update.threatScore > 35 -> ContextCompat.getColor(this, R.color.status_warning)
                else -> ContextCompat.getColor(this, R.color.text_primary)
            }
            binding.tvThreatScoreValue.setTextColor(scoreColor)

            appendLog("[ThreatEngine] Score=${update.threatScore}/100 Class=${update.acousticEvidence.predictedClass} Status='${update.evidenceStatus}'")
        }
    }

    private fun appendLog(message: String) {
        val timestamp = timeFormat.format(Date())
        val line = "[$timestamp] $message\n"
        logBuffer.append(line)

        // Limit buffer length
        if (logBuffer.length > 5000) {
            val startIdx = logBuffer.indexOf("\n", 1000)
            if (startIdx != -1) {
                logBuffer.delete(0, startIdx + 1)
            }
        }

        binding.tvLogConsole.text = logBuffer.toString()

        // Auto-scroll to bottom
        binding.tvLogConsole.post {
            val scrollAmount = binding.tvLogConsole.layout?.let {
                it.getLineTop(binding.tvLogConsole.lineCount) - binding.tvLogConsole.height
            } ?: 0
            if (scrollAmount > 0) {
                binding.tvLogConsole.scrollTo(0, scrollAmount)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        threatEngine.removeThreatListener(this)
        if (isServiceBound) {
            unbindService(serviceConnection)
            isServiceBound = false
        }
    }
}
