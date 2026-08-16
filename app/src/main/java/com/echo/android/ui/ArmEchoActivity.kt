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
import com.echo.android.ai.ModelStatus
import com.echo.android.ai.SensorEvidence
import com.echo.android.ai.SosState
import com.echo.android.ai.ThreatEngineInterface
import com.echo.android.ai.ThreatUpdate
import com.echo.android.audio.ForegroundAudioService
import com.echo.android.contacts.EmergencyContact
import com.echo.android.databinding.ActivityArmEchoBinding
import com.echo.android.permissions.PermissionsGateway
import com.echo.android.sos.CountdownController
import com.echo.android.sos.EchoSosOrchestrator
import com.echo.android.wake.DefaultVoiceCommandDetector
import com.echo.android.wake.DefaultWakeWordDetector
import com.echo.android.wake.VoiceCommandListener
import com.echo.android.wake.WakeWordDetector
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Main Activity for ECHO Prototype: Integrating Developer 1 Foundation, Developer 2 ML,
 * and Developer 3 Threat Engine & SOS Orchestrator.
 */
class ArmEchoActivity : AppCompatActivity(),
    ForegroundAudioService.ServiceStateListener,
    ThreatEngineInterface.ThreatUpdateListener,
    CountdownController.CountdownListener {

    companion object {
        private const val TAG = "ArmEchoActivity"
    }

    private lateinit var binding: ActivityArmEchoBinding
    private lateinit var permissionsGateway: PermissionsGateway

    private var audioService: ForegroundAudioService? = null
    private var isServiceBound = false
    private var isArmed = false

    lateinit var orchestrator: EchoSosOrchestrator
        private set

    private lateinit var wakeWordDetector: WakeWordDetector
    private lateinit var voiceCommandListener: VoiceCommandListener

    private val timeFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)
    private val logBuffer = StringBuilder()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val micGranted = permissions[android.Manifest.permission.RECORD_AUDIO] == true
        if (micGranted) {
            appendLog("[Permission] Microphone permission granted.")
            updatePermissionUi(true)
        } else {
            appendLog("[Warning] Microphone permission denied.")
            updatePermissionUi(false)
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
            orchestrator.arm()
            appendLog("[Service] ForegroundAudioService connected and armed.")
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            audioService?.setServiceStateListener(null)
            audioService = null
            isServiceBound = false
            isArmed = false
            orchestrator.disarm()
            updateArmUi(false)
            appendLog("[Service] Disconnected from ForegroundAudioService.")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityArmEchoBinding.inflate(layoutInflater)
        setContentView(binding.root)

        permissionsGateway = PermissionsGateway(this)
        orchestrator = EchoSosOrchestrator(this)
        wakeWordDetector = DefaultWakeWordDetector()
        voiceCommandListener = DefaultVoiceCommandDetector()

        // Initialize default demo contacts if none present (guarantees 2-5 contacts)
        if (orchestrator.contactManager.getContacts().isEmpty()) {
            orchestrator.contactManager.saveContacts(
                listOf(
                    EmergencyContact("c1", "Mom", "+15551234567"),
                    EmergencyContact("c2", "Primary Contact", "+15559876543")
                )
            )
        }

        setupViews()
        setupListeners()
        setupStateListeners()

        val hasMic = permissionsGateway.hasMicrophonePermission()
        updatePermissionUi(hasMic)
        if (!hasMic) {
            permissionsGateway.requestEssentialPermissions(permissionLauncher)
        }
    }

    private fun setupViews() {
        binding.tvLogConsole.movementMethod = ScrollingMovementMethod()
        updateArmUi(false)
        updateContactsUi()
    }

    private fun setupStateListeners() {
        orchestrator.threatEngine.addThreatListener(this)
        orchestrator.countdownController.setListener(this)

        orchestrator.stateMachine.addStateListener { previous, current ->
            runOnUiThread {
                updateSosStateUi(current)
                appendLog("[State] Transition: $previous -> $current")
            }
        }
    }

    private fun setupListeners() {
        binding.btnToggleArm.setOnClickListener {
            if (isArmed) disarmEcho() else armEcho()
        }

        binding.btnPermissions.setOnClickListener {
            permissionsGateway.requestEssentialPermissions(permissionLauncher)
        }

        binding.btnCancelSos.setOnClickListener {
            appendLog("[User] Cancel SOS button pressed.")
            orchestrator.cancelCountdown("Button cancel")
        }

        binding.btnResolveSos.setOnClickListener {
            appendLog("[User] Resolve SOS button pressed.")
            orchestrator.resolveSos()
        }

        binding.btnManualSos.setOnClickListener {
            appendLog("[User] Manual SOS triggered.")
            orchestrator.triggerManualSos()
        }

        // --- Demo Scenario Controls ---

        // Scenario 1: Loud Harmless (Fireworks -> Elevated threat score, but < 70 threshold -> No SOS)
        binding.btnScenario1Harmless.setOnClickListener {
            appendLog("[DEMO] Scenario 1: Loud Harmless (Fireworks). Expect: Score rises but < 70, No SOS.")
            val harmlessResult = ClassifierResult(
                timestampMs = System.currentTimeMillis(),
                normalProbability = 0.95f,
                accidentProbability = 0.03f,
                distressProbability = 0.01f,
                violentIncidentProbability = 0.01f,
                predictedClass = EmergencyClass.NORMAL,
                modelStatus = ModelStatus.OK
            )
            val noMotion = SensorEvidence(System.currentTimeMillis(), accelerationAnomaly = false, 0.1f, false, 0.1f)
            orchestrator.threatEngine.onClassifierResult(harmlessResult, noMotion)
        }

        // Scenario 2: Accident (Accident 88% + Impact Motion -> Score >= 70 -> Countdown -> Active SOS)
        binding.btnScenario2Accident.setOnClickListener {
            appendLog("[DEMO] Scenario 2: Vehicle Accident + Impact. Expect: Score >= 70 -> Countdown -> Active SOS.")
            val accidentResult = ClassifierResult(
                timestampMs = System.currentTimeMillis(),
                normalProbability = 0.05f,
                accidentProbability = 0.90f,
                distressProbability = 0.03f,
                violentIncidentProbability = 0.02f,
                predictedClass = EmergencyClass.ACCIDENT,
                modelStatus = ModelStatus.OK
            )
            val impactMotion = SensorEvidence(System.currentTimeMillis(), accelerationAnomaly = true, 0.85f, false, 0.3f)
            orchestrator.threatEngine.onClassifierResult(accidentResult, impactMotion)
        }

        // Scenario 3: Scream Alone (Distress + No Motion -> Score < 70 -> No SOS)
        binding.btnScenario3Scream.setOnClickListener {
            appendLog("[DEMO] Scenario 3: Scream Alone (No Motion). Expect: Score elevated but < 70 -> No SOS.")
            val screamResult = ClassifierResult(
                timestampMs = System.currentTimeMillis(),
                normalProbability = 0.20f,
                accidentProbability = 0.05f,
                distressProbability = 0.70f,
                violentIncidentProbability = 0.05f,
                predictedClass = EmergencyClass.DISTRESS,
                modelStatus = ModelStatus.OK
            )
            val noMotion = SensorEvidence(System.currentTimeMillis(), accelerationAnomaly = false, 0.0f, false, 0.0f)
            orchestrator.threatEngine.onClassifierResult(screamResult, noMotion)
        }

        // Scenario 4: Distress + Erratic Motion -> Score >= 70 -> Countdown -> Active SOS
        binding.btnScenario4DistressMotion.setOnClickListener {
            appendLog("[DEMO] Scenario 4: Distress + Erratic Motion. Expect: Score >= 70 -> Countdown -> Active SOS.")
            val distressResult = ClassifierResult(
                timestampMs = System.currentTimeMillis(),
                normalProbability = 0.02f,
                accidentProbability = 0.03f,
                distressProbability = 0.88f,
                violentIncidentProbability = 0.07f,
                predictedClass = EmergencyClass.DISTRESS,
                modelStatus = ModelStatus.OK
            )
            val erraticMotion = SensorEvidence(System.currentTimeMillis(), accelerationAnomaly = true, 0.75f, true, 0.80f)
            orchestrator.threatEngine.onClassifierResult(distressResult, erraticMotion)
        }

        // Scenario 5: Voice Cancel ("Hey Echo, cancel SOS")
        binding.btnSimulateVoiceCancel.setOnClickListener {
            appendLog("[DEMO] Voice Cancel: \"Hey Echo, cancel SOS\" received.")
            voiceCommandListener.simulateCancelCommand()
            orchestrator.cancelCountdown("Voice cancel command")
        }

        // Scenario 6: Network Toggle (Offline Simulation -> Siren Trigger during Active SOS)
        binding.btnToggleNetwork.setOnClickListener {
            val nextState = !orchestrator.isRemoteAvailable
            orchestrator.setNetworkConnectivity(nextState)
            val netStr = if (nextState) "ONLINE" else "OFFLINE"
            appendLog("[DEMO] Network connectivity set to: $netStr. Siren status: ${orchestrator.localSirenController.isSirenActive}")
            Toast.makeText(this, "Network: $netStr", Toast.LENGTH_SHORT).show()
        }

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

        appendLog("[User] Arm Echo initiated.")
        val serviceIntent = Intent(this, ForegroundAudioService::class.java).apply {
            action = ForegroundAudioService.ACTION_START_ARMED
        }
        ContextCompat.startForegroundService(this, serviceIntent)
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    private fun disarmEcho() {
        appendLog("[User] Disarm Echo initiated.")
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
        orchestrator.disarm()
        updateArmUi(false)
    }

    private fun updateSosStateUi(state: SosState) {
        when (state) {
            SosState.MONITORING -> {
                binding.llCountdownCard.visibility = View.GONE
                binding.llActiveSosCard.visibility = View.GONE
                updateArmUi(isArmed)
            }
            SosState.COUNTDOWN -> {
                binding.llCountdownCard.visibility = View.VISIBLE
                binding.llActiveSosCard.visibility = View.GONE
                binding.tvStatusValue.text = getString(R.string.status_countdown)
                binding.tvStatusValue.setTextColor(ContextCompat.getColor(this, R.color.status_emergency))
            }
            SosState.ACTIVE_SOS -> {
                binding.llCountdownCard.visibility = View.GONE
                binding.llActiveSosCard.visibility = View.VISIBLE
                binding.tvStatusValue.text = getString(R.string.status_sos)
                binding.tvStatusValue.setTextColor(ContextCompat.getColor(this, R.color.status_emergency))
                binding.tvIncidentId.text = "Incident: ${orchestrator.activeIncidentId}"
                val sirenTxt = if (orchestrator.localSirenController.isSirenActive) "SIREN: ACTIVE (Offline)" else "Siren: Inactive (Online)"
                binding.tvSirenStatus.text = sirenTxt
            }
            SosState.RESOLVED, SosState.CANCELLED -> {
                binding.llCountdownCard.visibility = View.GONE
                binding.llActiveSosCard.visibility = View.GONE
                binding.tvStatusValue.text = if (state == SosState.RESOLVED) getString(R.string.status_resolved) else "CANCELLED"
            }
            SosState.DEGRADED -> {
                binding.tvStatusValue.text = getString(R.string.status_degraded)
            }
        }
    }

    private fun updateArmUi(armed: Boolean) {
        if (armed) {
            binding.tvStatusValue.text = getString(R.string.status_armed)
            binding.tvStatusValue.setTextColor(ContextCompat.getColor(this, R.color.status_armed))
            binding.tvStatusDetail.text = "Microphone + Accelerometer + Gyroscope Active"
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
            binding.pbAudioRms.progress = 0
            binding.tvRmsValue.text = "0.00 (-∞ dBFS)"
            binding.pbThreatScore.progress = 0
            binding.tvThreatScoreValue.text = "0 / 100"
            binding.tvClassValue.text = "NORMAL (Standby)"
        }
    }

    private fun updateContactsUi() {
        val count = orchestrator.contactManager.getContacts().size
        binding.tvContactsDetail.text = "$count contacts configured (2–5 active)"
    }

    private fun updatePermissionUi(hasMic: Boolean) {
        binding.btnPermissions.visibility = if (hasMic) View.GONE else View.VISIBLE
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
        orchestrator.onAcousticInference(result)
        runOnUiThread {
            val prob = when (result.predictedClass) {
                EmergencyClass.NORMAL -> result.normalProbability
                EmergencyClass.ACCIDENT -> result.accidentProbability
                EmergencyClass.DISTRESS -> result.distressProbability
                EmergencyClass.VIOLENT_INCIDENT -> result.violentIncidentProbability
            }
            binding.tvClassValue.text = "${result.predictedClass} (${(prob * 100).toInt()}%)"
            val color = when (result.predictedClass) {
                EmergencyClass.NORMAL -> ContextCompat.getColor(this, R.color.status_armed)
                EmergencyClass.ACCIDENT -> ContextCompat.getColor(this, R.color.status_warning)
                else -> ContextCompat.getColor(this, R.color.status_emergency)
            }
            binding.tvClassValue.setTextColor(color)
        }
    }

    override fun onMonitoringStateChanged(isArmed: Boolean, status: ModelStatus) {
        runOnUiThread {
            this.isArmed = isArmed
            updateArmUi(isArmed)
            binding.tvAiStatusValue.text = "$status (Sensors Active)"
        }
    }

    // --- ThreatEngine Callbacks ---

    override fun onThreatUpdate(update: ThreatUpdate) {
        runOnUiThread {
            binding.pbThreatScore.progress = update.threatScore
            binding.tvThreatScoreValue.text = "${update.threatScore} / 100"
            val color = when {
                update.threatScore >= orchestrator.threatEngine.config.sosThreshold -> ContextCompat.getColor(this, R.color.status_emergency)
                update.threatScore > 35 -> ContextCompat.getColor(this, R.color.status_warning)
                else -> ContextCompat.getColor(this, R.color.text_primary)
            }
            binding.tvThreatScoreValue.setTextColor(color)
            appendLog("[ThreatEngine] Score=${update.threatScore}/100 Status='${update.evidenceStatus}'")
        }
    }

    // --- Countdown Callbacks ---

    override fun onTick(secondsRemaining: Int) {
        runOnUiThread {
            binding.tvCountdownSeconds.text = "$secondsRemaining"
        }
    }

    override fun onCountdownFinished() {
        // Handled in orchestrator
    }

    override fun onCountdownCancelled() {
        // Handled in orchestrator
    }

    private fun appendLog(message: String) {
        val timestamp = timeFormat.format(Date())
        val line = "[$timestamp] $message\n"
        logBuffer.append(line)
        if (logBuffer.length > 5000) {
            val startIdx = logBuffer.indexOf("\n", 1000)
            if (startIdx != -1) logBuffer.delete(0, startIdx + 1)
        }
        binding.tvLogConsole.text = logBuffer.toString()
    }

    override fun onDestroy() {
        super.onDestroy()
        orchestrator.disarm()
        if (isServiceBound) {
            unbindService(serviceConnection)
            isServiceBound = false
        }
    }
}
