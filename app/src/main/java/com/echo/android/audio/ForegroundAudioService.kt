package com.echo.android.audio

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import com.echo.android.R
import com.echo.android.ai.AudioEvent
import com.echo.android.ai.ClassifierResult
import com.echo.android.ai.MockThreatEngine
import com.echo.android.ai.ModelStatus
import com.echo.android.ai.SequenceClassifierRunner
import com.echo.android.ai.ThreatEngineInterface
import com.echo.android.ai.YamnetRunner
import com.echo.android.ui.ArmEchoActivity
import com.echo.android.wake.WakeWordDetector
import com.echo.android.wake.DefaultWakeWordDetector
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Android Microphone Foreground Service for continuous privacy-first acoustic monitoring.
 *
 * Requirements & Behavior:
 * - Started ONLY via explicit user action while app is visible.
 * - Shows persistent notification with MICROPHONE foreground service type (Android 14+).
 * - Continuously feeds 16kHz PCM audio into volatile 3-second [RollingAudioBuffer].
 * - Feeds wake detector and runs periodic AI inference via [YamnetRunner] and [SequenceClassifierRunner].
 * - Dispatches results to [ThreatEngineInterface].
 * - Handles mic errors gracefully by degrading status instead of crashing.
 *
 * NOTE: Hackathon prototype implementation.
 */
class ForegroundAudioService : Service() {

    companion object {
        private const val TAG = "ForegroundAudioService"
        const val NOTIFICATION_ID = 1001
        const val CHANNEL_ID = "echo_monitoring_channel"

        const val ACTION_START_ARMED = "com.echo.android.ACTION_START_ARMED"
        const val ACTION_STOP_DISARMED = "com.echo.android.ACTION_STOP_DISARMED"

        const val SAMPLE_RATE_HZ = 16000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        const val INFERENCE_INTERVAL_MS = 750L // AI inference hop rate
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var captureJob: Job? = null
    private var inferenceJob: Job? = null

    private val isMonitoring = AtomicBoolean(false)
    private val binder = LocalBinder()

    // Core components
    val rollingBuffer = RollingAudioBuffer(SAMPLE_RATE_HZ, 3)
    val featureExtractor: FeatureExtractor = DefaultFeatureExtractor()

    private lateinit var yamnetRunner: YamnetRunner
    private lateinit var sequenceClassifierRunner: SequenceClassifierRunner
    private lateinit var wakeWordDetector: WakeWordDetector
    var threatEngine: ThreatEngineInterface = MockThreatEngine.instance

    // AudioRecord instance
    private var audioRecord: AudioRecord? = null
    private var minBufferSize: Int = 0

    // Observable listeners
    private var serviceStateListener: ServiceStateListener? = null

    interface ServiceStateListener {
        fun onRmsLevelUpdated(rms: Float, dBFS: Float)
        fun onAudioEventReceived(event: AudioEvent)
        fun onClassifierResultReceived(result: ClassifierResult)
        fun onMonitoringStateChanged(isArmed: Boolean, status: ModelStatus)
    }

    inner class LocalBinder : Binder() {
        fun getService(): ForegroundAudioService = this@ForegroundAudioService
    }

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "Creating ForegroundAudioService")
        createNotificationChannel()

        // Initialize AI runners
        yamnetRunner = YamnetRunner(applicationContext)
        sequenceClassifierRunner = SequenceClassifierRunner(applicationContext)

        // Initialize wake-word detector prototype
        wakeWordDetector = DefaultWakeWordDetector(sampleRateHz = SAMPLE_RATE_HZ)
        wakeWordDetector.startListening {
            Log.i(TAG, "Wake word detected via audio loop!")
            // Trigger arming if not already armed
            if (!isMonitoring.get()) {
                startMonitoring()
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action
        Log.i(TAG, "onStartCommand received action: $action")

        when (action) {
            ACTION_START_ARMED -> {
                startForegroundWithNotification(isDegraded = false)
                startMonitoring()
            }
            ACTION_STOP_DISARMED -> {
                stopMonitoring()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
            else -> {
                startForegroundWithNotification(isDegraded = false)
                startMonitoring()
            }
        }

        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    fun setServiceStateListener(listener: ServiceStateListener?) {
        this.serviceStateListener = listener
    }

    /**
     * Starts continuous audio capture and AI inference loops.
     */
    fun startMonitoring() {
        if (isMonitoring.getAndSet(true)) {
            Log.d(TAG, "Monitoring is already active.")
            return
        }

        Log.i(TAG, "Starting ECHO audio capture and AI inference loop.")
        threatEngine.onMonitoringStateChanged(true)
        serviceStateListener?.onMonitoringStateChanged(true, yamnetRunner.modelStatus)

        // Check microphone permission
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "RECORD_AUDIO permission missing; cannot start AudioRecord")
            handleCaptureFailure("Microphone permission denied")
            return
        }

        try {
            minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE_HZ, CHANNEL_CONFIG, AUDIO_FORMAT)
            val bufferSize = minBufferSize.coerceAtLeast(SAMPLE_RATE_HZ * 2 / 10) // 100ms buffer

            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE_HZ,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                handleCaptureFailure("AudioRecord initialization failed")
                return
            }

            audioRecord?.startRecording()
            startCaptureLoop(bufferSize)
            startInferenceLoop()

        } catch (e: Exception) {
            Log.e(TAG, "Exception initializing AudioRecord", e)
            handleCaptureFailure(e.message ?: "Capture exception")
        }
    }

    /**
     * Captures PCM frames from microphone into volatile RAM buffer.
     */
    private fun startCaptureLoop(readChunkSize: Int) {
        captureJob?.cancel()
        captureJob = serviceScope.launch(Dispatchers.IO) {
            val audioData = ShortArray(readChunkSize / 2)
            Log.i(TAG, "Audio capture loop running (volatile RAM buffer: 3.0s @ 16kHz mono).")

            while (isActive && isMonitoring.get()) {
                val record = audioRecord
                if (record == null || record.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                    delay(50)
                    continue
                }

                val readCount = record.read(audioData, 0, audioData.size)
                if (readCount > 0) {
                    // 1. Ingest into 3-second volatile RAM buffer
                    rollingBuffer.write(audioData, readCount)

                    // 2. Feed slice into lightweight wake-word detector
                    wakeWordDetector.processAudio(audioData, readCount)

                    // 3. Compute live RMS level for UI visualization
                    val rms = rollingBuffer.calculateCurrentRms()
                    val dBFS = if (rms > 0.00001f) (20 * kotlin.math.log10(rms.toDouble())).toFloat() else -90f
                    serviceStateListener?.onRmsLevelUpdated(rms, dBFS)
                } else if (readCount < 0) {
                    Log.w(TAG, "AudioRecord read error code: $readCount")
                    delay(50)
                }
            }
            Log.i(TAG, "Audio capture loop terminated.")
        }
    }

    /**
     * Periodic AI inference loop running YAMNet and Sequence Classifier.
     */
    private fun startInferenceLoop() {
        inferenceJob?.cancel()
        inferenceJob = serviceScope.launch(Dispatchers.Default) {
            Log.i(TAG, "AI inference loop started (tick interval: ${INFERENCE_INTERVAL_MS}ms).")

            while (isActive && isMonitoring.get()) {
                delay(INFERENCE_INTERVAL_MS)

                if (!isMonitoring.get()) break

                val timestamp = System.currentTimeMillis()
                // Extract 3-second audio snapshot from volatile RAM (48,000 samples @ 16kHz)
                val audioSnapshot = rollingBuffer.getSnapshotFloat(48000)

                if (audioSnapshot.isNotEmpty()) {
                    // 1. Run YAMNet on sequential temporal windows across the 3-second buffer to extract 6 embeddings
                    val (audioEvent, sixEmbeddings) = yamnetRunner.processAudioWindows(
                        audioSnapshot = audioSnapshot,
                        numWindows = 6,
                        timestampMs = timestamp
                    )
                    threatEngine.onAudioEvent(audioEvent)
                    serviceStateListener?.onAudioEventReceived(audioEvent)

                    // 2. Feed the 6 temporal embeddings [1, 6, 1024] into the Echo Sequence Classifier
                    val classifierResult = sequenceClassifierRunner.classifySequence(
                        embeddings = sixEmbeddings,
                        latestAudioEvent = audioEvent,
                        timestampMs = timestamp
                    )

                    // 3. Dispatch 4-class result to Threat Engine
                    threatEngine.onClassifierResult(classifierResult)
                    serviceStateListener?.onClassifierResultReceived(classifierResult)
                }
            }
            Log.i(TAG, "AI inference loop terminated.")
        }
    }

    /**
     * Gracefully stops monitoring, frees audio resources, and clears volatile buffer.
     */
    fun stopMonitoring() {
        if (!isMonitoring.getAndSet(false)) {
            return
        }

        Log.i(TAG, "Disarming Echo and stopping audio monitoring.")
        captureJob?.cancel()
        inferenceJob?.cancel()

        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing AudioRecord", e)
        } finally {
            audioRecord = null
        }

        rollingBuffer.clear()
        sequenceClassifierRunner.clearHistory()
        threatEngine.onMonitoringStateChanged(false)
        serviceStateListener?.onMonitoringStateChanged(false, ModelStatus.UNAVAILABLE)
    }

    private fun handleCaptureFailure(reason: String) {
        Log.e(TAG, "Acoustic monitoring degraded: $reason")
        isMonitoring.set(false)
        startForegroundWithNotification(isDegraded = true)

        val degradedResult = ClassifierResult(
            timestampMs = System.currentTimeMillis(),
            normalProbability = 0f,
            accidentProbability = 0f,
            distressProbability = 0f,
            violentIncidentProbability = 0f,
            predictedClass = com.echo.android.ai.EmergencyClass.NORMAL,
            modelStatus = ModelStatus.DEGRADED
        )
        threatEngine.onClassifierResult(degradedResult)
        serviceStateListener?.onMonitoringStateChanged(false, ModelStatus.DEGRADED)
    }

    private fun startForegroundWithNotification(isDegraded: Boolean) {
        val notification = buildNotification(isDegraded)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun buildNotification(isDegraded: Boolean): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, ArmEchoActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val title = if (isDegraded) {
            getString(R.string.notification_title_degraded)
        } else {
            getString(R.string.notification_title_armed)
        }

        val text = if (isDegraded) {
            getString(R.string.notification_text_degraded)
        } else {
            getString(R.string.notification_text_armed)
        }

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_shield)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.notification_channel_desc)
                setShowBadge(false)
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "Destroying ForegroundAudioService")
        stopMonitoring()
        wakeWordDetector.stopListening()
        yamnetRunner.close()
        sequenceClassifierRunner.close()
        serviceScope.cancel()
    }
}
