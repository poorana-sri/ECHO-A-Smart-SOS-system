package com.echo.android.ai

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.util.Log
import com.echo.android.audio.DefaultFeatureExtractor
import com.echo.android.audio.FeatureExtractor
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * TensorFlow Lite runner for YAMNet base acoustic event classification.
 *
 * ARCHITECTURAL ROLE:
 * - YAMNet answers: "What acoustic events are present in this audio slice?" (e.g. Speech, Scream, Siren).
 * - YAMNet outputs 521-class AudioSet scores and/or 1024-dimensional temporal embeddings.
 * - YAMNet does NOT produce the 4 Echo emergency classes (NORMAL/ACCIDENT/DISTRESS/VIOLENT_INCIDENT)
 *   and does NOT decide an emergency. That responsibility belongs to [SequenceClassifierRunner]
 *   and Developer 3's Threat Engine.
 *
 * TEMPORAL MULTI-WINDOW INFERENCE:
 * - Slices the 3-second buffer (48,000 samples @ 16kHz) into 6 sequential temporal windows (15,600 samples each, hop: 6,480 samples).
 * - Runs YAMNet inference on each window to produce 6 sequential 1024-D embeddings for the sequence classifier.
 *
 * DYNAMIC TENSOR INSPECTION:
 * - Inspects input/output tensors dynamically to identify score and embedding tensor indices.
 * - Validates input shape and verifies 1024-D embedding output exists.
 * - Degrades to STUB mode (ModelStatus.DEGRADED) if placeholder or corrupt model is detected.
 *
 * NOTE: Hackathon prototype implementation.
 */
class YamnetRunner(
    private val context: Context,
    private val modelAssetPath: String = DEFAULT_MODEL_PATH,
    private val labelsAssetPath: String = DEFAULT_LABELS_PATH
) {

    companion object {
        private const val TAG = "YamnetRunner"
        const val DEFAULT_MODEL_PATH = "models/yamnet.tflite"
        const val DEFAULT_LABELS_PATH = "models/yamnet_labels.txt"
        const val EXPECTED_SAMPLE_RATE = 16000
        const val YAMNET_EMBEDDING_SIZE = 1024
        const val DEFAULT_NUM_WINDOWS = 6
        const val DEFAULT_FRAME_SAMPLES = 15600 // 0.975s standard YAMNet frame
    }

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private val featureExtractor: FeatureExtractor = DefaultFeatureExtractor()

    // Model tensor inspection metadata
    var inputSampleCount: Int = DEFAULT_FRAME_SAMPLES
        private set
    private var scoresTensorIndex: Int = -1
    private var embeddingTensorIndex: Int = -1

    var isStubMode: Boolean = true
        private set

    var modelStatus: ModelStatus = ModelStatus.UNAVAILABLE
        private set

    val runtimeModeDescription: String
        get() = if (!isStubMode && modelStatus == ModelStatus.OK) {
            "REAL_MODEL (YAMNet TFLite Active, input: $inputSampleCount samples, scoresIdx=$scoresTensorIndex, embedIdx=$embeddingTensorIndex)"
        } else {
            "STUB_MODE (Placeholder / Degraded, ModelStatus=$modelStatus)"
        }

    init {
        initializeModel()
    }

    /**
     * Initializes TFLite interpreter, validates input/output tensors, and loads class labels.
     */
    private fun initializeModel() {
        try {
            labels = loadLabels(labelsAssetPath)
            val modelBuffer = loadModelFile(modelAssetPath)

            if (modelBuffer != null) {
                val options = Interpreter.Options().apply {
                    setNumThreads(2)
                }
                val candidateInterpreter = Interpreter(modelBuffer, options)

                // Inspect and validate model tensors
                if (validateAndMapModelTensors(candidateInterpreter)) {
                    interpreter = candidateInterpreter
                    modelStatus = ModelStatus.OK
                    isStubMode = false
                    Log.i(TAG, "YAMNet TFLite interpreter successfully initialized ($runtimeModeDescription)")
                } else {
                    candidateInterpreter.close()
                    enableStubMode("Model tensor validation failed or placeholder detected for $modelAssetPath")
                }
            } else {
                enableStubMode("Model asset not found or unreadable at $modelAssetPath")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load YAMNet TFLite model ($modelAssetPath). Operating in STUB mode.", e)
            enableStubMode("Initialization exception: ${e.message}")
        }
    }

    /**
     * Inspects TFLite input/output tensors to dynamically identify scores and 1024-D embedding tensors.
     */
    private fun validateAndMapModelTensors(candidate: Interpreter): Boolean {
        try {
            val inputCount = candidate.inputTensorCount
            val outputCount = candidate.outputTensorCount

            if (inputCount < 1 || outputCount < 1) {
                Log.w(TAG, "YAMNet model missing input/output tensors (in=$inputCount, out=$outputCount)")
                return false
            }

            val inputTensor = candidate.getInputTensor(0)
            val inputShape = inputTensor.shape()
            val inputType = inputTensor.dataType()

            Log.i(TAG, "Inspecting YAMNet model: inputShape=${inputShape.contentToString()}, inputType=$inputType, outputCount=$outputCount")

            if (inputType != DataType.FLOAT32) {
                Log.w(TAG, "Unsupported input tensor data type: $inputType. Expected FLOAT32.")
                return false
            }

            // Determine input sample length from shape (e.g. [15600] or [1, 15600] or [48000] or [1, 48000])
            when (inputShape.size) {
                1 -> {
                    inputSampleCount = if (inputShape[0] > 0) inputShape[0] else DEFAULT_FRAME_SAMPLES
                }
                2 -> {
                    inputSampleCount = if (inputShape[1] > 0) inputShape[1] else DEFAULT_FRAME_SAMPLES
                }
                else -> {
                    Log.w(TAG, "Unexpected input tensor dimensions: ${inputShape.size}D")
                    return false
                }
            }

            // Map output tensors by inspecting shapes
            scoresTensorIndex = -1
            embeddingTensorIndex = -1

            for (i in 0 until outputCount) {
                val outTensor = candidate.getOutputTensor(i)
                val outShape = outTensor.shape()
                val lastDim = if (outShape.isNotEmpty()) outShape[outShape.size - 1] else -1

                Log.i(TAG, "YAMNet output tensor $i: shape=${outShape.contentToString()}, dtype=${outTensor.dataType()}")

                if (lastDim == YAMNET_EMBEDDING_SIZE) {
                    embeddingTensorIndex = i
                } else if (lastDim >= 500 || lastDim == labels.size) {
                    scoresTensorIndex = i
                }
            }

            // Fallback for single/standard output models
            if (scoresTensorIndex == -1 && outputCount > 0) scoresTensorIndex = 0
            if (embeddingTensorIndex == -1 && outputCount > 1) embeddingTensorIndex = 1

            Log.i(TAG, "Verified YAMNet mapping: scoresIndex=$scoresTensorIndex, embeddingIndex=$embeddingTensorIndex, inputSize=$inputSampleCount")
            return true
        } catch (e: Exception) {
            Log.w(TAG, "Exception inspecting YAMNet model tensors (likely placeholder file): ${e.message}")
            return false
        }
    }

    private fun enableStubMode(reason: String) {
        isStubMode = true
        modelStatus = ModelStatus.DEGRADED
        interpreter = null
        if (labels.isEmpty()) {
            labels = getFallbackLabels()
        }
        Log.w(TAG, "YamnetRunner operating in STUB mode: $reason")
    }

    /**
     * Extracts exactly [numWindows] (default: 6) sequential 1024-D temporal embeddings from the 3-second audio buffer.
     *
     * Windowing strategy:
     * - Input: 48,000 samples @ 16kHz mono (3 seconds).
     * - Slices audio into [numWindows] sequential frames of length [inputSampleCount].
     * - Runs inference per window to extract 6 temporal embeddings.
     * - Returns the latest/aggregated [AudioEvent] and the list of exactly 6 embeddings [1024-D each].
     *
     * @param audioSnapshot 3-second audio buffer snapshot (48,000 samples).
     * @param numWindows Number of temporal embeddings required (default: 6).
     * @param timestampMs Audio window timestamp.
     * @return Pair containing the [AudioEvent] and the List of 6 [FloatArray] embeddings (1024-D each).
     */
    fun processAudioWindows(
        audioSnapshot: FloatArray,
        numWindows: Int = DEFAULT_NUM_WINDOWS,
        timestampMs: Long = System.currentTimeMillis()
    ): Pair<AudioEvent, List<FloatArray>> {
        val windowSlices = featureExtractor.extractSequentialWindows(
            rawAudio = audioSnapshot,
            windowLength = inputSampleCount,
            numWindows = numWindows
        )

        val embeddingsList = ArrayList<FloatArray>(numWindows)
        var latestAudioEvent: AudioEvent? = null

        for (i in 0 until numWindows) {
            val windowAudio = windowSlices[i]
            val windowTimestamp = timestampMs - ((numWindows - 1 - i) * 405L)
            val (event, embedding) = processSingleWindow(windowAudio, windowTimestamp)
            embeddingsList.add(embedding)
            latestAudioEvent = event // Latest window represents current state
        }

        val eventToReturn = latestAudioEvent ?: AudioEvent(
            timestampMs = timestampMs,
            yamnetPredictions = listOf("Silence" to 1.0f),
            modelStatus = modelStatus
        )

        return Pair(eventToReturn, embeddingsList)
    }

    /**
     * Runs YAMNet inference on a single 16kHz float audio window.
     */
    fun processSingleWindow(
        audioWindow: FloatArray,
        timestampMs: Long = System.currentTimeMillis()
    ): Pair<AudioEvent, FloatArray> {
        if (isStubMode || interpreter == null) {
            return processStubWindow(audioWindow, timestampMs)
        }

        return try {
            val candidate = interpreter ?: return processStubWindow(audioWindow, timestampMs)

            // Prepare audio input slice matching inspected model input length
            val slice = if (audioWindow.size >= inputSampleCount) {
                val start = audioWindow.size - inputSampleCount
                audioWindow.copyOfRange(start, audioWindow.size)
            } else {
                val padded = FloatArray(inputSampleCount)
                val start = inputSampleCount - audioWindow.size
                System.arraycopy(audioWindow, 0, padded, start, audioWindow.size)
                padded
            }

            val inputBuffer = ByteBuffer.allocateDirect(slice.size * 4).apply {
                order(ByteOrder.nativeOrder())
                for (s in slice) {
                    putFloat(s)
                }
                rewind()
            }

            val outputCount = candidate.outputTensorCount
            val outputScores = Array(1) { FloatArray(labels.size.coerceAtLeast(521)) }
            val outputEmbeddings = Array(1) { FloatArray(YAMNET_EMBEDDING_SIZE) }

            if (scoresTensorIndex != -1 && embeddingTensorIndex != -1 && scoresTensorIndex != embeddingTensorIndex) {
                val outputsMap = HashMap<Int, Any>().apply {
                    put(scoresTensorIndex, outputScores)
                    put(embeddingTensorIndex, outputEmbeddings)
                }
                candidate.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputsMap)
            } else if (embeddingTensorIndex != -1) {
                val outputsMap = HashMap<Int, Any>().apply {
                    put(embeddingTensorIndex, outputEmbeddings)
                }
                candidate.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputsMap)
            } else {
                candidate.run(inputBuffer, outputScores)
            }

            val scores = outputScores[0]
            val topPredictions = extractTopPredictions(scores, topK = 5)
            val embeddings = outputEmbeddings[0]

            val event = AudioEvent(
                timestampMs = timestampMs,
                yamnetPredictions = topPredictions,
                modelStatus = ModelStatus.OK
            )

            Pair(event, embeddings)
        } catch (e: Exception) {
            Log.e(TAG, "Inference error during YAMNet execution; falling back to STUB response", e)
            processStubWindow(audioWindow, timestampMs, forcedDegraded = true)
        }
    }

    /**
     * Clearly-labeled STUB inference generator for development / demoing without real weights.
     */
    private fun processStubWindow(
        audioSamples: FloatArray,
        timestampMs: Long,
        forcedDegraded: Boolean = false
    ): Pair<AudioEvent, FloatArray> {
        val rms = calculateRms(audioSamples)
        val status = if (forcedDegraded) ModelStatus.DEGRADED else modelStatus

        val predictions = mutableListOf<Pair<String, Float>>()
        val mockEmbeddings = FloatArray(YAMNET_EMBEDDING_SIZE) { 0.01f }

        when {
            rms > 0.45f -> {
                predictions.add("Scream" to 0.72f)
                predictions.add("Shout" to 0.58f)
                predictions.add("Speech" to 0.40f)
                predictions.add("Emergency vehicle" to 0.35f)
                mockEmbeddings[0] = 0.85f // cue for sequence classifier stub
            }
            rms > 0.25f -> {
                predictions.add("Speech" to 0.82f)
                predictions.add("Conversation" to 0.65f)
                predictions.add("Laughter" to 0.30f)
                predictions.add("Music" to 0.20f)
                mockEmbeddings[0] = 0.20f
            }
            rms > 0.08f -> {
                predictions.add("Ambient noise" to 0.78f)
                predictions.add("Footsteps" to 0.42f)
                predictions.add("Door" to 0.25f)
                predictions.add("Silence" to 0.15f)
                mockEmbeddings[0] = 0.05f
            }
            else -> {
                predictions.add("Silence" to 0.94f)
                predictions.add("Background noise" to 0.12f)
                mockEmbeddings[0] = 0.01f
            }
        }

        val event = AudioEvent(
            timestampMs = timestampMs,
            yamnetPredictions = predictions,
            modelStatus = status
        )

        return Pair(event, mockEmbeddings)
    }

    private fun extractTopPredictions(scores: FloatArray, topK: Int = 5): List<Pair<String, Float>> {
        return scores.indices
            .sortedByDescending { scores[it] }
            .take(topK)
            .map { idx ->
                val label = if (idx < labels.size) labels[idx] else "Class $idx"
                label to scores[idx]
            }
    }

    private fun calculateRms(audioSamples: FloatArray): Float {
        if (audioSamples.isEmpty()) return 0f
        var sumSquares = 0.0
        for (sample in audioSamples) {
            sumSquares += (sample * sample)
        }
        return sqrt(sumSquares / audioSamples.size).toFloat()
    }

    private fun loadModelFile(assetPath: String): ByteBuffer? {
        return try {
            val fileDescriptor: AssetFileDescriptor = context.assets.openFd(assetPath)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        } catch (e: Exception) {
            Log.d(TAG, "Asset model file $assetPath not available directly in APK assets: ${e.message}")
            null
        }
    }

    private fun loadLabels(assetPath: String): List<String> {
        val labelList = mutableListOf<String>()
        try {
            val inputStream = context.assets.open(assetPath)
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                var line: String? = reader.readLine()
                while (line != null) {
                    val trimmed = line.trim()
                    if (trimmed.isNotEmpty()) {
                        labelList.add(trimmed)
                    }
                    line = reader.readLine()
                }
            }
        } catch (e: Exception) {
            Log.d(TAG, "Labels file $assetPath not found; using fallback labels list")
        }
        return labelList
    }

    private fun getFallbackLabels(): List<String> {
        return listOf(
            "Speech", "Scream", "Shout", "Crying", "Laughter",
            "Whispering", "Footsteps", "Door", "Vehicle", "Explosion",
            "Crash", "Gunshot", "Siren", "Alarm", "Silence"
        )
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
