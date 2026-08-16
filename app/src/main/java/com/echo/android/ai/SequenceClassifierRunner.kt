package com.echo.android.ai

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max

/**
 * TensorFlow Lite runner for the ECHO Sequence Classifier (LSTM/GRU).
 *
 * DEVELOPER 2 FINAL MODEL CONTRACT:
 * - INPUT TENSOR: [1, 6, 1024] Float32 (Batch=1, Timesteps=6, Features=1024)
 * - OUTPUT TENSOR: [1, 4] Float32 Softmax Probabilities
 * - CLASS MAPPING:
 *   0: NORMAL
 *   1: ACCIDENT
 *   2: DISTRESS
 *   3: VIOLENT_INCIDENT
 *
 * VALIDATION & RESILIENCE:
 * - Validates input rank = 3, input shape [1, 6, 1024], and output shape [1, 4].
 * - Deterministic 6-frame padding ensures the model NEVER receives variable or fewer than 6 frames.
 * - Graceful fallback to STUB mode (ModelStatus.DEGRADED) on missing/incompatible model files.
 *
 * NOTE: Hackathon prototype implementation.
 */
class SequenceClassifierRunner(
    private val context: Context,
    private val modelAssetPath: String = DEFAULT_MODEL_PATH
) {

    companion object {
        private const val TAG = "SeqClassifierRunner"
        const val DEFAULT_MODEL_PATH = "models/echo_sequence_classifier.tflite"
        const val REQUIRED_SEQUENCE_LENGTH = 6 // Exactly 6 temporal frames required by Dev 2 model
        const val FEATURE_DIM = 1024 // Matches YAMNet embedding dimension
        const val NUM_CLASSES = 4
    }

    private var interpreter: Interpreter? = null

    // Rolling history of temporal feature vectors (capacity: 6)
    private val featureHistory = ArrayDeque<FloatArray>(REQUIRED_SEQUENCE_LENGTH)

    var isStubMode: Boolean = true
        private set

    var modelStatus: ModelStatus = ModelStatus.UNAVAILABLE
        private set

    val runtimeModeDescription: String
        get() = if (!isStubMode && modelStatus == ModelStatus.OK) {
            "REAL_MODEL (Echo Sequence Classifier TFLite Active, Input: [1, $REQUIRED_SEQUENCE_LENGTH, $FEATURE_DIM], Output: [1, $NUM_CLASSES])"
        } else {
            "STUB_MODE (Placeholder / Degraded, ModelStatus=$modelStatus)"
        }

    init {
        initializeModel()
    }

    private fun initializeModel() {
        try {
            val modelBuffer = loadModelFile(modelAssetPath)
            if (modelBuffer != null) {
                val options = Interpreter.Options().apply {
                    setNumThreads(2)
                }
                val candidateInterpreter = Interpreter(modelBuffer, options)

                if (validateModelTensors(candidateInterpreter)) {
                    interpreter = candidateInterpreter
                    modelStatus = ModelStatus.OK
                    isStubMode = false
                    Log.i(TAG, "Echo Sequence Classifier TFLite successfully initialized ($runtimeModeDescription)")
                } else {
                    candidateInterpreter.close()
                    enableStubMode("Tensor validation failed or placeholder model file at $modelAssetPath")
                }
            } else {
                enableStubMode("Model asset file not found at $modelAssetPath")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to initialize Sequence Classifier TFLite model ($modelAssetPath). Operating in STUB mode.", e)
            enableStubMode("Initialization exception: ${e.message}")
        }
    }

    /**
     * Validates that the TFLite model conforms strictly to [1, 6, 1024] -> [1, 4].
     */
    private fun validateModelTensors(candidate: Interpreter): Boolean {
        try {
            if (candidate.inputTensorCount < 1 || candidate.outputTensorCount < 1) {
                Log.w(TAG, "Sequence Classifier model missing input/output tensors.")
                return false
            }

            val inputTensor = candidate.getInputTensor(0)
            val outputTensor = candidate.getOutputTensor(0)
            val inputShape = inputTensor.shape()
            val outputShape = outputTensor.shape()

            Log.i(
                TAG,
                "Inspecting Sequence Classifier: inputShape=${inputShape.contentToString()}, " +
                        "inputType=${inputTensor.dataType()}, outputShape=${outputShape.contentToString()}, outputType=${outputTensor.dataType()}"
            )

            // Validate Input: rank 3, dtype FLOAT32
            if (inputTensor.dataType() != DataType.FLOAT32) {
                Log.w(TAG, "Expected FLOAT32 input tensor for Sequence Classifier.")
                return false
            }

            if (inputShape.size != 3) {
                Log.w(TAG, "Expected 3D input tensor [1, $REQUIRED_SEQUENCE_LENGTH, $FEATURE_DIM], got ${inputShape.size}D")
                return false
            }

            // Check dimensions if specified in model metadata
            val seqDim = inputShape[1]
            val featDim = inputShape[2]
            if ((seqDim != -1 && seqDim != REQUIRED_SEQUENCE_LENGTH) || (featDim != -1 && featDim != FEATURE_DIM)) {
                Log.w(TAG, "Incompatible input shape ${inputShape.contentToString()}. Expected [1, $REQUIRED_SEQUENCE_LENGTH, $FEATURE_DIM]")
                return false
            }

            // Validate Output: dtype FLOAT32, num_classes = 4
            if (outputTensor.dataType() != DataType.FLOAT32) {
                Log.w(TAG, "Expected FLOAT32 output tensor for Sequence Classifier.")
                return false
            }

            val outLastDim = if (outputShape.isNotEmpty()) outputShape[outputShape.size - 1] else -1
            if (outLastDim != -1 && outLastDim != NUM_CLASSES) {
                Log.w(TAG, "Incompatible output classes count: $outLastDim. Expected $NUM_CLASSES.")
                return false
            }

            return true
        } catch (e: Exception) {
            Log.w(TAG, "Exception inspecting Sequence Classifier model tensors (likely placeholder file): ${e.message}")
            return false
        }
    }

    private fun enableStubMode(reason: String) {
        isStubMode = true
        modelStatus = ModelStatus.DEGRADED
        interpreter = null
        Log.w(TAG, "SequenceClassifierRunner operating in STUB mode: $reason")
    }

    /**
     * Ingests temporal feature embeddings and runs sequence classification.
     *
     * @param embeddings Sequence of 1024-D temporal feature vectors (e.g. 6 temporal frames extracted from YAMNet).
     * @param latestAudioEvent The latest [AudioEvent] for contextual heuristics in STUB mode.
     * @param timestampMs Timestamp of the current inference window.
     */
    fun classifySequence(
        embeddings: List<FloatArray>,
        latestAudioEvent: AudioEvent? = null,
        timestampMs: Long = System.currentTimeMillis()
    ): ClassifierResult {
        // Ensure exactly 6 frames via deterministic padding/slicing
        val normalizedFrames = prepareSixFrames(embeddings)

        if (isStubMode || interpreter == null) {
            return processStubSequence(latestAudioEvent, timestampMs)
        }

        return try {
            val candidate = interpreter ?: return processStubSequence(latestAudioEvent, timestampMs)

            // Prepare fixed input tensor [1, 6, 1024]
            val inputBuffer = ByteBuffer.allocateDirect(1 * REQUIRED_SEQUENCE_LENGTH * FEATURE_DIM * 4).apply {
                order(ByteOrder.nativeOrder())
                for (frame in normalizedFrames) {
                    val frameLength = frame.size.coerceAtMost(FEATURE_DIM)
                    for (i in 0 until frameLength) {
                        putFloat(frame[i])
                    }
                    // Pad frame if smaller than 1024
                    for (i in frameLength until FEATURE_DIM) {
                        putFloat(0.0f)
                    }
                }
                rewind()
            }

            // Output tensor [1, 4] for 4 classes
            val outputBuffer = Array(1) { FloatArray(NUM_CLASSES) }
            candidate.run(inputBuffer, outputBuffer)

            val probs = outputBuffer[0]
            val normalProb = probs[0].coerceIn(0.0f, 1.0f)
            val accidentProb = probs[1].coerceIn(0.0f, 1.0f)
            val distressProb = probs[2].coerceIn(0.0f, 1.0f)
            val violentProb = probs[3].coerceIn(0.0f, 1.0f)

            val predictedClass = determinePredictedClass(normalProb, accidentProb, distressProb, violentProb)

            ClassifierResult(
                timestampMs = timestampMs,
                normalProbability = normalProb,
                accidentProbability = accidentProb,
                distressProbability = distressProb,
                violentIncidentProbability = violentProb,
                predictedClass = predictedClass,
                modelStatus = ModelStatus.OK
            )
        } catch (e: Exception) {
            Log.e(TAG, "Inference error in Sequence Classifier; falling back to STUB output", e)
            processStubSequence(latestAudioEvent, timestampMs, forcedDegraded = true)
        }
    }

    /**
     * Deterministically produces a list of exactly 6 frames.
     * If fewer than 6 frames are provided, front-pads with zeroed silence embeddings.
     */
    private fun prepareSixFrames(input: List<FloatArray>): List<FloatArray> {
        val result = ArrayList<FloatArray>(REQUIRED_SEQUENCE_LENGTH)

        if (input.size >= REQUIRED_SEQUENCE_LENGTH) {
            // Take the most recent 6 frames
            val startIdx = input.size - REQUIRED_SEQUENCE_LENGTH
            for (i in 0 until REQUIRED_SEQUENCE_LENGTH) {
                result.add(input[startIdx + i])
            }
        } else {
            // Deterministic front-padding with zero/silence vectors
            val missing = REQUIRED_SEQUENCE_LENGTH - input.size
            repeat(missing) {
                result.add(FloatArray(FEATURE_DIM) { 0.0f })
            }
            result.addAll(input)
        }

        return result
    }

    /**
     * Context-aware STUB sequence classifier logic for development & demoing without real weights.
     */
    private fun processStubSequence(
        latestAudioEvent: AudioEvent?,
        timestampMs: Long,
        forcedDegraded: Boolean = false
    ): ClassifierResult {
        val status = if (forcedDegraded) ModelStatus.DEGRADED else modelStatus

        var normalProb = 0.94f
        var accidentProb = 0.02f
        var distressProb = 0.02f
        var violentProb = 0.02f

        if (latestAudioEvent != null) {
            for ((label, prob) in latestAudioEvent.yamnetPredictions) {
                when {
                    label.contains("Crash", ignoreCase = true) || label.contains("Impact", ignoreCase = true) -> {
                        accidentProb = max(accidentProb, prob * 0.85f)
                    }
                    label.contains("Scream", ignoreCase = true) || label.contains("Shout", ignoreCase = true) || label.contains("Crying", ignoreCase = true) -> {
                        distressProb = max(distressProb, prob * 0.80f)
                    }
                    label.contains("Explosion", ignoreCase = true) || label.contains("Gunshot", ignoreCase = true) -> {
                        violentProb = max(violentProb, prob * 0.75f)
                    }
                }
            }

            val maxEmergency = max(accidentProb, max(distressProb, violentProb))
            if (maxEmergency > 0.30f) {
                normalProb = max(0.05f, 1.0f - (accidentProb + distressProb + violentProb))
            }
        }

        // Normalize sum to 1.0
        val sum = normalProb + accidentProb + distressProb + violentProb
        val normNormal = normalProb / sum
        val normAccident = accidentProb / sum
        val normDistress = distressProb / sum
        val normViolent = violentProb / sum

        val predictedClass = determinePredictedClass(normNormal, normAccident, normDistress, normViolent)

        return ClassifierResult(
            timestampMs = timestampMs,
            normalProbability = normNormal,
            accidentProbability = normAccident,
            distressProbability = normDistress,
            violentIncidentProbability = normViolent,
            predictedClass = predictedClass,
            modelStatus = status
        )
    }

    private fun determinePredictedClass(
        normal: Float,
        accident: Float,
        distress: Float,
        violent: Float
    ): EmergencyClass {
        return when {
            accident >= normal && accident >= distress && accident >= violent -> EmergencyClass.ACCIDENT
            distress >= normal && distress >= accident && distress >= violent -> EmergencyClass.DISTRESS
            violent >= normal && violent >= accident && violent >= distress -> EmergencyClass.VIOLENT_INCIDENT
            else -> EmergencyClass.NORMAL
        }
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
            Log.d(TAG, "Asset model file $assetPath not found directly in APK assets: ${e.message}")
            null
        }
    }

    fun clearHistory() {
        featureHistory.clear()
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
