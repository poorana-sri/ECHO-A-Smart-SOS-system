package com.echo.android.audio

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Interface isolating audio preprocessing and feature extraction.
 *
 * Developer 2 supplies the exact feature extraction logic (e.g. 96x64 mel spectrogram,
 * STFT windowing, or log-mel filterbanks).
 *
 * This contract ensures a clean, drop-in replacement once Developer 2's pipeline lands.
 */
interface FeatureExtractor {

    /**
     * Preprocesses a raw float audio window for AI model ingestion.
     *
     * @param rawAudio 16kHz mono audio samples normalized to [-1.0, 1.0].
     * @param targetLength Target number of samples (e.g. 15,600 for 0.975s YAMNet frame or 48,000 for 3.0s).
     * @return Formatted float array ready for model input tensor.
     */
    fun extractFeatures(rawAudio: FloatArray, targetLength: Int = 48000): FloatArray

    /**
     * Extracts a fixed number of sequential temporal windows from the rolling audio buffer.
     *
     * Slicing strategy:
     * - Divides the 3-second buffer (48,000 samples) across [numWindows] (default: 6) overlapping frames of size [windowLength] (default: 15,600).
     * - Hop size: (48,000 - 15,600) / 5 = 6,480 samples (~405 ms).
     * - If buffer has fewer than [windowLength] samples, deterministically zero-pads each window.
     *
     * @param rawAudio Audio samples from 3-second rolling buffer (16kHz mono).
     * @param windowLength Length of each YAMNet input window (e.g. 15,600 samples = 0.975s).
     * @param numWindows Number of temporal windows to produce (default: 6).
     * @return List of exactly [numWindows] float arrays, each of length [windowLength].
     */
    fun extractSequentialWindows(
        rawAudio: FloatArray,
        windowLength: Int = 15600,
        numWindows: Int = 6
    ): List<FloatArray>

    /**
     * Computes the Root Mean Square (RMS) energy of an audio slice.
     */
    fun computeRms(audio: FloatArray): Float

    /**
     * Computes Zero Crossing Rate (ZCR) — a useful lightweight acoustic descriptor.
     */
    fun computeZeroCrossingRate(audio: FloatArray): Float
}

/**
 * Standard implementation of [FeatureExtractor].
 * Handles zero-padding, truncation, and sequential multi-window slicing for temporal AI pipelines.
 */
class DefaultFeatureExtractor : FeatureExtractor {

    override fun extractFeatures(rawAudio: FloatArray, targetLength: Int): FloatArray {
        val result = FloatArray(targetLength)

        if (rawAudio.isEmpty()) {
            return result
        }

        if (rawAudio.size >= targetLength) {
            // Take the most recent `targetLength` samples
            val srcOffset = rawAudio.size - targetLength
            System.arraycopy(rawAudio, srcOffset, result, 0, targetLength)
        } else {
            // Right-align recent samples, zero-pad the front
            val dstOffset = targetLength - rawAudio.size
            System.arraycopy(rawAudio, 0, result, dstOffset, rawAudio.size)
        }

        return result
    }

    override fun extractSequentialWindows(
        rawAudio: FloatArray,
        windowLength: Int,
        numWindows: Int
    ): List<FloatArray> {
        val windows = ArrayList<FloatArray>(numWindows)

        if (rawAudio.isEmpty() || windowLength <= 0 || numWindows <= 0) {
            // Return empty / zeroed windows
            repeat(numWindows) {
                windows.add(FloatArray(max(1, windowLength)))
            }
            return windows
        }

        if (rawAudio.size < windowLength) {
            // Pad raw audio to at least windowLength
            val padded = FloatArray(windowLength)
            val offset = windowLength - rawAudio.size
            System.arraycopy(rawAudio, 0, padded, offset, rawAudio.size)
            repeat(numWindows) {
                windows.add(padded.copyOf())
            }
            return windows
        }

        val totalSpan = rawAudio.size - windowLength
        val hop = if (numWindows > 1) totalSpan / (numWindows - 1) else 0

        for (w in 0 until numWindows) {
            val start = (w * hop).coerceAtMost(rawAudio.size - windowLength)
            val window = rawAudio.copyOfRange(start, start + windowLength)
            windows.add(window)
        }

        return windows
    }

    override fun computeRms(audio: FloatArray): Float {
        if (audio.isEmpty()) return 0f
        var sumSquares = 0.0
        for (sample in audio) {
            sumSquares += (sample * sample)
        }
        return sqrt(sumSquares / audio.size).toFloat()
    }

    override fun computeZeroCrossingRate(audio: FloatArray): Float {
        if (audio.size < 2) return 0f
        var zeroCrossings = 0
        for (i in 1 until audio.size) {
            if ((audio[i] >= 0 && audio[i - 1] < 0) || (audio[i] < 0 && audio[i - 1] >= 0)) {
                zeroCrossings++
            }
        }
        return zeroCrossings.toFloat() / (audio.size - 1)
    }
}
