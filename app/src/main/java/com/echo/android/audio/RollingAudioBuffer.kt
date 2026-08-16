package com.echo.android.audio

import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import kotlin.math.sqrt

/**
 * Thread-safe 3-second rolling audio buffer held strictly in volatile RAM.
 *
 * Requirements & Privacy Guarantees:
 * - Buffer size: 3 seconds @ 16kHz mono 16-bit PCM = 48,000 samples (96,000 bytes).
 * - Circular FIFO: continuously overwrites oldest audio frames as new frames arrive.
 * - Volatile RAM only: never writes to disk during normal monitoring, never uploads to network.
 * - Thread-safe concurrent writing (AudioRecord capture loop) and reading (AI inference / wake detector).
 *
 * NOTE: Hackathon prototype implementation.
 */
class RollingAudioBuffer(
    val sampleRateHz: Int = DEFAULT_SAMPLE_RATE_HZ,
    val durationSeconds: Int = DEFAULT_BUFFER_SECONDS
) {

    companion object {
        const val DEFAULT_SAMPLE_RATE_HZ = 16000
        const val DEFAULT_BUFFER_SECONDS = 3
        const val DEFAULT_CAPACITY_SAMPLES = DEFAULT_SAMPLE_RATE_HZ * DEFAULT_BUFFER_SECONDS // 48,000
    }

    val capacitySamples: Int = sampleRateHz * durationSeconds

    // Volatile RAM storage for 16-bit PCM samples
    private val buffer: ShortArray = ShortArray(capacitySamples)

    // Circular write index
    private var writeHead: Int = 0

    // Total samples written (tracks if buffer has filled at least once)
    private var totalSamplesWritten: Long = 0L

    // Read-write lock for thread safety
    private val rwLock = ReentrantReadWriteLock()

    /**
     * Ingests a new slice of 16-bit PCM audio samples into the circular buffer.
     * Older samples are overwritten once capacity is reached.
     *
     * @param samples Array of short PCM samples.
     * @param count Number of valid samples in the array to ingest.
     */
    fun write(samples: ShortArray, count: Int = samples.size) {
        if (count <= 0) return

        rwLock.write {
            val toWrite = count.coerceAtMost(capacitySamples)
            val offset = count - toWrite // If chunk > capacity, take the most recent slice

            for (i in 0 until toWrite) {
                buffer[writeHead] = samples[offset + i]
                writeHead = (writeHead + 1) % capacitySamples
            }
            totalSamplesWritten += count
        }
    }

    /**
     * Extracts a chronological snapshot of the rolling buffer as a normalized float array [-1.0f, 1.0f].
     * Ideal for feeding into YAMNet TFLite interpreter.
     *
     * @param requestedSamples Number of recent samples to extract (defaults to full capacity).
     * @return FloatArray of normalized audio samples in chronological order.
     */
    fun getSnapshotFloat(requestedSamples: Int = capacitySamples): FloatArray {
        rwLock.read {
            val available = availableSamples().coerceAtMost(requestedSamples)
            val result = FloatArray(available)

            if (available == 0) return result

            // Calculate starting read index in circular buffer
            val startIdx = if (totalSamplesWritten >= capacitySamples) {
                (writeHead - available + capacitySamples) % capacitySamples
            } else {
                0
            }

            for (i in 0 until available) {
                val idx = (startIdx + i) % capacitySamples
                result[i] = buffer[idx] / 32768.0f
            }

            return result
        }
    }

    /**
     * Extracts a chronological snapshot of recent PCM short samples.
     *
     * @param requestedSamples Number of recent samples to extract.
     * @return ShortArray in chronological order.
     */
    fun getSnapshotShort(requestedSamples: Int = capacitySamples): ShortArray {
        rwLock.read {
            val available = availableSamples().coerceAtMost(requestedSamples)
            val result = ShortArray(available)

            if (available == 0) return result

            val startIdx = if (totalSamplesWritten >= capacitySamples) {
                (writeHead - available + capacitySamples) % capacitySamples
            } else {
                0
            }

            for (i in 0 until available) {
                val idx = (startIdx + i) % capacitySamples
                result[i] = buffer[idx]
            }

            return result
        }
    }

    /**
     * Calculates the Root Mean Square (RMS) energy of the current buffer.
     * Returns a value between 0.0 (silence) and 1.0 (full scale).
     */
    fun calculateCurrentRms(): Float {
        rwLock.read {
            val available = availableSamples()
            if (available == 0) return 0f

            var sumSquares = 0.0
            val startIdx = if (totalSamplesWritten >= capacitySamples) {
                (writeHead - available + capacitySamples) % capacitySamples
            } else {
                0
            }

            for (i in 0 until available) {
                val idx = (startIdx + i) % capacitySamples
                val normalized = buffer[idx] / 32768.0
                sumSquares += (normalized * normalized)
            }

            return sqrt(sumSquares / available).toFloat()
        }
    }

    /**
     * Returns the number of valid samples currently recorded in the buffer.
     */
    fun availableSamples(): Int {
        return totalSamplesWritten.coerceAtMost(capacitySamples.toLong()).toInt()
    }

    /**
     * Returns true if the buffer has accumulated at least [capacitySamples].
     */
    fun isFull(): Boolean {
        return totalSamplesWritten >= capacitySamples
    }

    /**
     * Resets the buffer in RAM without allocating new arrays.
     */
    fun clear() {
        rwLock.write {
            buffer.fill(0)
            writeHead = 0
            totalSamplesWritten = 0L
        }
    }
}
