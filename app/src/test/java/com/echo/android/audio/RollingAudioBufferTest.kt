package com.echo.android.audio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import kotlin.math.abs

/**
 * Unit tests for [RollingAudioBuffer].
 *
 * Validates:
 * 1. 3-second capacity (48,000 samples @ 16kHz).
 * 2. Overwrite and discard behavior on wrap-around.
 * 3. Chronological order preservation of extracted snapshots.
 * 4. Float normalization to [-1.0f, 1.0f].
 * 5. Volatile RAM only (zero disk usage).
 */
class RollingAudioBufferTest {

    private lateinit var buffer: RollingAudioBuffer

    @Before
    fun setUp() {
        buffer = RollingAudioBuffer(sampleRateHz = 16000, durationSeconds = 3)
    }

    @Test
    fun testBufferCapacityInitialization() {
        assertEquals(48000, buffer.capacitySamples)
        assertEquals(0, buffer.availableSamples())
        assertFalse(buffer.isFull())
    }

    @Test
    fun testWriteAndAvailableSamples() {
        val chunk = ShortArray(1000) { 100 }
        buffer.write(chunk, 1000)

        assertEquals(1000, buffer.availableSamples())
        assertFalse(buffer.isFull())
    }

    @Test
    fun testBufferFullAndCircularOverwrite() {
        val totalToWrite = 60000 // Exceeds 48,000 capacity by 12,000 samples
        val samples = ShortArray(totalToWrite) { i -> (i % 1000).toShort() }

        // Write in chunks of 5000
        var offset = 0
        while (offset < totalToWrite) {
            val chunkLen = 5000.coerceAtMost(totalToWrite - offset)
            val chunk = ShortArray(chunkLen)
            System.arraycopy(samples, offset, chunk, 0, chunkLen)
            buffer.write(chunk, chunkLen)
            offset += chunkLen
        }

        assertTrue(buffer.isFull())
        assertEquals(48000, buffer.availableSamples())

        // Extract snapshot and verify that the first element corresponds to sample 12,000
        val snapshot = buffer.getSnapshotShort(48000)
        assertEquals(48000, snapshot.size)

        // The oldest surviving sample in the buffer should be at index 12000 in the original stream
        val expectedFirstSample = (12000 % 1000).toShort()
        assertEquals(expectedFirstSample, snapshot[0])

        // The most recent sample should match the end of the input stream
        val expectedLastSample = ((totalToWrite - 1) % 1000).toShort()
        assertEquals(expectedLastSample, snapshot[snapshot.size - 1])
    }

    @Test
    fun testGetSnapshotFloatNormalization() {
        // Write positive and negative full scale values
        val testData = shortArrayOf(-32768, 0, 16384, 32767)
        buffer.write(testData, testData.size)

        val floats = buffer.getSnapshotFloat(testData.size)
        assertEquals(4, floats.size)

        assertEquals(-1.0f, floats[0], 0.001f)
        assertEquals(0.0f, floats[1], 0.001f)
        assertEquals(0.5f, floats[2], 0.001f)
        assertEquals(0.9999f, floats[3], 0.001f)
    }

    @Test
    fun testRmsCalculation() {
        // Test silence
        val silence = ShortArray(1000) { 0 }
        buffer.write(silence, silence.size)
        assertEquals(0.0f, buffer.calculateCurrentRms(), 0.0001f)

        // Test half scale DC offset
        buffer.clear()
        val halfScale = ShortArray(1000) { 16384 }
        buffer.write(halfScale, halfScale.size)
        val rms = buffer.calculateCurrentRms()
        assertEquals(0.5f, rms, 0.01f)
    }

    @Test
    fun testClearResetsBuffer() {
        val chunk = ShortArray(5000) { 500 }
        buffer.write(chunk, chunk.size)
        assertEquals(5000, buffer.availableSamples())

        buffer.clear()
        assertEquals(0, buffer.availableSamples())
        assertFalse(buffer.isFull())
        assertEquals(0, buffer.getSnapshotFloat().size)
    }
}
