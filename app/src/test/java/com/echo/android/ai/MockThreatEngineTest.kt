package com.echo.android.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for [MockThreatEngine].
 *
 * Validates:
 * 1. Threat score rises on ACCIDENT or DISTRESS predictions while armed.
 * 2. Threat score decays back towards 0 on NORMAL predictions.
 * 3. Threshold crossing flag triggers at >= 70.
 * 4. Manual SOS triggers score = 100.
 */
class MockThreatEngineTest {

    private lateinit var threatEngine: MockThreatEngine

    @Before
    fun setUp() {
        threatEngine = MockThreatEngine()
        threatEngine.onMonitoringStateChanged(true)
    }

    @Test
    fun testAccidentClassificationIncreasesScore() {
        assertEquals(0, threatEngine.currentThreatScore)

        val accidentResult = ClassifierResult(
            timestampMs = System.currentTimeMillis(),
            normalProbability = 0.05f,
            accidentProbability = 0.90f,
            distressProbability = 0.03f,
            violentIncidentProbability = 0.02f,
            predictedClass = EmergencyClass.ACCIDENT,
            modelStatus = ModelStatus.OK
        )

        threatEngine.onClassifierResult(accidentResult)
        assertTrue("Threat score must increase on ACCIDENT", threatEngine.currentThreatScore > 40)
    }

    @Test
    fun testScoreDecaysOnNormalPredictions() {
        // First inject an emergency to raise score
        val distressResult = ClassifierResult(
            timestampMs = System.currentTimeMillis(),
            normalProbability = 0.05f,
            accidentProbability = 0.05f,
            distressProbability = 0.85f,
            violentIncidentProbability = 0.05f,
            predictedClass = EmergencyClass.DISTRESS,
            modelStatus = ModelStatus.OK
        )
        threatEngine.onClassifierResult(distressResult)
        val elevatedScore = threatEngine.currentThreatScore
        assertTrue(elevatedScore > 40)

        // Inject series of NORMAL predictions
        val normalResult = ClassifierResult(
            timestampMs = System.currentTimeMillis(),
            normalProbability = 0.95f,
            accidentProbability = 0.02f,
            distressProbability = 0.02f,
            violentIncidentProbability = 0.01f,
            predictedClass = EmergencyClass.NORMAL,
            modelStatus = ModelStatus.OK
        )

        threatEngine.onClassifierResult(normalResult)
        assertTrue("Threat score must decay on NORMAL", threatEngine.currentThreatScore < elevatedScore)

        repeat(5) {
            threatEngine.onClassifierResult(normalResult)
        }
        assertEquals("Threat score should settle to 0 after repeated NORMAL frames", 0, threatEngine.currentThreatScore)
    }

    @Test
    fun testThresholdCrossingAndListenerNotification() {
        var lastUpdate: ThreatUpdate? = null
        threatEngine.addThreatListener { update ->
            lastUpdate = update
        }

        val highDistressResult = ClassifierResult(
            timestampMs = System.currentTimeMillis(),
            normalProbability = 0.0f,
            accidentProbability = 0.0f,
            distressProbability = 0.99f,
            violentIncidentProbability = 0.01f,
            predictedClass = EmergencyClass.DISTRESS,
            modelStatus = ModelStatus.OK
        )

        val sensorAnomaly = SensorEvidence(
            timestampMs = System.currentTimeMillis(),
            accelerationAnomaly = true,
            accelerationScore = 0.8f
        )

        threatEngine.onClassifierResult(highDistressResult, sensorAnomaly)

        assertTrue(threatEngine.currentThreatScore >= MockThreatEngine.SOS_THRESHOLD)
        assertTrue(lastUpdate?.thresholdCrossed == true)
    }

    @Test
    fun testManualSosTrigger() {
        var triggered = false
        threatEngine.addThreatListener { update ->
            if (update.threatScore == 100 && update.thresholdCrossed) {
                triggered = true
            }
        }

        threatEngine.triggerManualSos()
        assertTrue("Manual SOS must set score to 100 and crossed threshold", triggered)
        assertEquals(100, threatEngine.currentThreatScore)
    }
}
