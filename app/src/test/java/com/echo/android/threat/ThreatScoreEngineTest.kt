package com.echo.android.threat

import com.echo.android.ai.ClassifierResult
import com.echo.android.ai.EmergencyClass
import com.echo.android.ai.ModelStatus
import com.echo.android.ai.SensorEvidence
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class ThreatScoreEngineTest {

    private lateinit var engine: ThreatScoreEngine

    @Before
    fun setup() {
        engine = ThreatScoreEngine(ThreatScoreConfig(sosThreshold = 70))
        engine.onMonitoringStateChanged(true)
    }

    @Test
    fun testNormalSoundDecaysThreatScore() {
        // First raise score
        val accident = ClassifierResult(
            timestampMs = 1000L,
            normalProbability = 0.05f,
            accidentProbability = 0.90f,
            distressProbability = 0.03f,
            violentIncidentProbability = 0.02f,
            predictedClass = EmergencyClass.ACCIDENT,
            modelStatus = ModelStatus.OK
        )
        engine.onClassifierResult(accident, SensorEvidence(1000L, true, 0.8f, false, 0.2f))
        val peakScore = engine.currentThreatScore
        assertTrue(peakScore > 50)

        // Normal sound decays score
        val normal = ClassifierResult(
            timestampMs = 2000L,
            normalProbability = 0.95f,
            accidentProbability = 0.02f,
            distressProbability = 0.02f,
            violentIncidentProbability = 0.01f,
            predictedClass = EmergencyClass.NORMAL,
            modelStatus = ModelStatus.OK
        )
        engine.onClassifierResult(normal, null)
        assertTrue(engine.currentThreatScore < peakScore)

        // Multiple normals decay to 0
        for (i in 0..5) {
            engine.onClassifierResult(normal, null)
        }
        assertEquals(0, engine.currentThreatScore)
    }

    @Test
    fun testAccidentWithMotionCrossesThreshold() {
        val accident = ClassifierResult(
            timestampMs = 1000L,
            normalProbability = 0.02f,
            accidentProbability = 0.92f,
            distressProbability = 0.03f,
            violentIncidentProbability = 0.03f,
            predictedClass = EmergencyClass.ACCIDENT,
            modelStatus = ModelStatus.OK
        )
        val impactMotion = SensorEvidence(1000L, accelerationAnomaly = true, 0.85f, false, 0.4f)
        engine.onClassifierResult(accident, impactMotion)

        assertTrue(engine.currentThreatScore >= 70)
    }

    @Test
    fun testHarmlessSoundWithoutMotionDoesNotCrossThreshold() {
        val harmless = ClassifierResult(
            timestampMs = 1000L,
            normalProbability = 0.90f,
            accidentProbability = 0.05f,
            distressProbability = 0.03f,
            violentIncidentProbability = 0.02f,
            predictedClass = EmergencyClass.NORMAL,
            modelStatus = ModelStatus.OK
        )
        engine.onClassifierResult(harmless, SensorEvidence(1000L, false, 0.0f, false, 0.0f))
        assertFalse(engine.currentThreatScore >= 70)
    }

    @Test
    fun testScoreBoundsStayWithin0To100() {
        val violent = ClassifierResult(
            timestampMs = 1000L,
            normalProbability = 0.0f,
            accidentProbability = 0.0f,
            distressProbability = 0.0f,
            violentIncidentProbability = 1.0f,
            predictedClass = EmergencyClass.VIOLENT_INCIDENT,
            modelStatus = ModelStatus.OK
        )
        val extremeMotion = SensorEvidence(1000L, true, 1.0f, true, 1.0f)
        for (i in 0..5) {
            engine.onClassifierResult(violent, extremeMotion)
        }
        assertTrue(engine.currentThreatScore in 0..100)
    }
}
