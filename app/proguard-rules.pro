# TensorFlow Lite rules
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# Keep data models
-keep class com.echo.android.ai.AudioEvent { *; }
-keep class com.echo.android.ai.ClassifierResult { *; }
-keep class com.echo.android.ai.EmergencyClass { *; }
-keep class com.echo.android.ai.ModelStatus { *; }
-keep class com.echo.android.ai.SensorEvidence { *; }
-keep class com.echo.android.ai.ThreatUpdate { *; }
