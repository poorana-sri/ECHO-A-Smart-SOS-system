package com.echo.android.sos

import android.os.Handler
import android.os.Looper
import android.util.Log

/**
 * Manages the 10-second pre-SOS warning countdown.
 * Allows cancellation via UI button or voice ("Hey Echo, cancel SOS").
 */
class CountdownController(
    private val totalSeconds: Int = 10,
    private val mainHandler: Handler = Handler(Looper.getMainLooper())
) {

    companion object {
        private const val TAG = "CountdownController"
    }

    interface CountdownListener {
        fun onTick(secondsRemaining: Int)
        fun onCountdownFinished()
        fun onCountdownCancelled()
    }

    @Volatile
    var isRunning: Boolean = false
        private set

    @Volatile
    var secondsRemaining: Int = totalSeconds
        private set

    private var listener: CountdownListener? = null

    private val tickRunnable = object : Runnable {
        override fun run() {
            if (!isRunning) return

            secondsRemaining--
            Log.d(TAG, "Countdown tick: $secondsRemaining seconds remaining")
            listener?.onTick(secondsRemaining)

            if (secondsRemaining <= 0) {
                isRunning = false
                Log.w(TAG, "Countdown finished. Confirming ACTIVE_SOS.")
                listener?.onCountdownFinished()
            } else {
                mainHandler.postDelayed(this, 1000L)
            }
        }
    }

    fun setListener(listener: CountdownListener?) {
        this.listener = listener
    }

    /**
     * Starts the 10-second warning countdown.
     */
    @Synchronized
    fun startCountdown() {
        if (isRunning) return
        isRunning = true
        secondsRemaining = totalSeconds
        Log.i(TAG, "Starting $totalSeconds-second SOS countdown...")
        listener?.onTick(secondsRemaining)
        mainHandler.postDelayed(tickRunnable, 1000L)
    }

    /**
     * Cancels the countdown immediately.
     */
    @Synchronized
    fun cancelCountdown() {
        if (!isRunning) return
        isRunning = false
        mainHandler.removeCallbacks(tickRunnable)
        Log.i(TAG, "SOS countdown cancelled by user.")
        listener?.onCountdownCancelled()
    }
}
