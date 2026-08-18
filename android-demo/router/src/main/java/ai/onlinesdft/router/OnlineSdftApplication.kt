package ai.onlinesdft.router

import ai.onlinesdft.router.state.DemoRuntime
import android.app.Application
import android.content.Context

class OnlineSdftApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        runtimeInstance = DemoRuntime(this)
    }

    companion object {
        @Volatile
        private lateinit var runtimeInstance: DemoRuntime

        fun runtime(context: Context): DemoRuntime {
            if (!::runtimeInstance.isInitialized) {
                synchronized(this) {
                    if (!::runtimeInstance.isInitialized) {
                        runtimeInstance = DemoRuntime(context.applicationContext)
                    }
                }
            }
            return runtimeInstance
        }
    }
}
