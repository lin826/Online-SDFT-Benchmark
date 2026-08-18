import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.onlinesdft.publisher"
    compileSdk = 34

    defaultConfig {
        applicationId = "ai.onlinesdft.publisher"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    flavorDimensions += "publisher"
    productFlavors {
        create("chat") {
            dimension = "publisher"
            applicationId = "ai.onlinesdft.publisher.chat"
            versionNameSuffix = "-chat"
            resValue("string", "app_name", "Chat")
            resValue("string", "publisher_category", "chat")
        }
        create("calendar") {
            dimension = "publisher"
            applicationId = "ai.onlinesdft.publisher.calendar"
            versionNameSuffix = "-calendar"
            resValue("string", "app_name", "Calendar")
            resValue("string", "publisher_category", "calendar")
        }
        create("mail") {
            dimension = "publisher"
            applicationId = "ai.onlinesdft.publisher.mail"
            versionNameSuffix = "-mail"
            resValue("string", "app_name", "Mail")
            resValue("string", "publisher_category", "mail")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_17)
        }
    }
}
