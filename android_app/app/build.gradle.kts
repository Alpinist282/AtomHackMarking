// Файл сборки для модуля приложения. Здесь настраиваются плагины, конфигурации компиляции, зависимости и другие параметры, специфичные для приложения.

plugins {
    id("com.android.application") // Плагин для Android-приложений
    id("org.jetbrains.kotlin.android") // Плагин для Kotlin на Android
}

android {
    namespace = "com.example.myapplicationrosatom" // Пространство имен приложения
    compileSdk = 34 // Версия SDK для компиляции

    defaultConfig {
        applicationId = "com.example.myapplicationrosatom" // Идентификатор приложения
        minSdk = 21 // Минимальная поддерживаемая версия SDK
        targetSdk = 34 // Целевая версия SDK
        versionCode = 1 // Код версии приложения
        versionName = "1.0" // Имя версии приложения
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner" // Тестовый раннер
    }

    buildTypes {
        release {
            isMinifyEnabled = false // Отключение минификации для сборки release
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro" // Файлы ProGuard для оптимизации и защиты кода
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_19 // Совместимость исходного кода с Java 19
        targetCompatibility = JavaVersion.VERSION_19 // Целевая совместимость с Java 19
    }

    kotlinOptions {
        jvmTarget = "19" // Целевая версия JVM для Kotlin
    }

    buildFeatures {
        compose = true // Включение поддержки Jetpack Compose
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.3" // Версия расширения компилятора для Compose
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}" // Исключение некоторых файлов лицензий из ресурсов
        }
    }
}

dependencies {
    // Зависимость для загрузки изображений с помощью Coil
    implementation("io.coil-kt:coil-compose:2.2.2")
    implementation("io.coil-kt:coil-gif:2.2.2") // Поддержка GIF в Coil

    // Основные зависимости Android
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")
    implementation("androidx.activity:activity-compose:1.7.2")

    // Зависимости для Jetpack Compose, используя BOM (Bill of Materials)
    implementation(platform("androidx.compose:compose-bom:2023.10.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")

    // Coil для загрузки изображений в Compose
    implementation("io.coil-kt:coil-compose:2.4.0")

    // Зависимости для тестирования
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")

    // Зависимость для UI-тестов Compose, поддержка BOM
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")

    debugImplementation("androidx.compose.ui:ui-tooling") // Инструменты отладки для UI
    debugImplementation("androidx.compose.ui:ui-test-manifest") // Зависимость для манифеста в тестах
}