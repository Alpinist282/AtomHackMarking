// Файл настроек проекта settings.gradle.kts, где определяются общие конфигурации для плагинов и зависимости

pluginManagement {
    repositories {
        google() // Официальный репозиторий Google для Android SDK и AndroidX
        mavenCentral() // Основной репозиторий Maven для библиотек с открытым исходным кодом
        gradlePluginPortal() // Портал плагинов Gradle для поддержки дополнительных плагинов
    }

    plugins {
        id("com.android.application") version "8.7.0" // Плагин для создания Android приложений
        id("org.jetbrains.kotlin.android") version "1.9.10" // Плагин для поддержки Kotlin в Android
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS) // Запрещает использование локальных репозиториев в модулях
    repositories {
        google() // Официальный репозиторий Google
        mavenCentral() // Основной репозиторий Maven
    }
}

rootProject.name = "My Application RosAtom" // Название корневого проекта
include(":app") // Подключение модуля "app" к проекту