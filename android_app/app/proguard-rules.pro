# Добавьте здесь специфичные для проекта правила ProGuard.
# Вы можете управлять набором применяемых файлов конфигурации, используя
# параметр proguardFiles в build.gradle.
#
# Для более подробной информации см.:
#   http://developer.android.com/guide/developing/tools/proguard.html

# Если ваш проект использует WebView с JavaScript, раскомментируйте следующее
# и укажите полностью квалифицированное имя класса для интерфейса JavaScript:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Раскомментируйте, чтобы сохранить информацию о номерах строк
# для отладки трассировок стека.
#-keepattributes SourceFile,LineNumberTable

# Если вы сохраняете информацию о номерах строк, раскомментируйте, чтобы
# скрыть оригинальные имена файлов исходного кода.
#-renamesourcefileattribute SourceFile