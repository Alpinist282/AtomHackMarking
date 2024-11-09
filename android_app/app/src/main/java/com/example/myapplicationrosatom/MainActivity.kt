package com.example.myapplicationrosatom

import android.Manifest
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.draw.clip
import coil.compose.AsyncImage
import coil.compose.SubcomposeAsyncImage
import coil.compose.SubcomposeAsyncImageContent
import com.example.myapplicationrosatom.ui.theme.MyApplicationRosAtomTheme
import androidx.compose.ui.graphics.toArgb
import androidx.compose.foundation.shape.RoundedCornerShape
import coil.ImageLoader
import coil.decode.GifDecoder
import coil.decode.ImageDecoderDecoder
import coil.request.ImageRequest
import androidx.compose.ui.platform.LocalContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import androidx.compose.runtime.rememberCoroutineScope

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Полноэкранный режим приложения и установка цвета панели навигации
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(false)
        } else {
            window.decorView.systemUiVisibility = (
                    View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                            or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                            or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                            or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                            or View.SYSTEM_UI_FLAG_FULLSCREEN
                            or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    )
        }
        window.navigationBarColor = Color.White.toArgb()

        setContent {
            MyApplicationRosAtomTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    MainScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(modifier: Modifier = Modifier) {
    // Состояния для отображения информации, диалога, изображения и процесса обработки
    var showInfo by remember { mutableStateOf(false) }
    var showDialog by remember { mutableStateOf(false) }
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var capturedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isImageLoaded by remember { mutableStateOf(false) }
    var isProcessing by remember { mutableStateOf(false) }
    var showGif by remember { mutableStateOf(false) } // Новое состояние для отображения GIF

    val coroutineScope = rememberCoroutineScope()

    // Лаунчер камеры для съемки фото
    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        capturedBitmap = bitmap
        isImageLoaded = bitmap != null
        isProcessing = false
        showGif = false
    }

    // Лаунчер для выбора изображения из галереи
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        imageUri = uri
        isImageLoaded = uri != null
        isProcessing = false
        showGif = false
    }

    // Запрос разрешения на доступ к изображениям
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            galleryLauncher.launch("image/*")
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(top = 16.dp),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        Column {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
            ) {
                // Кнопка с логотипом, для возврата на главный экран
                Image(
                    painter = painterResource(id = R.drawable.rosatom),
                    contentDescription = "RosAtom Logo",
                    modifier = Modifier
                        .size(80.dp)
                        .clickable {
                            showInfo = false
                            if (imageUri == null && capturedBitmap == null) {
                                imageUri = imageUri
                                capturedBitmap = capturedBitmap
                            }
                        }
                )

                Spacer(modifier = Modifier.weight(1f))

                // Кнопка "О ПРИЛОЖЕНИИ", для отображения информации о приложении
                Text(
                    text = "О ПРИЛОЖЕНИИ",
                    color = Color.DarkGray,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.SansSerif,
                    textDecoration = TextDecoration.None,
                    modifier = Modifier
                        .padding(end = 0.dp)
                        .clickable { showInfo = true }
                        .padding(vertical = 20.dp)
                )
            }

            Divider(
                color = Color.LightGray,
                thickness = 1.dp,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp)
            )

            // Раздел с информацией о приложении
            AnimatedVisibility(
                visible = showInfo,
                enter = fadeIn() + slideInVertically(initialOffsetY = { it / 2 }),
                exit = fadeOut() + slideOutVertically(targetOffsetY = { it / 2 })
            ) {
                Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
                    Text(
                        text = "Наше приложение разработано для автоматического распознавания и анализа маркировок деталей и сборочных единиц. С помощью нейросети, размещенной непосредственно на устройстве, приложение позволяет пользователю сфотографировать маркировку, распознать её и определить обозначение и порядковый номер. На основе заранее загруженных данных, приложение находит соответствие этим параметрам, извлекая всю необходимую информацию о позиции изделия или детали.\n\nВысокая скорость работы и автономность обеспечивают получение результата в течение нескольких секунд после съемки, что позволяет значительно повысить удобство и эффективность использования системы.\n" +
                                "\n\nКоманда Toffee Solutions\n09.11.2024",
                        color = Color.DarkGray,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.SansSerif,
                        textDecoration = TextDecoration.None,
                    )
                }
            }

            // Отображение выбранного изображения и кнопки для его очистки
            AnimatedVisibility(
                visible = !showInfo && (imageUri != null || capturedBitmap != null) && isImageLoaded,
                enter = fadeIn() + slideInVertically(initialOffsetY = { it / 2 }),
                exit = fadeOut(animationSpec = tween(durationMillis = 500)) + slideOutVertically(targetOffsetY = { it / 2 }, animationSpec = tween(durationMillis = 500))
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .background(Color.LightGray)
                        .animateContentSize()
                ) {
                    imageUri?.let { uri ->
                        SubcomposeAsyncImage(
                            model = uri,
                            contentDescription = "Selected Image",
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(16.dp)),
                            contentScale = ContentScale.Fit,
                            loading = {
                                // Здесь можно добавить индикатор загрузки
                            },
                            success = { imageState ->
                                val painter = imageState.painter
                                val size = painter.intrinsicSize
                                val aspectRatio = if (size.height != 0f) size.width / size.height else 1f
                                Image(
                                    painter = painter,
                                    contentDescription = null,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .aspectRatio(aspectRatio)
                                        .clip(RoundedCornerShape(16.dp)),
                                    contentScale = ContentScale.Fit
                                )
                            }
                        )
                    }
                    capturedBitmap?.let { bitmap ->
                        val aspectRatio = bitmap.width.toFloat() / bitmap.height.toFloat()
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "Captured Image",
                            modifier = Modifier
                                .fillMaxWidth()
                                .aspectRatio(aspectRatio)
                                .clip(RoundedCornerShape(16.dp)),
                            contentScale = ContentScale.Fit
                        )
                    }

                    // Кнопка для удаления изображения
                    IconButton(
                        onClick = {
                            imageUri = null
                            capturedBitmap = null
                            isImageLoaded = false
                        },
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .size(40.dp)
                            .background(Color(0xFF37487F), shape = CircleShape)
                    ) {
                        Icon(
                            painter = painterResource(id = R.drawable.close),
                            contentDescription = "Очистить изображение",
                            modifier = Modifier.size(24.dp),
                            tint = Color.White
                        )
                    }
                }
            }

            // Отображение GIF, если нажата кнопка "ОБРАБОТАТЬ" через 0.5 секунд
            AnimatedVisibility(
                visible = showGif,
                enter = fadeIn(animationSpec = tween(durationMillis = 500)) + slideInVertically(initialOffsetY = { it / 2 }, animationSpec = tween(durationMillis = 500)),
                exit = fadeOut() + slideOutVertically(targetOffsetY = { it / 2 })
            ) {
                val context = LocalContext.current
                val imageLoader = ImageLoader.Builder(context)
                    .components {
                        if (Build.VERSION.SDK_INT >= 28) {
                            add(ImageDecoderDecoder.Factory())
                        } else {
                            add(GifDecoder.Factory())
                        }
                    }
                    .build()

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .animateContentSize(),
                    contentAlignment = Alignment.Center
                ) {
                    AsyncImage(
                        model = ImageRequest.Builder(context)
                            .data(R.drawable.atom)
                            .build(),
                        contentDescription = "Processing GIF",
                        imageLoader = imageLoader,
                        contentScale = ContentScale.Fit,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp)
                            .clip(RoundedCornerShape(16.dp))
                    )
                }
            }
        }

        // Кнопка внизу экрана для обработки изображения или открытия информации
        AnimatedVisibility(
            visible = !showInfo,
            enter = fadeIn() + slideInVertically(initialOffsetY = { it / 2 }),
            exit = fadeOut() + slideOutVertically(targetOffsetY = { it / 2 })
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 32.dp, vertical = 16.dp),
                horizontalArrangement = Arrangement.Center
            ) {
                Button(
                    onClick = {
                        if (isImageLoaded) {
                            isImageLoaded = false
                            isProcessing = true
                            coroutineScope.launch {
                                delay(500) // Задержка перед отображением GIF
                                showGif = true
                            }
                        } else if (isProcessing) {
                            isProcessing = false
                            showGif = false
                            imageUri = null
                            capturedBitmap = null
                        } else {
                            showDialog = true
                        }
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = when {
                            isProcessing -> "ОТМЕНА"
                            isImageLoaded -> "ОБРАБОТАТЬ"
                            else -> "ИНФО О ДЕТАЛИ"
                        },
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White,
                        fontFamily = FontFamily.SansSerif
                    )
                }
            }
        }
    }

    // Диалог для выбора действия (сделать фото или выбрать из галереи)
    if (showDialog) {
        AlertDialog(
            onDismissRequest = { showDialog = false },
            title = {
                Text(
                    text = "ВЫБЕРИТЕ ДЕЙСТВИЕ",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.DarkGray,
                    fontFamily = FontFamily.SansSerif,
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center
                )
            },
            text = {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Button(
                        onClick = {
                            showDialog = false
                            cameraLauncher.launch(null)
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 8.dp)
                    ) {
                        Text(
                            "СДЕЛАТЬ ФОТО",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White,
                            fontFamily = FontFamily.SansSerif
                        )
                    }
                    Button(
                        onClick = {
                            showDialog = false
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                permissionLauncher.launch(Manifest.permission.READ_MEDIA_IMAGES)
                            } else {
                                permissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
                            }
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 8.dp)
                    ) {
                        Text(
                            "ВЫБРАТЬ ИЗ ГАЛЕРЕИ",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color.White,
                            fontFamily = FontFamily.SansSerif
                        )
                    }
                }
            },
            confirmButton = {
                TextButton(
                    onClick = { showDialog = false },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        "ОТМЕНА",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.DarkGray,
                        fontFamily = FontFamily.SansSerif,
                        modifier = Modifier.fillMaxWidth(),
                        textAlign = TextAlign.Center
                    )
                }
            }
        )
    }
}

@Preview(showBackground = true)
@Composable
fun MainScreenPreview() {
    MyApplicationRosAtomTheme {
        MainScreen()
    }
}