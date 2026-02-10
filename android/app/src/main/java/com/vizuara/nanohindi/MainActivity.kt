package com.vizuara.nanohindi

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private val viewModel: ChatViewModel by viewModels()
    private lateinit var voiceInput: VoiceInput
    private lateinit var hindiTTS: HindiTTS

    private val micPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) voiceInput.startListening()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        voiceInput = VoiceInput(this)
        hindiTTS = HindiTTS(this)

        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = Color(0xFFFF9800),
                    onPrimary = Color.Black,
                    surface = Color(0xFF1A1A2E),
                    onSurface = Color.White,
                    background = Color(0xFF0F0F23),
                    onBackground = Color.White,
                    surfaceVariant = Color(0xFF252547),
                    onSurfaceVariant = Color(0xFFB0B0C0),
                )
            ) {
                NanoHindiApp()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        voiceInput.destroy()
        hindiTTS.shutdown()
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun NanoHindiApp() {
        val modelState by viewModel.modelState.collectAsState()
        val downloadState by viewModel.modelManager.downloadState.collectAsState()
        val messages by viewModel.messages.collectAsState()
        val isGenerating by viewModel.isGenerating.collectAsState()
        val currentResponse by viewModel.currentResponse.collectAsState()
        val voiceState by voiceInput.state.collectAsState()
        val ttsReady by hindiTTS.isReady.collectAsState()
        val scope = rememberCoroutineScope()

        // Auto-send voice result
        LaunchedEffect(voiceState) {
            if (voiceState is VoiceInput.VoiceState.Result) {
                val text = (voiceState as VoiceInput.VoiceState.Result).text
                viewModel.sendMessage(text)
                voiceInput.resetState()
            }
        }

        // Auto-load model when download completes
        LaunchedEffect(downloadState) {
            if (downloadState is ModelManager.DownloadState.Completed && modelState is ModelState.NotLoaded) {
                viewModel.loadModel()
            }
        }

        // Check if model is ready on start
        LaunchedEffect(Unit) {
            if (viewModel.modelManager.isModelReady()) {
                viewModel.loadModel()
            }
        }

        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Column {
                            Text("Nano Hindi", fontWeight = FontWeight.Bold, fontSize = 20.sp)
                            Text(
                                when (modelState) {
                                    is ModelState.NotLoaded -> "Model not loaded"
                                    is ModelState.Loading -> "Loading model..."
                                    is ModelState.Ready -> "Ready"
                                    is ModelState.Error -> "Error"
                                },
                                fontSize = 12.sp,
                                color = when (modelState) {
                                    is ModelState.Ready -> Color(0xFF4CAF50)
                                    is ModelState.Error -> Color(0xFFF44336)
                                    else -> Color(0xFFB0B0C0)
                                }
                            )
                        }
                    },
                    actions = {
                        IconButton(onClick = { viewModel.clearChat() }) {
                            Icon(Icons.Default.Delete, "Clear chat")
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = Color(0xFF1A1A2E)
                    )
                )
            },
            containerColor = Color(0xFF0F0F23)
        ) { paddingValues ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
            ) {
                when {
                    // Show download screen if model not ready
                    !viewModel.modelManager.isModelReady() && modelState !is ModelState.Ready -> {
                        DownloadScreen(
                            downloadState = downloadState,
                            onDownload = {
                                scope.launch { viewModel.modelManager.downloadModel() }
                            }
                        )
                    }
                    // Show loading indicator
                    modelState is ModelState.Loading -> {
                        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                CircularProgressIndicator(color = Color(0xFFFF9800))
                                Spacer(Modifier.height(16.dp))
                                Text("Loading model into memory...", color = Color.White)
                                Text("This may take a moment", fontSize = 12.sp, color = Color(0xFFB0B0C0))
                            }
                        }
                    }
                    // Show error
                    modelState is ModelState.Error -> {
                        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("Error", color = Color(0xFFF44336), fontSize = 20.sp, fontWeight = FontWeight.Bold)
                                Text((modelState as ModelState.Error).message, color = Color.White)
                                Spacer(Modifier.height(16.dp))
                                Button(onClick = { viewModel.loadModel() }) {
                                    Text("Retry")
                                }
                            }
                        }
                    }
                    // Chat UI
                    else -> {
                        ChatScreen(
                            messages = messages,
                            currentResponse = currentResponse,
                            isGenerating = isGenerating,
                            voiceState = voiceState,
                            ttsReady = ttsReady,
                            onSend = { viewModel.sendMessage(it) },
                            onMicClick = { requestMicAndListen() },
                            onSpeakClick = { hindiTTS.speak(it) }
                        )
                    }
                }
            }
        }
    }

    @Composable
    fun DownloadScreen(
        downloadState: ModelManager.DownloadState,
        onDownload: () -> Unit
    ) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.padding(32.dp)
            ) {
                Text(
                    "Nano Hindi",
                    fontSize = 32.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFFFF9800)
                )
                Text(
                    "by Vizuara",
                    fontSize = 16.sp,
                    color = Color(0xFFB0B0C0)
                )
                Spacer(Modifier.height(32.dp))
                Text(
                    "Hindi AI assistant that runs completely offline on your device.",
                    textAlign = TextAlign.Center,
                    color = Color.White,
                    fontSize = 14.sp
                )
                Spacer(Modifier.height(32.dp))

                when (downloadState) {
                    is ModelManager.DownloadState.NotStarted -> {
                        Text(
                            "Model size: ~968 MB",
                            color = Color(0xFFB0B0C0),
                            fontSize = 13.sp
                        )
                        Spacer(Modifier.height(16.dp))
                        Button(
                            onClick = onDownload,
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFF9800))
                        ) {
                            Text("Download Model", color = Color.Black, fontWeight = FontWeight.Bold)
                        }
                    }
                    is ModelManager.DownloadState.Downloading -> {
                        LinearProgressIndicator(
                            progress = { downloadState.progress },
                            modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp)),
                            color = Color(0xFFFF9800),
                            trackColor = Color(0xFF252547)
                        )
                        Spacer(Modifier.height(8.dp))
                        Text(
                            "%.0f / %.0f MB (%.0f%%)".format(
                                downloadState.downloadedMB,
                                downloadState.totalMB,
                                downloadState.progress * 100
                            ),
                            color = Color(0xFFB0B0C0),
                            fontSize = 13.sp
                        )
                    }
                    is ModelManager.DownloadState.Completed -> {
                        Text("Download complete!", color = Color(0xFF4CAF50))
                        Spacer(Modifier.height(8.dp))
                        CircularProgressIndicator(color = Color(0xFFFF9800), modifier = Modifier.size(24.dp))
                        Spacer(Modifier.height(8.dp))
                        Text("Loading model...", color = Color(0xFFB0B0C0), fontSize = 13.sp)
                    }
                    is ModelManager.DownloadState.Error -> {
                        Text("Error: ${downloadState.message}", color = Color(0xFFF44336))
                        Spacer(Modifier.height(16.dp))
                        Button(onClick = onDownload) {
                            Text("Retry")
                        }
                    }
                }
            }
        }
    }

    @Composable
    fun ChatScreen(
        messages: List<ChatMessage>,
        currentResponse: String,
        isGenerating: Boolean,
        voiceState: VoiceInput.VoiceState,
        ttsReady: Boolean,
        onSend: (String) -> Unit,
        onMicClick: () -> Unit,
        onSpeakClick: (String) -> Unit
    ) {
        var inputText by remember { mutableStateOf("") }
        val listState = rememberLazyListState()
        val scope = rememberCoroutineScope()

        // Auto-scroll to bottom
        LaunchedEffect(messages.size, currentResponse) {
            if (messages.isNotEmpty() || currentResponse.isNotEmpty()) {
                listState.animateScrollToItem(
                    (messages.size + if (currentResponse.isNotEmpty()) 1 else 0).coerceAtLeast(0)
                )
            }
        }

        Column(Modifier.fillMaxSize()) {
            // Messages list
            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                contentPadding = PaddingValues(vertical = 8.dp)
            ) {
                if (messages.isEmpty() && currentResponse.isEmpty()) {
                    item {
                        Box(Modifier.fillMaxWidth().padding(top = 64.dp), contentAlignment = Alignment.Center) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("🙏", fontSize = 48.sp)
                                Spacer(Modifier.height(8.dp))
                                Text("नमस्ते!", fontSize = 24.sp, color = Color(0xFFFF9800), fontWeight = FontWeight.Bold)
                                Spacer(Modifier.height(4.dp))
                                Text(
                                    "Hindi me kuch bhi puchiye!",
                                    color = Color(0xFFB0B0C0),
                                    fontSize = 14.sp
                                )
                            }
                        }
                    }
                }

                items(messages) { message ->
                    MessageBubble(
                        message = message,
                        ttsReady = ttsReady,
                        onSpeakClick = onSpeakClick
                    )
                }

                // Show current streaming response
                if (currentResponse.isNotEmpty()) {
                    item {
                        MessageBubble(
                            message = ChatMessage(text = currentResponse, isUser = false),
                            ttsReady = false,
                            onSpeakClick = {},
                            isStreaming = true
                        )
                    }
                }
            }

            // Input bar
            Surface(
                color = Color(0xFF1A1A2E),
                tonalElevation = 8.dp
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Mic button
                    IconButton(
                        onClick = onMicClick,
                        modifier = Modifier
                            .size(44.dp)
                            .background(
                                if (voiceState is VoiceInput.VoiceState.Listening) Color(0xFFF44336) else Color(0xFF252547),
                                CircleShape
                            )
                    ) {
                        Icon(
                            if (voiceState is VoiceInput.VoiceState.Listening) Icons.Default.MicOff else Icons.Default.Mic,
                            contentDescription = "Voice input",
                            tint = Color.White,
                            modifier = Modifier.size(20.dp)
                        )
                    }

                    Spacer(Modifier.width(8.dp))

                    // Text input
                    OutlinedTextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        modifier = Modifier.weight(1f),
                        placeholder = { Text("Hindi me type karo...", color = Color(0xFF606080)) },
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White,
                            focusedBorderColor = Color(0xFFFF9800),
                            unfocusedBorderColor = Color(0xFF404060),
                            cursorColor = Color(0xFFFF9800),
                        ),
                        shape = RoundedCornerShape(24.dp),
                        singleLine = false,
                        maxLines = 4,
                        enabled = !isGenerating
                    )

                    Spacer(Modifier.width(8.dp))

                    // Send button
                    IconButton(
                        onClick = {
                            if (inputText.isNotBlank() && !isGenerating) {
                                onSend(inputText.trim())
                                inputText = ""
                            }
                        },
                        enabled = inputText.isNotBlank() && !isGenerating,
                        modifier = Modifier
                            .size(44.dp)
                            .background(
                                if (inputText.isNotBlank() && !isGenerating) Color(0xFFFF9800) else Color(0xFF252547),
                                CircleShape
                            )
                    ) {
                        Icon(
                            Icons.AutoMirrored.Filled.Send,
                            contentDescription = "Send",
                            tint = if (inputText.isNotBlank() && !isGenerating) Color.Black else Color(0xFF606080),
                            modifier = Modifier.size(20.dp)
                        )
                    }
                }
            }
        }
    }

    @Composable
    fun MessageBubble(
        message: ChatMessage,
        ttsReady: Boolean,
        onSpeakClick: (String) -> Unit,
        isStreaming: Boolean = false
    ) {
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = if (message.isUser) Alignment.End else Alignment.Start
        ) {
            Surface(
                shape = RoundedCornerShape(
                    topStart = 16.dp, topEnd = 16.dp,
                    bottomStart = if (message.isUser) 16.dp else 4.dp,
                    bottomEnd = if (message.isUser) 4.dp else 16.dp
                ),
                color = if (message.isUser) Color(0xFFFF9800) else Color(0xFF252547),
                modifier = Modifier.widthIn(max = 300.dp)
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text(
                        text = message.text,
                        color = if (message.isUser) Color.Black else Color.White,
                        fontSize = 15.sp
                    )

                    // TTS button and speed info for assistant messages
                    if (!message.isUser && !isStreaming) {
                        Spacer(Modifier.height(4.dp))
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            if (ttsReady) {
                                IconButton(
                                    onClick = { onSpeakClick(message.text) },
                                    modifier = Modifier.size(28.dp)
                                ) {
                                    Icon(
                                        Icons.Default.VolumeUp,
                                        contentDescription = "Speak",
                                        tint = Color(0xFFFF9800),
                                        modifier = Modifier.size(16.dp)
                                    )
                                }
                            }
                            if (message.tokensPerSec > 0) {
                                Text(
                                    "%.1f tok/s".format(message.tokensPerSec),
                                    fontSize = 11.sp,
                                    color = Color(0xFF808090)
                                )
                            }
                        }
                    }

                    // Streaming indicator
                    if (isStreaming) {
                        Spacer(Modifier.height(4.dp))
                        Text("●", color = Color(0xFFFF9800), fontSize = 12.sp)
                    }
                }
            }
        }
    }

    private fun requestMicAndListen() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED
        ) {
            voiceInput.startListening()
        } else {
            micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }
}
