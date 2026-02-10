package com.vizuara.nanohindi

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ChatMessage(
    val text: String,
    val isUser: Boolean,
    val tokensPerSec: Float = 0f
)

sealed class ModelState {
    data object NotLoaded : ModelState()
    data object Loading : ModelState()
    data object Ready : ModelState()
    data class Error(val message: String) : ModelState()
}

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    val modelManager = ModelManager(application)
    private val nanoHindi = NanoHindi()

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages

    private val _modelState = MutableStateFlow<ModelState>(ModelState.NotLoaded)
    val modelState: StateFlow<ModelState> = _modelState

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating

    private val _currentResponse = MutableStateFlow("")
    val currentResponse: StateFlow<String> = _currentResponse

    // Generation settings
    var temperature = 0.7f
    var topK = 40
    var maxTokens = 512

    /** Load the model into memory. Call after download is complete. */
    fun loadModel() {
        if (_modelState.value == ModelState.Loading) return

        viewModelScope.launch {
            _modelState.value = ModelState.Loading

            // Extract tokenizer from resources
            modelManager.extractTokenizer()

            withContext(Dispatchers.IO) {
                try {
                    val success = nanoHindi.loadModel(
                        modelManager.modelPath,
                        modelManager.tokenizerPath
                    )
                    if (success) {
                        _modelState.value = ModelState.Ready
                    } else {
                        _modelState.value = ModelState.Error("Failed to load model")
                    }
                } catch (e: Exception) {
                    _modelState.value = ModelState.Error(e.message ?: "Unknown error")
                }
            }
        }
    }

    /** Send a user message and generate a response. */
    fun sendMessage(userText: String) {
        if (_isGenerating.value || _modelState.value != ModelState.Ready) return
        if (userText.isBlank()) return

        // Add user message
        _messages.value = _messages.value + ChatMessage(text = userText, isUser = true)
        _currentResponse.value = ""
        _isGenerating.value = true

        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                try {
                    val responseBuilder = StringBuilder()

                    nanoHindi.generate(
                        prompt = userText,
                        maxTokens = maxTokens,
                        temperature = temperature,
                        topK = topK,
                        callback = object : TokenCallback {
                            override fun onToken(token: String) {
                                responseBuilder.append(token)
                                _currentResponse.value = responseBuilder.toString()
                            }

                            override fun onComplete(tokenCount: Int, tokensPerSec: Float) {
                                val finalText = responseBuilder.toString().trim()
                                _messages.value = _messages.value + ChatMessage(
                                    text = finalText,
                                    isUser = false,
                                    tokensPerSec = tokensPerSec
                                )
                                _currentResponse.value = ""
                                _isGenerating.value = false
                            }
                        }
                    )
                } catch (e: Exception) {
                    _messages.value = _messages.value + ChatMessage(
                        text = "Error: ${e.message}",
                        isUser = false
                    )
                    _isGenerating.value = false
                }
            }
        }
    }

    /** Clear chat history. */
    fun clearChat() {
        _messages.value = emptyList()
        _currentResponse.value = ""
    }

    override fun onCleared() {
        super.onCleared()
        try {
            nanoHindi.freeModel()
        } catch (_: Exception) {}
    }
}
