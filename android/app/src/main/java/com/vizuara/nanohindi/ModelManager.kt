package com.vizuara.nanohindi

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * Manages model download from HuggingFace and local file management.
 *
 * Model file: nano_hindi_250m.bin (~968 MB)
 * Tokenizer file: tokenizer.bin (~2 MB) — bundled in res/raw/
 */
class ModelManager(private val context: Context) {

    companion object {
        private const val MODEL_FILENAME = "nano_hindi_250m.bin"
        private const val TOKENIZER_FILENAME = "tokenizer.bin"
        private const val MODEL_URL =
            "https://huggingface.co/omunaman/nano-hindi-sft/resolve/main/nano_hindi_250m.bin"
        private const val EXPECTED_MODEL_SIZE = 1014497568L // ~967.5 MB
        private const val BUFFER_SIZE = 8192
    }

    sealed class DownloadState {
        data object NotStarted : DownloadState()
        data class Downloading(val progress: Float, val downloadedMB: Float, val totalMB: Float) : DownloadState()
        data object Completed : DownloadState()
        data class Error(val message: String) : DownloadState()
    }

    private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.NotStarted)
    val downloadState: StateFlow<DownloadState> = _downloadState

    private val modelsDir: File
        get() = File(context.filesDir, "models").also { it.mkdirs() }

    val modelPath: String
        get() = File(modelsDir, MODEL_FILENAME).absolutePath

    val tokenizerPath: String
        get() = File(modelsDir, TOKENIZER_FILENAME).absolutePath

    /** Check if model is already downloaded and valid. */
    fun isModelReady(): Boolean {
        val modelFile = File(modelsDir, MODEL_FILENAME)
        val tokFile = File(modelsDir, TOKENIZER_FILENAME)
        return modelFile.exists() && modelFile.length() == EXPECTED_MODEL_SIZE && tokFile.exists()
    }

    /** Extract tokenizer from bundled res/raw resource. */
    suspend fun extractTokenizer() = withContext(Dispatchers.IO) {
        val tokFile = File(modelsDir, TOKENIZER_FILENAME)
        if (tokFile.exists()) return@withContext

        try {
            context.resources.openRawResource(R.raw.tokenizer).use { input ->
                FileOutputStream(tokFile).use { output ->
                    input.copyTo(output)
                }
            }
        } catch (e: Exception) {
            // If resource not found, tokenizer needs to be downloaded separately
            // For now, we'll handle this in the download step
        }
    }

    /** Download model from HuggingFace with progress tracking and resume support. */
    suspend fun downloadModel() = withContext(Dispatchers.IO) {
        val modelFile = File(modelsDir, MODEL_FILENAME)
        val tempFile = File(modelsDir, "$MODEL_FILENAME.tmp")

        // Check if already downloaded
        if (modelFile.exists() && modelFile.length() == EXPECTED_MODEL_SIZE) {
            _downloadState.value = DownloadState.Completed
            return@withContext
        }

        try {
            // Support resume: check how much we already have
            var downloadedBytes = if (tempFile.exists()) tempFile.length() else 0L

            val url = URL(MODEL_URL)
            val connection = url.openConnection() as HttpURLConnection
            connection.connectTimeout = 30000
            connection.readTimeout = 30000

            // Resume support via Range header
            if (downloadedBytes > 0) {
                connection.setRequestProperty("Range", "bytes=$downloadedBytes-")
            }

            connection.connect()

            val responseCode = connection.responseCode
            val totalBytes: Long

            if (responseCode == 206) {
                // Partial content — resume worked
                totalBytes = downloadedBytes + connection.contentLength
            } else if (responseCode == 200) {
                // Full content — start fresh
                downloadedBytes = 0
                totalBytes = connection.contentLength.toLong()
            } else {
                _downloadState.value = DownloadState.Error("HTTP $responseCode")
                return@withContext
            }

            val totalMB = totalBytes / (1024f * 1024f)

            _downloadState.value = DownloadState.Downloading(
                progress = downloadedBytes.toFloat() / totalBytes,
                downloadedMB = downloadedBytes / (1024f * 1024f),
                totalMB = totalMB
            )

            // Download with progress
            val inputStream = connection.inputStream
            val outputStream = FileOutputStream(tempFile, downloadedBytes > 0)
            val buffer = ByteArray(BUFFER_SIZE)

            var bytesRead: Int
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
                downloadedBytes += bytesRead

                _downloadState.value = DownloadState.Downloading(
                    progress = downloadedBytes.toFloat() / totalBytes,
                    downloadedMB = downloadedBytes / (1024f * 1024f),
                    totalMB = totalMB
                )
            }

            outputStream.close()
            inputStream.close()
            connection.disconnect()

            // Verify size and rename
            if (tempFile.length() == EXPECTED_MODEL_SIZE || totalBytes <= 0) {
                tempFile.renameTo(modelFile)
                _downloadState.value = DownloadState.Completed
            } else {
                _downloadState.value = DownloadState.Error(
                    "Size mismatch: ${tempFile.length()} != $EXPECTED_MODEL_SIZE"
                )
            }
        } catch (e: Exception) {
            _downloadState.value = DownloadState.Error(e.message ?: "Unknown error")
        }
    }

    /** Delete downloaded model to free space. */
    fun deleteModel() {
        File(modelsDir, MODEL_FILENAME).delete()
        File(modelsDir, "$MODEL_FILENAME.tmp").delete()
        _downloadState.value = DownloadState.NotStarted
    }
}
