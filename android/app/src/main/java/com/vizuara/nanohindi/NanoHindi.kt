package com.vizuara.nanohindi

/**
 * JNI wrapper for the native C inference engine.
 * Loads the nanohindi shared library and exposes native methods.
 */
class NanoHindi {
    companion object {
        init {
            System.loadLibrary("nanohindi")
        }
    }

    /**
     * Load model weights and tokenizer from file paths.
     * @param modelPath Absolute path to nano_hindi_250m.bin
     * @param tokenizerPath Absolute path to tokenizer.bin
     * @return true if loaded successfully
     */
    external fun loadModel(modelPath: String, tokenizerPath: String): Boolean

    /**
     * Generate a response for the given prompt.
     * Tokens are streamed back via the callback interface.
     * This is a blocking call — run on a background thread.
     *
     * @param prompt User's message text (raw, not tokenized)
     * @param maxTokens Maximum tokens to generate
     * @param temperature Sampling temperature (0 = greedy)
     * @param topK Top-k sampling parameter
     * @param callback Receives tokens as they're generated
     */
    external fun generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        callback: TokenCallback
    )

    /**
     * Free model memory. Call when done or before loading a new model.
     */
    external fun freeModel()
}

/**
 * Callback interface for streaming token generation.
 * Implemented in Kotlin, called from C via JNI.
 */
interface TokenCallback {
    /** Called for each generated token (decoded string fragment). */
    fun onToken(token: String)

    /** Called when generation is complete.
     * @param tokenCount Total tokens generated
     * @param tokensPerSec Generation speed
     */
    fun onComplete(tokenCount: Int, tokensPerSec: Float)
}
