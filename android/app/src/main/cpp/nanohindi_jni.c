/*
 * nanohindi_jni.c — JNI bridge between C inference engine and Android/Kotlin
 *
 * Exposes 3 native functions:
 *   - loadModel(modelPath, tokenizerPath) → boolean
 *   - generate(prompt, maxTokens, temperature, topK, callback) → void
 *   - freeModel() → void
 *
 * The generate function streams tokens back to Kotlin via a TokenCallback interface.
 */

#include <jni.h>
#include <string.h>
#include <stdlib.h>
#include <android/log.h>

#define NO_MAIN
#include "run.c"

#define LOG_TAG "NanoHindi"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global model state
static Transformer g_transformer;
static Tokenizer g_tokenizer;
static int g_model_loaded = 0;

// JNI function: loadModel
JNIEXPORT jboolean JNICALL
Java_com_vizuara_nanohindi_NanoHindi_loadModel(
    JNIEnv* env, jobject thiz,
    jstring model_path, jstring tokenizer_path
) {
    if (g_model_loaded) {
        LOGI("Model already loaded, freeing first...");
        free_tokenizer(&g_tokenizer);
        free_transformer(&g_transformer);
        g_model_loaded = 0;
    }

    const char* model_cpath = (*env)->GetStringUTFChars(env, model_path, NULL);
    const char* tok_cpath = (*env)->GetStringUTFChars(env, tokenizer_path, NULL);

    LOGI("Loading model from: %s", model_cpath);
    LOGI("Loading tokenizer from: %s", tok_cpath);

    // Build transformer (mmap the model file)
    build_transformer(&g_transformer, model_cpath);

    // Build tokenizer
    build_tokenizer(&g_tokenizer, tok_cpath, g_transformer.config.vocab_size);

    (*env)->ReleaseStringUTFChars(env, model_path, model_cpath);
    (*env)->ReleaseStringUTFChars(env, tokenizer_path, tok_cpath);

    g_model_loaded = 1;
    LOGI("Model loaded successfully! dim=%d layers=%d heads=%d vocab=%d",
         g_transformer.config.dim, g_transformer.config.n_layers,
         g_transformer.config.n_heads, g_transformer.config.vocab_size);

    return JNI_TRUE;
}

// JNI function: generate
JNIEXPORT void JNICALL
Java_com_vizuara_nanohindi_NanoHindi_generate(
    JNIEnv* env, jobject thiz,
    jstring prompt, jint max_tokens, jfloat temperature, jint top_k,
    jobject callback
) {
    if (!g_model_loaded) {
        LOGE("Model not loaded!");
        return;
    }

    const char* prompt_cstr = (*env)->GetStringUTFChars(env, prompt, NULL);
    LOGI("Generating for prompt: %.50s...", prompt_cstr);

    // Get callback method IDs
    jclass callbackClass = (*env)->GetObjectClass(env, callback);
    jmethodID onTokenMethod = (*env)->GetMethodID(env, callbackClass, "onToken", "(Ljava/lang/String;)V");
    jmethodID onCompleteMethod = (*env)->GetMethodID(env, callbackClass, "onComplete", "(IF)V");

    if (!onTokenMethod || !onCompleteMethod) {
        LOGE("Failed to find callback methods!");
        (*env)->ReleaseStringUTFChars(env, prompt, prompt_cstr);
        return;
    }

    // Reset KV cache
    reset_kv_cache(&g_transformer);

    // Build sampler
    Sampler sampler;
    unsigned long long seed = (unsigned long long)time(NULL);
    build_sampler(&sampler, temperature, top_k, seed);

    // Encode chat prompt
    int* prompt_tokens = (int*)malloc(g_transformer.config.seq_len * sizeof(int));
    int n_prompt_tokens = encode_chat_prompt(&g_tokenizer, prompt_cstr, prompt_tokens);

    (*env)->ReleaseStringUTFChars(env, prompt, prompt_cstr);

    LOGI("Prompt encoded to %d tokens", n_prompt_tokens);

    // Process prompt tokens (prefill)
    int pos;
    for (pos = 0; pos < n_prompt_tokens; pos++) {
        forward(&g_transformer, prompt_tokens[pos], pos);
    }

    // Sample first generated token
    float* logits = g_transformer.state.logits;
    int next_token = sample(&sampler, logits, g_transformer.config.vocab_size);

    int gen_count = 0;
    int prev_token = prompt_tokens[n_prompt_tokens - 1];

    // Token generation buffer for building multi-byte UTF-8 strings
    char token_buffer[256];

    // Timing
    long start_ms = time_in_ms();

    while (pos < g_transformer.config.seq_len && gen_count < max_tokens) {
        // Check for EOS before decoding (don't send </s> to callback)
        if (next_token == 2) break;

        // Decode token to string
        char* piece = decode(&g_tokenizer, prev_token, next_token);

        // Handle SentencePiece ▁ → space
        if (piece[0] == '\xe2' && (unsigned char)piece[1] == 0x96 && (unsigned char)piece[2] == 0x81) {
            snprintf(token_buffer, sizeof(token_buffer), " %s", piece + 3);
        } else {
            snprintf(token_buffer, sizeof(token_buffer), "%s", piece);
        }

        // Send token to Kotlin callback
        jstring jtoken = (*env)->NewStringUTF(env, token_buffer);
        if (jtoken) {
            (*env)->CallVoidMethod(env, callback, onTokenMethod, jtoken);
            (*env)->DeleteLocalRef(env, jtoken);
        }

        // Check for exceptions (in case callback threw)
        if ((*env)->ExceptionCheck(env)) {
            LOGE("Exception in callback, stopping generation");
            (*env)->ExceptionClear(env);
            break;
        }

        prev_token = next_token;
        logits = forward(&g_transformer, prev_token, pos);
        next_token = sample(&sampler, logits, g_transformer.config.vocab_size);
        pos++;
        gen_count++;
    }

    long end_ms = time_in_ms();
    float tok_per_sec = gen_count > 0 ? gen_count / ((end_ms - start_ms) / 1000.0f) : 0;

    LOGI("Generated %d tokens in %ld ms (%.1f tok/s)", gen_count, end_ms - start_ms, tok_per_sec);

    // Notify completion
    (*env)->CallVoidMethod(env, callback, onCompleteMethod, (jint)gen_count, (jfloat)tok_per_sec);

    free(prompt_tokens);
}

// JNI function: freeModel
JNIEXPORT void JNICALL
Java_com_vizuara_nanohindi_NanoHindi_freeModel(JNIEnv* env, jobject thiz) {
    if (g_model_loaded) {
        LOGI("Freeing model...");
        free_tokenizer(&g_tokenizer);
        free_transformer(&g_transformer);
        g_model_loaded = 0;
        LOGI("Model freed");
    }
}
