/*
 * run.c — nano_hindi C inference engine
 *
 * Pure C implementation for running the 254M nano_hindi model on Android.
 * Adapted from llama2.c by Andrej Karpathy for the nano_hindi architecture.
 *
 * Key differences from llama2.c:
 *   1. RMSNorm has NO learnable parameters (just x / sqrt(mean(x²) + eps))
 *   2. QK normalization: per-head RMSNorm on Q and K AFTER RoPE
 *   3. Half-split RoPE: rotates (x[i], x[i+half]) not (x[2i], x[2i+1])
 *   4. ReLU² MLP: relu(x)² with 2 weight matrices (not SwiGLU with 3)
 *   5. Per-layer scalar mixing: x = λ_r * x + λ_0 * x0 before each block
 *   6. Sliding window attention: SSSL pattern (3 short, 1 long, repeat)
 *   7. Logit softcap: 15 * tanh(logits / 15)
 *   8. Tied embeddings: logits = x @ wte.T (no separate lm_head)
 *   9. GQA: 12 query heads, 4 KV heads (ratio 3:1)
 *  10. No bias in any linear layer
 */

#include "run.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Platform-specific includes for mmap
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>  // CommandLineToArgvW
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------
// Math helpers

static void rmsnorm(float* o, const float* x, int size) {
    // RMSNorm with NO learnable parameters: o = x / sqrt(mean(x²) + eps)
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-6f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = x[j] * ss;
    }
}

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

static void matmul(float* xout, const float* x, const float* w, int n, int d) {
    // W (d, n) @ x (n,) = xout (d,)
    // xout[i] = sum_j W[i*n + j] * x[j]
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const float* wi = w + i * n;
        for (int j = 0; j < n; j++) {
            val += wi[j] * x[j];
        }
        xout[i] = val;
    }
}

// ----------------------------------------------------------------------------
// Memory allocation helpers

static void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = p->n_kv_heads * (p->dim / p->n_heads);
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(p->dim, sizeof(float));
    s->xb2 = (float*)calloc(p->dim, sizeof(float));
    s->x0 = (float*)calloc(p->dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->q = (float*)calloc(p->dim, sizeof(float));
    s->k = (float*)calloc(kv_dim, sizeof(float));
    s->v = (float*)calloc(kv_dim, sizeof(float));
    s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)calloc(p->padded_vocab_size, sizeof(float));
    s->key_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->x0 || !s->hb || !s->q ||
        !s->k || !s->v || !s->att || !s->logits ||
        !s->key_cache || !s->value_cache) {
        fprintf(stderr, "malloc failed for RunState!\n");
        exit(EXIT_FAILURE);
    }
}

static void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->x0);
    free(s->hb);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Model loading via memory mapping

static void memory_map_weights(TransformerWeights* w, Config* p, float* ptr) {
    int head_dim = p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_dim;

    w->token_embedding_table = ptr;
    ptr += p->padded_vocab_size * p->dim;

    w->wq = ptr;
    ptr += p->n_layers * p->dim * p->dim;

    w->wk = ptr;
    ptr += p->n_layers * kv_dim * p->dim;

    w->wv = ptr;
    ptr += p->n_layers * kv_dim * p->dim;

    w->wo = ptr;
    ptr += p->n_layers * p->dim * p->dim;

    w->w_fc = ptr;
    ptr += p->n_layers * p->hidden_dim * p->dim;

    w->w_proj = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;

    w->resid_lambdas = ptr;
    ptr += p->n_layers;

    w->x0_lambdas = ptr;
    ptr += p->n_layers;
}

void build_transformer(Transformer* t, const char* checkpoint_path) {
    // Read config header
    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->config, sizeof(int), 8, file) != 8) {
        fprintf(stderr, "Failed to read config header\n");
        exit(EXIT_FAILURE);
    }
    fclose(file);

    Config* p = &t->config;
    fprintf(stderr, "Model config: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq=%d padded=%d\n",
            p->dim, p->hidden_dim, p->n_layers, p->n_heads, p->n_kv_heads,
            p->vocab_size, p->seq_len, p->padded_vocab_size);

    // Calculate expected file size
    int head_dim = p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_dim;
    size_t expected = 32; // header
    expected += (size_t)p->padded_vocab_size * p->dim * sizeof(float);
    expected += (size_t)p->n_layers * p->dim * p->dim * sizeof(float);       // wq
    expected += (size_t)p->n_layers * kv_dim * p->dim * sizeof(float);       // wk
    expected += (size_t)p->n_layers * kv_dim * p->dim * sizeof(float);       // wv
    expected += (size_t)p->n_layers * p->dim * p->dim * sizeof(float);       // wo
    expected += (size_t)p->n_layers * p->hidden_dim * p->dim * sizeof(float);// w_fc
    expected += (size_t)p->n_layers * p->dim * p->hidden_dim * sizeof(float);// w_proj
    expected += (size_t)p->n_layers * sizeof(float);                         // resid_lambdas
    expected += (size_t)p->n_layers * sizeof(float);                         // x0_lambdas

    // Memory map the file
#ifdef _WIN32
    HANDLE hFile = CreateFileA(checkpoint_path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "CreateFile failed for %s\n", checkpoint_path);
        exit(EXIT_FAILURE);
    }
    LARGE_INTEGER fileSize;
    GetFileSizeEx(hFile, &fileSize);
    t->file_size = (size_t)fileSize.QuadPart;

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        fprintf(stderr, "CreateFileMapping failed\n");
        CloseHandle(hFile);
        exit(EXIT_FAILURE);
    }
    t->data = (float*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!t->data) {
        fprintf(stderr, "MapViewOfFile failed\n");
        CloseHandle(hMapping);
        CloseHandle(hFile);
        exit(EXIT_FAILURE);
    }
    CloseHandle(hMapping);
    CloseHandle(hFile);
    t->fd = -1; // not used on Windows
#else
    t->fd = open(checkpoint_path, O_RDONLY);
    if (t->fd == -1) {
        fprintf(stderr, "open failed for %s\n", checkpoint_path);
        exit(EXIT_FAILURE);
    }
    t->file_size = lseek(t->fd, 0, SEEK_END);
    t->data = (float*)mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
    if (t->data == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        close(t->fd);
        exit(EXIT_FAILURE);
    }
#endif

    if (t->file_size != expected) {
        fprintf(stderr, "File size mismatch: got %zu, expected %zu\n", t->file_size, expected);
        exit(EXIT_FAILURE);
    }

    // Point weights into mmap'd data (skip 32-byte header = 8 floats)
    float* weights_ptr = t->data + 8; // 8 ints = 8 floats (both 32 bytes)
    memory_map_weights(&t->weights, &t->config, weights_ptr);

    // Allocate run state buffers
    malloc_run_state(&t->state, &t->config);

    fprintf(stderr, "Model loaded successfully (%zu MB)\n", t->file_size / (1024 * 1024));
}

void free_transformer(Transformer* t) {
    free_run_state(&t->state);
#ifdef _WIN32
    if (t->data) UnmapViewOfFile(t->data);
#else
    if (t->data != MAP_FAILED) munmap(t->data, t->file_size);
    if (t->fd != -1) close(t->fd);
#endif
}

void reset_kv_cache(Transformer* t) {
    int head_dim = t->config.dim / t->config.n_heads;
    int kv_dim = t->config.n_kv_heads * head_dim;
    size_t cache_size = (size_t)t->config.n_layers * t->config.seq_len * kv_dim;
    memset(t->state.key_cache, 0, cache_size * sizeof(float));
    memset(t->state.value_cache, 0, cache_size * sizeof(float));
}

// ----------------------------------------------------------------------------
// Forward pass — the core inference logic

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_dim = dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads; // 3 (query heads per KV head)
    int half_dim = head_dim / 2; // 32 (for half-split RoPE)

    // 1. Token embedding lookup
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(s->x, content_row, dim * sizeof(float));

    // 2. Post-embedding RMSNorm (no learnable params)
    rmsnorm(s->x, s->x, dim);

    // 3. Save x0 for per-layer scalar mixing
    memcpy(s->x0, s->x, dim * sizeof(float));

    // 4. Forward through all transformer layers
    for (int l = 0; l < p->n_layers; l++) {

        // 4a. Per-layer scalar mixing: x = λ_r * x + λ_0 * x0
        float lambda_r = w->resid_lambdas[l];
        float lambda_0 = w->x0_lambdas[l];
        for (int j = 0; j < dim; j++) {
            s->x[j] = lambda_r * s->x[j] + lambda_0 * s->x0[j];
        }

        // 4b. Pre-attention RMSNorm
        rmsnorm(s->xb, s->x, dim);

        // 4c. QKV projections
        //   q = xb @ wq[l].T  →  (dim,)
        //   k = xb @ wk[l].T  →  (kv_dim,)
        //   v = xb @ wv[l].T  →  (kv_dim,)
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * kv_dim * dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * kv_dim * dim, dim, kv_dim);

        // 4d. Apply RoPE (half-split style)
        // For each head, rotate (x[i], x[i+half]) pairs
        // Query heads
        for (int h = 0; h < p->n_heads; h++) {
            float* qh = s->q + h * head_dim;
            for (int i = 0; i < half_dim; i++) {
                float freq = 1.0f / powf(10000.0f, (float)(2 * i) / (float)head_dim);
                float angle = (float)pos * freq;
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);
                float x1 = qh[i];
                float x2 = qh[i + half_dim];
                qh[i]            = x1 * cos_val + x2 * sin_val;
                qh[i + half_dim] = -x1 * sin_val + x2 * cos_val;
            }
        }
        // Key heads
        for (int h = 0; h < p->n_kv_heads; h++) {
            float* kh = s->k + h * head_dim;
            for (int i = 0; i < half_dim; i++) {
                float freq = 1.0f / powf(10000.0f, (float)(2 * i) / (float)head_dim);
                float angle = (float)pos * freq;
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);
                float x1 = kh[i];
                float x2 = kh[i + half_dim];
                kh[i]            = x1 * cos_val + x2 * sin_val;
                kh[i + half_dim] = -x1 * sin_val + x2 * cos_val;
            }
        }

        // 4e. QK normalization — per-head RMSNorm on Q and K (AFTER RoPE)
        for (int h = 0; h < p->n_heads; h++) {
            rmsnorm(s->q + h * head_dim, s->q + h * head_dim, head_dim);
        }
        for (int h = 0; h < p->n_kv_heads; h++) {
            rmsnorm(s->k + h * head_dim, s->k + h * head_dim, head_dim);
        }

        // 4f. Store K,V in cache at current position
        int loff = l * p->seq_len * kv_dim; // layer offset into cache
        float* kcache_pos = s->key_cache + loff + pos * kv_dim;
        float* vcache_pos = s->value_cache + loff + pos * kv_dim;
        memcpy(kcache_pos, s->k, kv_dim * sizeof(float));
        memcpy(vcache_pos, s->v, kv_dim * sizeof(float));

        // 4g. Attention with sliding window and GQA
        // Determine window size: SSSL pattern
        // S (512) for layers where l%4 != 3, L (1024) for l%4 == 3
        // Final layer always gets full context
        int window = (l % 4 == 3 || l == p->n_layers - 1) ? p->seq_len : p->seq_len / 2;
        int start = pos - window + 1;
        if (start < 0) start = 0;

        // Multi-head attention with GQA
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float* qh = s->q + h * head_dim;
            int kv_h = h / kv_mul; // which KV head this query head uses
            float* att = s->att + h * p->seq_len;
            float scale = 1.0f / sqrtf((float)head_dim);

            // Compute attention scores for positions in window
            for (int t = start; t <= pos; t++) {
                float* kh = s->key_cache + loff + t * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int j = 0; j < head_dim; j++) {
                    score += qh[j] * kh[j];
                }
                att[t] = score * scale;
            }

            // Softmax over [start, pos]
            softmax(att + start, pos - start + 1);

            // Weighted sum of values → xb (reuse as attention output buffer)
            float* oh = s->xb + h * head_dim;
            memset(oh, 0, head_dim * sizeof(float));
            for (int t = start; t <= pos; t++) {
                float* vh = s->value_cache + loff + t * kv_dim + kv_h * head_dim;
                float a = att[t];
                for (int j = 0; j < head_dim; j++) {
                    oh[j] += a * vh[j];
                }
            }
        }

        // 4h. Output projection + residual
        // xb2 = wo[l] @ xb (attention output)
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        for (int j = 0; j < dim; j++) {
            s->x[j] += s->xb2[j];
        }

        // 4i. Pre-MLP RMSNorm
        rmsnorm(s->xb, s->x, dim);

        // 4j. MLP: ReLU² activation
        // hb = xb @ w_fc[l].T  →  (hidden_dim,)
        matmul(s->hb, s->xb, w->w_fc + l * hidden_dim * dim, dim, hidden_dim);
        // ReLU²: relu(x) * relu(x)
        for (int j = 0; j < hidden_dim; j++) {
            float val = s->hb[j];
            val = val > 0.0f ? val : 0.0f; // ReLU
            s->hb[j] = val * val;          // square
        }
        // xb2 = w_proj[l] @ hb  →  (dim,)
        matmul(s->xb2, s->hb, w->w_proj + l * dim * hidden_dim, hidden_dim, dim);
        // Residual
        for (int j = 0; j < dim; j++) {
            s->x[j] += s->xb2[j];
        }
    }

    // 5. Final RMSNorm
    rmsnorm(s->x, s->x, dim);

    // 6. Compute logits using tied embeddings: logits = x @ wte.T
    // wte is (padded_vocab_size, dim), treated as weight matrix
    matmul(s->logits, s->x, w->token_embedding_table, dim, p->padded_vocab_size);

    // 7. Logit softcap: 15 * tanh(logits / 15)
    // Only apply to actual vocab (not padding), then zero out padding
    for (int i = 0; i < p->vocab_size; i++) {
        s->logits[i] = 15.0f * tanhf(s->logits[i] / 15.0f);
    }
    for (int i = p->vocab_size; i < p->padded_vocab_size; i++) {
        s->logits[i] = -1e9f; // mask padding tokens
    }

    return s->logits;
}

// ----------------------------------------------------------------------------
// Tokenizer — BPE encode/decode for Sarvam-1

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->vocab_lengths = (int*)malloc(vocab_size * sizeof(int));
    t->sorted_indices = NULL; // built lazily on first encode

    // Read tokenizer binary file
    FILE* file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open tokenizer file %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }

    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Failed to read max_token_length\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < vocab_size; i++) {
        float score;
        int len;
        if (fread(&score, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "Failed to read score for token %d\n", i);
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "Failed to read len for token %d\n", i);
            exit(EXIT_FAILURE);
        }
        t->vocab_scores[i] = score;
        t->vocab_lengths[i] = len;
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], 1, len, file) != (size_t)len) {
            fprintf(stderr, "Failed to read piece for token %d\n", i);
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
    fprintf(stderr, "Tokenizer loaded: %d tokens, max_len=%d\n", vocab_size, t->max_token_length);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->vocab_lengths);
    free(t->sorted_indices);
}

// Comparison function for sorted token lookup
static Tokenizer* _sort_tokenizer; // used by comparator
static int compare_tokens(const void* a, const void* b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    return strcmp(_sort_tokenizer->vocab[ia], _sort_tokenizer->vocab[ib]);
}

static void build_sorted_index(Tokenizer* t) {
    t->sorted_indices = (int*)malloc(t->vocab_size * sizeof(int));
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_indices[i] = i;
    }
    _sort_tokenizer = t;
    qsort(t->sorted_indices, t->vocab_size, sizeof(int), compare_tokens);
}

static int str_lookup(const char* str, Tokenizer* t) {
    // Binary search for string in sorted vocabulary
    if (t->sorted_indices == NULL) {
        build_sorted_index(t);
    }
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int idx = t->sorted_indices[mid];
        int cmp = strcmp(str, t->vocab[idx]);
        if (cmp == 0) return idx;
        if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

char* decode(Tokenizer* t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) return "";
    char* piece = t->vocab[token];

    // SentencePiece: ▁ (U+2581, 3 bytes: 0xE2 0x96 0x81) represents space
    // If piece starts with ▁ and it's not the first token, replace with space
    // We handle this in the caller/JNI layer for simplicity
    return piece;
}

void encode(Tokenizer* t, const char* text, int bos, int eos,
            int* tokens, int* n_tokens) {
    // BPE encoding with SentencePiece preprocessing.
    // SentencePiece converts: spaces → ▁ (U+2581, 3 bytes: 0xE2 0x96 0x81)
    // and prepends ▁ to the beginning of the text.
    if (text == NULL) { *n_tokens = 0; return; }

    // Build sorted index if not done yet
    if (t->sorted_indices == NULL) {
        build_sorted_index(t);
    }

    *n_tokens = 0;

    // Add BOS token if requested
    if (bos) {
        tokens[(*n_tokens)++] = 1; // BOS token id
    }

    // If empty text, just return BOS
    if (*text == '\0') {
        if (eos) tokens[(*n_tokens)++] = 2; // EOS token id
        return;
    }

    // SentencePiece preprocessing: prepend ▁ and replace spaces with ▁
    // ▁ = 0xE2 0x96 0x81 (3 bytes)
    size_t text_len = strlen(text);
    // Worst case: every char is a space → each space (1 byte) becomes ▁ (3 bytes), plus leading ▁
    char* sp_text = (char*)malloc(text_len * 3 + 4);
    int sp_pos = 0;
    // Prepend ▁
    sp_text[sp_pos++] = (char)0xE2;
    sp_text[sp_pos++] = (char)0x96;
    sp_text[sp_pos++] = (char)0x81;

    for (size_t i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
            // Replace space with ▁
            sp_text[sp_pos++] = (char)0xE2;
            sp_text[sp_pos++] = (char)0x96;
            sp_text[sp_pos++] = (char)0x81;
        } else {
            sp_text[sp_pos++] = text[i];
        }
    }
    sp_text[sp_pos] = '\0';

    // Temporary buffer for building merge candidates
    char* str_buffer = (char*)malloc((t->max_token_length * 2 + 3) * sizeof(char));

    // First pass: encode each UTF-8 character as an individual token
    // UTF-8 character lengths: 1 byte (0xxxxxxx), 2 (110xxxxx), 3 (1110xxxx), 4 (11110xxx)
    const char* ptr = sp_text;
    while (*ptr != '\0') {
        int char_len = 1;
        unsigned char c = (unsigned char)*ptr;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        // Copy this UTF-8 character to buffer and look up
        memcpy(str_buffer, ptr, char_len);
        str_buffer[char_len] = '\0';

        int id = str_lookup(str_buffer, t);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            // Fallback: encode as individual bytes using byte tokens
            for (int b = 0; b < char_len; b++) {
                unsigned char byte = (unsigned char)ptr[b];
                snprintf(str_buffer, t->max_token_length + 1, "<0x%02X>", byte);
                id = str_lookup(str_buffer, t);
                if (id != -1) {
                    tokens[(*n_tokens)++] = id;
                }
            }
        }
        ptr += char_len;
    }

    free(sp_text);

    // Second pass: BPE merge loop
    // Repeatedly find the highest-scoring adjacent pair and merge them
    while (1) {
        float best_score = -1e10f;
        int best_idx = -1;
        int best_id = -1;

        for (int i = 0; i < (*n_tokens) - 1; i++) {
            // Build merged string from tokens[i] and tokens[i+1]
            snprintf(str_buffer, t->max_token_length * 2 + 2, "%s%s",
                     t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);

            int id = str_lookup(str_buffer, t);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_idx = i;
                best_id = id;
            }
        }

        if (best_idx == -1) break; // no more merges possible

        // Merge: replace tokens[best_idx] with merged token, shift rest left
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens) - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;
    }

    // Add EOS token if requested
    if (eos) {
        tokens[(*n_tokens)++] = 2; // EOS token id
    }

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler — top-k sampling with temperature

static unsigned int random_u32(unsigned long long* state) {
    // xorshift64
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (unsigned int)((*state * 0x2545F4914F6CDD1DULL) >> 32);
}

static float random_f32(unsigned long long* state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int compare_prob_index(const void* a, const void* b) {
    const ProbIndex* pa = (const ProbIndex*)a;
    const ProbIndex* pb = (const ProbIndex*)b;
    if (pa->prob > pb->prob) return -1;
    if (pa->prob < pb->prob) return 1;
    return 0;
}

void build_sampler(Sampler* s, float temperature, int top_k, unsigned long long rng_seed) {
    s->temperature = temperature;
    s->top_k = top_k;
    s->rng_state = rng_seed;
}

int sample(Sampler* s, float* logits, int vocab_size) {
    // Greedy (temperature = 0)
    if (s->temperature == 0.0f) {
        int max_i = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_i = i;
            }
        }
        return max_i;
    }

    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= s->temperature;
    }

    // Softmax
    softmax(logits, vocab_size);

    // Top-k: keep only top-k tokens by probability
    int n = vocab_size;
    int k = s->top_k;
    if (k > 0 && k < n) {
        // Partial sort: find the k-th largest probability
        // Use a simple selection approach for small k
        ProbIndex* probindex = (ProbIndex*)malloc(n * sizeof(ProbIndex));
        for (int i = 0; i < n; i++) {
            probindex[i].prob = logits[i];
            probindex[i].index = i;
        }
        // Sort descending by probability
        qsort(probindex, n, sizeof(ProbIndex), compare_prob_index);

        // Zero out everything below top-k
        float cutoff = probindex[k - 1].prob;
        for (int i = 0; i < n; i++) {
            if (logits[i] < cutoff) {
                logits[i] = 0.0f;
            }
        }
        free(probindex);

        // Re-normalize
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += logits[i];
        for (int i = 0; i < n; i++) logits[i] /= sum;
    }

    // Sample from the distribution
    float coin = random_f32(&s->rng_state);
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += logits[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // fallback
}

// ----------------------------------------------------------------------------
// Chat template helpers for nano_hindi
// Template: <s>### उपयोगकर्ता:\n{user}\n\n### सहायक:\n{assistant}</s>

// UTF-8 constants for chat markers
static const char* USER_MARKER = "### \xe0\xa4\x89\xe0\xa4\xaa\xe0\xa4\xaf\xe0\xa5\x8b\xe0\xa4\x97\xe0\xa4\x95\xe0\xa4\xb0\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa4\xbe:\n";
static const char* ASSISTANT_MARKER = "### \xe0\xa4\xb8\xe0\xa4\xb9\xe0\xa4\xbe\xe0\xa4\xaf\xe0\xa4\x95:\n";

int encode_chat_prompt(Tokenizer* t, const char* user_message, int* tokens) {
    // Build the full prompt string first, then encode all at once.
    // This ensures BPE merges happen correctly across boundaries.
    // Format: USER_MARKER + user_message + "\n\n" + ASSISTANT_MARKER
    size_t um_len = strlen(USER_MARKER);
    size_t msg_len = strlen(user_message);
    size_t am_len = strlen(ASSISTANT_MARKER);
    size_t total_len = um_len + msg_len + 2 + am_len + 1; // +2 for "\n\n", +1 for null

    char* full_prompt = (char*)malloc(total_len);
    memcpy(full_prompt, USER_MARKER, um_len);
    memcpy(full_prompt + um_len, user_message, msg_len);
    full_prompt[um_len + msg_len] = '\n';
    full_prompt[um_len + msg_len + 1] = '\n';
    memcpy(full_prompt + um_len + msg_len + 2, ASSISTANT_MARKER, am_len);
    full_prompt[um_len + msg_len + 2 + am_len] = '\0';

    int n_tokens;
    encode(t, full_prompt, 1, 0, tokens, &n_tokens); // BOS=1, EOS=0

    free(full_prompt);
    return n_tokens;
}

// ----------------------------------------------------------------------------
// Timing utility (used by both PC main and JNI bridge)

static long time_in_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (long)(counter.QuadPart * 1000 / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
#endif
}

// ----------------------------------------------------------------------------
// Main entry point (for PC testing)

#ifndef __ANDROID__
#ifndef NO_MAIN

int main(int argc, char* argv[]) {
    // On Windows, argv uses the system codepage (not UTF-8), which mangles
    // Hindi/Devanagari text. Re-parse from the wide-char command line.
#ifdef _WIN32
    SetConsoleOutputCP(65001);  // UTF-8 output
    SetConsoleCP(65001);        // UTF-8 input (for --chat mode)

    int wargc;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv) {
        // Allocate UTF-8 argv
        char** utf8_argv = (char**)malloc((wargc + 1) * sizeof(char*));
        for (int i = 0; i < wargc; i++) {
            int needed = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
            utf8_argv[i] = (char*)malloc(needed);
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, utf8_argv[i], needed, NULL, NULL);
        }
        utf8_argv[wargc] = NULL;
        LocalFree(wargv);
        argc = wargc;
        argv = utf8_argv;
    }
#endif

    // Default parameters
    char* checkpoint_path = NULL;
    char* tokenizer_path = NULL;
    float temperature = 0.7f;
    int top_k = 40;
    int max_tokens = 256;
    char* prompt = NULL;
    int chat_mode = 0;
    unsigned long long rng_seed = 0;

    // Parse command line
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc && strcmp(argv[i], "--chat") != 0) {
            fprintf(stderr, "Missing argument for %s\n", argv[i]);
            return 1;
        }
        if (strcmp(argv[i], "--model") == 0) checkpoint_path = argv[i + 1];
        else if (strcmp(argv[i], "--tokenizer") == 0) tokenizer_path = argv[i + 1];
        else if (strcmp(argv[i], "--temp") == 0) temperature = atof(argv[i + 1]);
        else if (strcmp(argv[i], "--top_k") == 0) top_k = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--max_tokens") == 0) max_tokens = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--prompt") == 0) prompt = argv[i + 1];
        else if (strcmp(argv[i], "--seed") == 0) rng_seed = atoll(argv[i + 1]);
        else if (strcmp(argv[i], "--chat") == 0) { chat_mode = 1; i--; }
    }

    if (checkpoint_path == NULL) {
        fprintf(stderr, "Usage: %s --model <checkpoint.bin> --tokenizer <tokenizer.bin> [options]\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --prompt <text>     Input prompt\n");
        fprintf(stderr, "  --chat              Interactive chat mode\n");
        fprintf(stderr, "  --temp <float>      Temperature (default: 0.7)\n");
        fprintf(stderr, "  --top_k <int>       Top-k sampling (default: 40)\n");
        fprintf(stderr, "  --max_tokens <int>  Max tokens to generate (default: 256)\n");
        fprintf(stderr, "  --seed <int>        RNG seed (default: time-based)\n");
        return 1;
    }

    if (rng_seed == 0) rng_seed = (unsigned long long)time(NULL);

    // Load model and tokenizer
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, temperature, top_k, rng_seed);

    if (chat_mode) {
        // Interactive chat loop
        char input_buffer[1024];
        int* prompt_tokens = (int*)malloc(transformer.config.seq_len * sizeof(int));

        fprintf(stderr, "\n=== Nano Hindi Chat ===\n");
        fprintf(stderr, "Type your message in Hindi (or English). Type 'quit' to exit.\n\n");

        while (1) {
            fprintf(stdout, "You: ");
            fflush(stdout);
            if (fgets(input_buffer, sizeof(input_buffer), stdin) == NULL) break;

            // Strip newline
            int len = (int)strlen(input_buffer);
            while (len > 0 && (input_buffer[len - 1] == '\n' || input_buffer[len - 1] == '\r'))
                input_buffer[--len] = '\0';

            if (strcmp(input_buffer, "quit") == 0 || strcmp(input_buffer, "exit") == 0) break;
            if (len == 0) continue;

            // Reset KV cache for new turn (single-turn for simplicity)
            reset_kv_cache(&transformer);

            // Encode chat prompt
            int n_prompt_tokens = encode_chat_prompt(&tokenizer, input_buffer, prompt_tokens);

            fprintf(stdout, "Assistant: ");
            fflush(stdout);

            // Process prompt tokens
            long start = time_in_ms();
            int pos = 0;
            int next_token;
            int token = prompt_tokens[0];

            for (pos = 0; pos < n_prompt_tokens; pos++) {
                forward(&transformer, prompt_tokens[pos], pos);
            }

            // The last forward pass gave us logits, sample from them
            float* logits = transformer.state.logits;
            next_token = sample(&sampler, logits, transformer.config.vocab_size);

            int gen_count = 0;
            while (pos < transformer.config.seq_len && gen_count < max_tokens) {
                if (next_token == 2) break; // EOS — stop before printing </s>

                // Decode and print token
                char* piece = decode(&tokenizer, token, next_token);
                // Handle SentencePiece ▁ → space
                if (piece[0] == '\xe2' && piece[1] == '\x96' && piece[2] == '\x81') {
                    fprintf(stdout, " %s", piece + 3);
                } else {
                    fprintf(stdout, "%s", piece);
                }
                fflush(stdout);

                token = next_token;
                logits = forward(&transformer, token, pos);
                next_token = sample(&sampler, logits, transformer.config.vocab_size);
                pos++;
                gen_count++;
            }

            long end = time_in_ms();
            fprintf(stdout, "\n");
            fprintf(stderr, "[%d tokens, %.1f tok/s]\n\n",
                    gen_count, gen_count / ((end - start) / 1000.0));
        }

        free(prompt_tokens);
    } else if (prompt != NULL) {
        // Single prompt mode
        int* prompt_tokens = (int*)malloc(transformer.config.seq_len * sizeof(int));
        int n_prompt_tokens = encode_chat_prompt(&tokenizer, prompt, prompt_tokens);

        long start = time_in_ms();
        int pos = 0;
        int token;

        // Process prompt
        for (pos = 0; pos < n_prompt_tokens; pos++) {
            forward(&transformer, prompt_tokens[pos], pos);
        }

        float* logits = transformer.state.logits;
        int next_token = sample(&sampler, logits, transformer.config.vocab_size);

        int gen_count = 0;
        token = prompt_tokens[n_prompt_tokens - 1];
        while (pos < transformer.config.seq_len && gen_count < max_tokens) {
            if (next_token == 2) break; // EOS — stop before printing </s>

            char* piece = decode(&tokenizer, token, next_token);
            if (piece[0] == '\xe2' && piece[1] == '\x96' && piece[2] == '\x81') {
                printf(" %s", piece + 3);
            } else {
                printf("%s", piece);
            }
            fflush(stdout);

            token = next_token;
            logits = forward(&transformer, token, pos);
            next_token = sample(&sampler, logits, transformer.config.vocab_size);
            pos++;
            gen_count++;
        }

        long end = time_in_ms();
        printf("\n");
        fprintf(stderr, "[%d tokens, %.1f tok/s]\n",
                gen_count, gen_count / ((end - start) / 1000.0));

        free(prompt_tokens);
    }

    // Cleanup
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}

#endif // NO_MAIN
#endif // __ANDROID__
