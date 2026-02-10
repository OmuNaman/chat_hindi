/*
 * run.h — nano_hindi C inference engine header
 *
 * Adapted from llama2.c by Andrej Karpathy, modified for nano_hindi architecture:
 *   - RMSNorm with NO learnable parameters
 *   - QK normalization (per-head RMSNorm on Q,K after RoPE)
 *   - Half-split RoPE (not adjacent pairs)
 *   - ReLU² MLP (not SwiGLU)
 *   - Per-layer scalar residual mixing (resid_lambdas, x0_lambdas)
 *   - Sliding window attention (SSSL pattern)
 *   - Logit softcap: 15 * tanh(logits / 15)
 *   - Tied embeddings (lm_head = wte.T)
 *   - GQA with 12 query heads, 4 KV heads
 *   - No bias anywhere
 */

#ifndef RUN_H
#define RUN_H

#include <stdint.h>

// ----------------------------------------------------------------------------
// Model configuration (read from binary header)

typedef struct {
    int dim;               // 768 — transformer dimension
    int hidden_dim;        // 3072 — MLP intermediate size (4 * dim)
    int n_layers;          // 32 — number of transformer blocks
    int n_heads;           // 12 — number of query heads
    int n_kv_heads;        // 4 — number of key/value heads (GQA)
    int vocab_size;        // 68096 — actual vocabulary size
    int seq_len;           // 1024 — max sequence length
    int padded_vocab_size; // 68160 — padded for alignment
} Config;

// Derived constants (computed from config)
// head_dim = dim / n_heads = 64
// kv_dim = n_kv_heads * head_dim = 256
// kv_mul = n_heads / n_kv_heads = 3

// ----------------------------------------------------------------------------
// Transformer weights (pointers into mmap'd binary file)
// No allocations needed — these point directly into the file data.

typedef struct {
    float* token_embedding_table;  // (padded_vocab_size, dim)
    float* wq;                     // (n_layers, dim, dim)
    float* wk;                     // (n_layers, kv_dim, dim)
    float* wv;                     // (n_layers, kv_dim, dim)
    float* wo;                     // (n_layers, dim, dim)
    float* w_fc;                   // (n_layers, hidden_dim, dim)
    float* w_proj;                 // (n_layers, dim, hidden_dim)
    float* resid_lambdas;          // (n_layers,)
    float* x0_lambdas;            // (n_layers,)
    // NOTE: No rms_weight — RMSNorm has no learnable params in nano_hindi
    // NOTE: No wcls — tied embeddings, logits use token_embedding_table
} TransformerWeights;

// ----------------------------------------------------------------------------
// Runtime state (allocated buffers for inference)

typedef struct {
    float* x;          // (dim,) activation at current time step
    float* xb;         // (dim,) buffer after rmsnorm / attention output
    float* xb2;        // (dim,) second buffer for residual additions
    float* x0;         // (dim,) saved x0 for per-layer scalar mixing
    float* hb;         // (hidden_dim,) hidden state in MLP
    float* q;          // (dim,) query vector (all heads concatenated)
    float* k;          // (kv_dim,) key vector at current position
    float* v;          // (kv_dim,) value vector at current position
    float* att;        // (n_heads, seq_len) attention scores per head
    float* logits;     // (padded_vocab_size,) output logits
    float* key_cache;  // (n_layers, seq_len, kv_dim) key cache
    float* value_cache;// (n_layers, seq_len, kv_dim) value cache
} RunState;

// ----------------------------------------------------------------------------
// Top-level transformer handle

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    // Memory-mapped file data
    int fd;            // file descriptor
    float* data;       // mmap'd data pointer
    size_t file_size;  // total file size in bytes
} Transformer;

// ----------------------------------------------------------------------------
// Tokenizer (BPE, loaded from tokenizer.bin)

typedef struct {
    char** vocab;          // token id -> piece string
    float* vocab_scores;   // token id -> BPE merge score
    int* vocab_lengths;    // token id -> piece byte length
    int vocab_size;
    int max_token_length;
    // Sorted index for fast string lookup during encoding
    int* sorted_indices;   // indices sorted by piece string (for bsearch)
} Tokenizer;

// ----------------------------------------------------------------------------
// Sampler (top-k sampling with temperature)

typedef struct {
    float temperature;
    int top_k;
    unsigned long long rng_state;
} Sampler;

typedef struct {
    float prob;
    int index;
} ProbIndex;

// ----------------------------------------------------------------------------
// API functions

// Model loading and cleanup
void build_transformer(Transformer* t, const char* checkpoint_path);
void free_transformer(Transformer* t);

// Tokenizer
void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
void encode(Tokenizer* t, const char* text, int bos, int eos,
            int* tokens, int* n_tokens);
char* decode(Tokenizer* t, int prev_token, int token);

// Sampler
void build_sampler(Sampler* s, float temperature, int top_k, unsigned long long rng_seed);
int sample(Sampler* s, float* logits, int vocab_size);

// Inference — returns logits for next token
float* forward(Transformer* transformer, int token, int pos);

// Reset KV cache (for new conversation)
void reset_kv_cache(Transformer* transformer);

#endif // RUN_H
