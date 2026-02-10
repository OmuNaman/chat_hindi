/* Test the full decode pipeline: generate correct token IDs, decode them, print text.
 * This bypasses command-line encoding entirely to isolate decode bugs.
 *
 * Also tests: encoding a hardcoded UTF-8 Hindi prompt via encode_chat_prompt.
 */
#include <stdio.h>
#include <stdlib.h>

#define NO_MAIN
#include "run.c"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.bin> <tokenizer.bin>\n", argv[0]);
        return 1;
    }

    Transformer t;
    build_transformer(&t, argv[1]);

    Tokenizer tok;
    build_tokenizer(&tok, argv[2], t.config.vocab_size);

    // ===== TEST 1: Decode known-correct token IDs =====
    fprintf(stderr, "\n=== TEST 1: Decode known token IDs ===\n");

    // These are the greedy-generated tokens verified against PyTorch
    int gen_tokens[] = {10785, 4425, 4551, 35509, 10881, 4432, 67491, 8219, 6426, 15865};
    int n_gen = 10;

    // Print what each token decodes to
    int prev = 4103; // last prompt token
    for (int i = 0; i < n_gen; i++) {
        char* piece = decode(&tok, prev, gen_tokens[i]);
        int len = (int)strlen(piece);
        fprintf(stderr, "  token %5d → piece len=%d bytes=[", gen_tokens[i], len);
        for (int j = 0; j < len && j < 20; j++) {
            fprintf(stderr, "%02X ", (unsigned char)piece[j]);
        }
        fprintf(stderr, "]\n");

        // Now do the same ▁ handling as main() does
        if (len >= 3 && (unsigned char)piece[0] == 0xE2 &&
            (unsigned char)piece[1] == 0x96 && (unsigned char)piece[2] == 0x81) {
            // Print " " + rest to stdout (redirectable to file)
            fprintf(stdout, " %s", piece + 3);
            fprintf(stderr, "    → printed: ' ' + piece[3:] (▁ detected)\n");
        } else {
            fprintf(stdout, "%s", piece);
            fprintf(stderr, "    → printed: raw piece\n");
        }
        prev = gen_tokens[i];
    }
    fprintf(stdout, "\n");
    fflush(stdout);

    // ===== TEST 2: Full pipeline with hardcoded UTF-8 prompt =====
    fprintf(stderr, "\n=== TEST 2: Full encode → forward → greedy → decode ===\n");

    // "भारत" in UTF-8: E0 A4 AD E0 A4 BE E0 A4 B0 E0 A4 A4
    const char prompt[] = "\xe0\xa4\xad\xe0\xa4\xbe\xe0\xa4\xb0\xe0\xa4\xa4";

    int* prompt_tokens = (int*)malloc(t.config.seq_len * sizeof(int));
    int n_prompt = encode_chat_prompt(&tok, prompt, prompt_tokens);

    fprintf(stderr, "Prompt tokens (%d): ", n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        fprintf(stderr, "%d ", prompt_tokens[i]);
    }
    fprintf(stderr, "\n");

    // Expected: 1 5337 13273 60202 67736 4103 51230 4103 4103 13273 67938 20431 67736 4103
    int expected_prompt[] = {1, 5337, 13273, 60202, 67736, 4103, 51230, 4103, 4103, 13273, 67938, 20431, 67736, 4103};
    int n_expected = 14;

    int match = (n_prompt == n_expected);
    if (match) {
        for (int i = 0; i < n_expected; i++) {
            if (prompt_tokens[i] != expected_prompt[i]) {
                match = 0;
                break;
            }
        }
    }
    fprintf(stderr, "Prompt encoding: %s\n", match ? "MATCHES PyTorch" : "MISMATCH!");

    // Forward pass through prompt
    float* logits;
    for (int pos = 0; pos < n_prompt; pos++) {
        logits = forward(&t, prompt_tokens[pos], pos);
    }

    // Greedy generate 20 tokens
    fprintf(stderr, "\nGenerated (greedy):\n");
    prev = prompt_tokens[n_prompt - 1];
    int expected_gen[] = {10785, 4425, 4551, 35509, 10881, 4432, 67491, 8219, 6426, 15865};

    for (int i = 0; i < 20; i++) {
        // Argmax
        int best = 0;
        for (int j = 1; j < t.config.vocab_size; j++) {
            if (logits[j] > logits[best]) best = j;
        }

        // Decode and print
        char* piece = decode(&tok, prev, best);
        int len = (int)strlen(piece);

        fprintf(stderr, "  [%2d] token=%5d", i, best);
        if (i < 10) {
            fprintf(stderr, " (expected=%5d %s)", expected_gen[i],
                    best == expected_gen[i] ? "OK" : "MISMATCH!");
        }
        fprintf(stderr, " bytes=[");
        for (int j = 0; j < len && j < 20; j++) {
            fprintf(stderr, "%02X ", (unsigned char)piece[j]);
        }
        fprintf(stderr, "]\n");

        // Print decoded text to stdout
        if (len >= 3 && (unsigned char)piece[0] == 0xE2 &&
            (unsigned char)piece[1] == 0x96 && (unsigned char)piece[2] == 0x81) {
            fprintf(stdout, " %s", piece + 3);
        } else {
            fprintf(stdout, "%s", piece);
        }
        fflush(stdout);

        if (best == 2) break; // EOS

        prev = best;
        logits = forward(&t, best, n_prompt + i);
    }
    fprintf(stdout, "\n");
    fflush(stdout);

    // ===== TEST 3: Check ▁ detection with signed char =====
    fprintf(stderr, "\n=== TEST 3: Signed char ▁ detection ===\n");
    char test_piece[] = "\xe2\x96\x81\xe0\xa4\xad";  // ▁भ
    fprintf(stderr, "  test_piece[0] = 0x%02X, '\\xe2' = 0x%02X, equal? %s\n",
            (unsigned char)test_piece[0], (unsigned char)'\xe2',
            test_piece[0] == '\xe2' ? "YES" : "NO");
    fprintf(stderr, "  (char)0xE2 = %d, '\\xe2' = %d\n", (int)(char)0xE2, (int)'\xe2');

    // Also test with plain char comparison (what main() does)
    if (test_piece[0] == '\xe2' && test_piece[1] == '\x96' && test_piece[2] == '\x81') {
        fprintf(stderr, "  ▁ detection: WORKS (plain char comparison)\n");
    } else {
        fprintf(stderr, "  ▁ detection: FAILED\n");
    }

    free(prompt_tokens);
    free_tokenizer(&tok);
    free_transformer(&t);
    fprintf(stderr, "\nAll tests complete.\n");
    return 0;
}
