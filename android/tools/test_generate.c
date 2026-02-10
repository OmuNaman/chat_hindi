/* Test full generation with hard-coded correct prompt tokens.
 * Generates 30 tokens and prints their IDs.
 */
#include <stdio.h>
#include <stdlib.h>

#define NO_MAIN
#include "run.c"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.bin>\n", argv[0]);
        return 1;
    }

    Transformer t;
    build_transformer(&t, argv[1]);

    // Exact tokens from PyTorch
    int prompt[] = {1, 5337, 13273, 60202, 67736, 4103, 51230, 4103, 4103, 13273, 67938, 20431, 67736, 4103};
    int n_prompt = 14;

    // Process prompt
    float* logits;
    for (int pos = 0; pos < n_prompt; pos++) {
        logits = forward(&t, prompt[pos], pos);
    }
    int pos = n_prompt;

    // Greedy generate 30 tokens
    fprintf(stderr, "Generated tokens (greedy):\n");
    int expected[] = {10785, 4425, 4551, 35509, 10881, 4432, 67491, 8219, 6426, 15865};
    for (int i = 0; i < 30; i++) {
        // Find argmax
        int best = 0;
        for (int j = 1; j < t.config.vocab_size; j++) {
            if (logits[j] > logits[best]) best = j;
        }
        fprintf(stderr, "  gen[%2d] pos=%2d token=%5d", i, pos, best);
        if (i < 10) {
            fprintf(stderr, " (expected=%5d %s)", expected[i], best == expected[i] ? "OK" : "MISMATCH!");
        }
        fprintf(stderr, "\n");

        if (best == 2) break; // EOS

        // Feed generated token back
        logits = forward(&t, best, pos);
        pos++;
    }

    free_transformer(&t);
    return 0;
}
