/* Direct test: hard-code PyTorch's exact prompt tokens, run C forward pass,
 * compare the predicted next token. This isolates forward pass bugs from tokenizer bugs.
 *
 * PyTorch prompt tokens for "भारत":
 *   [1, 5337, 13273, 60202, 67736, 4103, 51230, 4103, 4103, 13273, 67938, 20431, 67736, 4103]
 * PyTorch greedy first output token: 10785
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

    fprintf(stderr, "Processing %d prompt tokens...\n", n_prompt);

    // Forward pass for all prompt tokens
    float* logits = NULL;
    for (int pos = 0; pos < n_prompt; pos++) {
        logits = forward(&t, prompt[pos], pos);
        // Print top predicted token after each position
        int top = 0;
        for (int i = 1; i < t.config.vocab_size; i++) {
            if (logits[i] > logits[top]) top = i;
        }
        fprintf(stderr, "  pos=%2d token=%5d -> top_next=%5d (logit=%.4f)\n",
                pos, prompt[pos], top, logits[top]);
    }

    // Find top-5 tokens after processing full prompt
    fprintf(stderr, "\nTop-5 after full prompt:\n");
    int top5[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < t.config.vocab_size; i++) {
        for (int k = 0; k < 5; k++) {
            if (logits[i] > logits[top5[k]]) {
                for (int j = 4; j > k; j--) top5[j] = top5[j-1];
                top5[k] = i;
                break;
            }
        }
    }
    for (int k = 0; k < 5; k++) {
        fprintf(stderr, "  #%d: token=%d logit=%.4f\n", k+1, top5[k], logits[top5[k]]);
    }

    fprintf(stderr, "\nExpected: token 10785 (PyTorch greedy)\n");

    free_transformer(&t);
    return 0;
}
