/* Test the C tokenizer encoding against expected PyTorch output. */
#include <stdio.h>
#include <stdlib.h>

#define NO_MAIN
#include "run.c"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <tokenizer.bin>\n", argv[0]);
        return 1;
    }

    Tokenizer tok;
    build_tokenizer(&tok, argv[1], 68096);

    // Test encoding the full prompt
    int tokens[1024];
    int n_tokens;

    // Same prompt the chat mode would build
    char full_prompt[4096];
    snprintf(full_prompt, sizeof(full_prompt), "%s%s\n\n%s",
             "### \xe0\xa4\x89\xe0\xa4\xaa\xe0\xa4\xaf\xe0\xa5\x8b\xe0\xa4\x97\xe0\xa4\x95\xe0\xa4\xb0\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa4\xbe:\n",
             "\xe0\xa4\xad\xe0\xa4\xbe\xe0\xa4\xb0\xe0\xa4\xa4",  // भारत
             "### \xe0\xa4\xb8\xe0\xa4\xb9\xe0\xa4\xbe\xe0\xa4\xaf\xe0\xa4\x95:\n");

    // Encode with BOS
    encode(&tok, full_prompt, 1, 0, tokens, &n_tokens);

    fprintf(stderr, "C tokens (%d): [", n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        fprintf(stderr, "%d", tokens[i]);
        if (i < n_tokens - 1) fprintf(stderr, ", ");
    }
    fprintf(stderr, "]\n");

    // Expected from PyTorch
    int expected[] = {1, 5337, 13273, 60202, 67736, 4103, 51230, 4103, 4103, 13273, 67938, 20431, 67736, 4103};
    int n_expected = 14;

    fprintf(stderr, "Expected  (%d): [", n_expected);
    for (int i = 0; i < n_expected; i++) {
        fprintf(stderr, "%d", expected[i]);
        if (i < n_expected - 1) fprintf(stderr, ", ");
    }
    fprintf(stderr, "]\n");

    // Compare
    int match = (n_tokens == n_expected);
    if (match) {
        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i] != expected[i]) { match = 0; break; }
        }
    }
    fprintf(stderr, "Match: %s\n", match ? "YES" : "NO");

    // Print decoded pieces for both
    fprintf(stderr, "\nC token pieces:\n");
    for (int i = 0; i < n_tokens && i < 30; i++) {
        fprintf(stderr, "  [%d] id=%d piece_len=%d\n", i, tokens[i], tok.vocab_lengths[tokens[i]]);
    }

    free_tokenizer(&tok);
    return 0;
}
