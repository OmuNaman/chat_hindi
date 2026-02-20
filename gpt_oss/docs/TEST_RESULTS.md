# GPT-OSS 20B C Inference Engine — Test Results

## Metrics Reference

| Metric | What It Means |
|--------|---------------|
| **Model size (disk)** | Size of the memory-mapped binary file containing all weights (MXFP4 + float16). The OS pages it into RAM on demand. |
| **KV cache** | Pre-allocated key/value cache for all 24 layers across 4096 positions. Stores past attention states so we don't recompute them. |
| **Activation buffers** | Scratch space for intermediate computations (hidden states, QKV projections, MoE buffers, logits). |
| **Total RAM alloc** | KV cache + activations — the actual malloc'd memory. The 12.82 GB model file is mmap'd (OS-managed). |
| **Active params/token** | Only 4 of 32 experts run per token, so 3.6B of the 20.9B params are "active" per token. |
| **Prompt tokens** | Number of tokens in the input (including Harmony template overhead in chat mode). |
| **Prompt time** | Time to process all prompt tokens through the model (prefill phase). |
| **Prompt speed (tok/s)** | Prompt tokens processed per second. Higher = faster prefill. |
| **Generated tokens** | Total tokens the model produced (includes hidden analysis tokens + visible response). |
| **Generation speed (tok/s)** | Tokens generated per second (autoregressive, one at a time). This is the main throughput metric. |
| **Time per token (ms)** | Average milliseconds per generated token. Inverse of generation speed. |
| **Thinking tokens** | Tokens generated in the `analysis` channel (chain-of-thought). Hidden unless `--show-thinking`. |
| **Response tokens** | Tokens generated in the `final` channel (user-visible answer). |
| **Expert utilization** | Which of the 32 MoE experts were selected most often. Shows routing diversity. |

---

## Test Environment

- **Hardware**: Laptop CPU (multi-core, 32 GB RAM)
- **OS**: Windows 11
- **Compiler**: GCC (MinGW) with `-O3 -fopenmp`
- **Model**: `gpt_oss_20b.bin` (12.82 GB, MXFP4 + float16)
- **Tokenizer**: `tokenizer_gptoss.bin` (3 MB, o200k_harmony, 201088 tokens)

---

## Test 1: Raw Completion — "Hello"

**Command:**
```
./run_gptoss.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin \
    --prompt "Hello" --max_tokens 20 --temp 0
```

**Output:**
```
, World!".

Answer:

import org.springframework.web.bind.annotation.GetMapping;
import org
```

**Metrics:**
| Metric | Value |
|--------|-------|
| Prompt tokens | 1 |
| Prompt time | 1.19 s |
| Prompt speed | 0.8 tok/s |
| Generated tokens | 20 |
| Generation time | 9.73 s |
| Generation speed | **2.06 tok/s** |
| Time per token | 486 ms |

**Notes:** Raw completion mode (no Harmony template). Model completes "Hello" as "Hello, World!" then goes into code. Expected base model behavior.

---

## Test 2: Raw Completion — "The capital of France is"

**Command:**
```
./run_gptoss.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin \
    --prompt "The capital of France is" --max_tokens 10 --temp 0
```

**Output:**
```
 Paris."
    # Test with a non-existent article
```

**Metrics:**
| Metric | Value |
|--------|-------|
| Prompt tokens | 5 |
| Prompt time | 2.46 s |
| Prompt speed | 2.0 tok/s |
| Generated tokens | 10 |
| Generation time | 4.90 s |
| Generation speed | **2.04 tok/s** |
| Time per token | 490 ms |

**Notes:** Correctly identifies Paris. Continues into code-like output (base model, no chat template).

---

## Test 3: Harmony Chat — "What is 2 + 2?" (Reasoning High, Thinking Visible)

**Command:**
```
./run_gptoss.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin \
    --chat --reasoning high --show-thinking --max_tokens 256 --temp 0
```

**Input:** `What is 2 + 2?`

**Output (Thinking — analysis channel):**
```
[Thinking...]
We need to answer the question: "What is 2 + 2?" The user is asking a simple
math question. The answer is 4. There's no policy conflict. So we respond with "4".
[Thought for 24.7 seconds, 43 tokens]
```

**Output (Response — final channel):**
```
4
```

**Metrics:**
| Metric | Value |
|--------|-------|
| Prompt tokens | 23 |
| Prompt time | 10.95 s |
| Prompt speed | 2.1 tok/s |
| Generated tokens | 53 |
| Generation speed | **1.95 tok/s** |
| Thinking tokens | 43 |
| Response tokens | 1 |

**Notes:** Full Harmony format working. Model uses `analysis` channel for reasoning, then `final` channel for the answer. The 23 prompt tokens include Harmony template overhead (system message + user wrapper + generation prefix).

---

## Test 4: Harmony Chat — "What is 2 + 2?" (Thinking Hidden)

**Command:**
```
./run_gptoss.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin \
    --chat --reasoning low --max_tokens 256 --temp 0
```

**Input:** `What is 2 + 2?`

**Output:**
```
GPT-OSS: 4
```

**Metrics:**
| Metric | Value |
|--------|-------|
| Prompt tokens | 23 |
| Prompt time | 10.15 s |
| Prompt speed | 2.3 tok/s |
| Generated tokens | 53 |
| Generation speed | **1.76 tok/s** |
| Thinking tokens | 43 (hidden) |
| Response tokens | 1 |

**Notes:** Same generation as Test 3, but analysis channel output is suppressed. User only sees "4". The model still thinks internally (43 tokens), just not displayed.

---

## Test 5: Harmony Chat — "Explain recursion" (Reasoning Low, Thinking Visible)

**Command:**
```
./run_gptoss.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin \
    --chat --reasoning low --show-thinking --max_tokens 512 --temp 0
```

**Input:** `Explain recursion in one sentence.`

**Output (Thinking — analysis channel):**
```
[Thinking...]
We need to answer: "Explain recursion in one sentence." The user wants a
one-sentence explanation. We should provide a concise definition. For example:
"Recursion is a method of solving a problem where the solution depends on
solutions to smaller instances of the same problem." That is a one sentence.
Or "Recursion is a technique where a function calls itself to solve a problem
by breaking it into smaller subproblems." That is also one sentence. The user
wants a one sentence explanation. So we should give a single sentence. Let's
produce a concise answer.
[Thought for 73.4 seconds, 114 tokens]
```

**Output (Response — final channel):**
```
Recursion is a technique in which a function solves a problem by calling itself
on smaller instances of the same problem until reaching a base case.
```

**Metrics:**
| Metric | Value |
|--------|-------|
| Prompt tokens | 21 |
| Prompt time | 9.98 s |
| Prompt speed | 2.1 tok/s |
| Generated tokens | 151 |
| Generation time | 97.33 s |
| Generation speed | **1.55 tok/s** |
| Thinking tokens | 114 |
| Response tokens | 28 |
| Total time | 107.31 s |

**Notes:** Model deliberates over multiple phrasings in the analysis channel, then picks a clean one-sentence answer. The thinking/response split is clearly visible.

---

## Test 6: Harmony Chat — "What is the speed of light?" (Reasoning Low)

**Command:**
```
./run_gptoss.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin \
    --chat --reasoning low --max_tokens 256 --temp 0
```

**Input:** `What is the speed of light?`

**Output:**
```
The speed of light in a vacuum is a fundamental constant, denoted by c. Its value is

c = 299,792,458 meters per second (m/s).

This is the speed at which electromagnetic waves (including visible light)
propagate through empty space.
```

**Metrics:**
| Metric | Value |
|--------|-------|
| Prompt tokens | 22 |
| Prompt time | 9.49 s |
| Prompt speed | 2.3 tok/s |
| Generated tokens | 154 |
| Generation time | 61.55 s |
| Generation speed | **2.50 tok/s** |
| Thinking tokens | 78 (hidden) |
| Response tokens | 67 |
| Total time | 71.04 s |

| Top Experts | Selections |
|-------------|-----------|
| Expert 13 | 693 |
| Expert 7 | 676 |
| Expert 10 | 665 |
| Expert 21 | 663 |
| Expert 30 | 631 |

**Notes:** Correct factual answer (299,792,458 m/s) with LaTeX-style formatting. Peak generation speed of 2.50 tok/s.

---

## Summary

| Test | Mode | Reasoning | Output | Gen Speed | Think/Response |
|------|------|-----------|--------|-----------|----------------|
| 1 | `--prompt` | N/A | "Hello, World!" | 2.06 tok/s | N/A |
| 2 | `--prompt` | N/A | "Paris" | 2.04 tok/s | N/A |
| 3 | `--chat` | high + visible | "4" (with CoT) | 1.95 tok/s | 43 / 1 |
| 4 | `--chat` | low + hidden | "4" | 1.76 tok/s | 43 / 1 |
| 5 | `--chat` | low + visible | Recursion definition | 1.55 tok/s | 114 / 28 |
| 6 | `--chat` | low + hidden | Speed of light = 299,792,458 m/s | 2.50 tok/s | 78 / 67 |

**Average generation speed: ~2.0 tok/s** on laptop CPU with OpenMP.

---

## Memory Footprint

| Component | Size |
|-----------|------|
| Model file (mmap'd) | 12.82 GB |
| KV cache | 384.0 MB |
| Activation buffers | 1.9 MB |
| **Total malloc'd RAM** | **385.9 MB** |

The 12.82 GB model file is memory-mapped — the OS pages in only the parts needed. On a 32 GB machine, the full file fits in RAM after a few prompts. On 16 GB, there will be page faults but it still works.
