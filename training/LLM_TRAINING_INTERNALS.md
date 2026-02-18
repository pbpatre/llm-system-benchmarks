# How an LLM Learns: The Full Picture from Text to GPU

*A ground-up walkthrough of what actually happens when you train a language model — the math, the memory, and the machines — for engineers who want to understand, not just call `.fit()`.*

---

## 1. From Text to Numbers

A language model never sees text. It sees arrays of integers.

Before training begins, a **tokenizer** splits raw text into subword chunks and maps each chunk to an integer ID using a fixed vocabulary of 50,257 entries (for GPT-2). The tokenizer is not part of the model — it runs once on CPU before training and converts the entire dataset into a stream of integers.

For example:

```
"The cat sat on the mat" → ["The", " cat", " sat", " on", " the", " mat"]
                         → [464,   3797,   3332,   319,   262,    2603]
```

Some words become one token (" cat" → 3797). Rare words get split into pieces ("unbelievable" → "un" + "believ" + "able" — three tokens). This is called **Byte-Pair Encoding (BPE)**: start with individual characters, repeatedly merge the most common adjacent pair into a new token, repeat 50,257 times. The result is a vocabulary that handles common words as single tokens and rare words as composable pieces.

Training a language model means predicting the **next token**. Given tokens at positions 0 through 1023, predict what comes at position 1 through 1024. The input and target are the same sequence, shifted by one position:

```
Input:   [The] [cat] [sat] [on]  [the] [mat] ...  [token 1023]
Target:  [cat] [sat] [on]  [the] [mat] [and] ...  [token 1024]
```

Position 0 sees only "The" and tries to predict "cat". Position 5 sees "The cat sat on the mat" and tries to predict what comes next. The model makes 1,024 predictions per sequence, simultaneously.

---

## 2. From Numbers to Meaning

An integer tells the model nothing about what a word means. The number 3797 could be "cat" or "quantum" — the model cannot do math on a dictionary index. It needs a richer representation.

The first layer of the model is an **embedding table**: a matrix of shape (50,257 rows, 768 columns). Each row is a learnable 768-dimensional vector representing one vocabulary token. To process token 3797, the model looks up row 3797 and retrieves a vector of 768 numbers:

```
token ID 3797 ("cat") → embedding_table[3797] → [0.12, -0.45, 0.88, ..., -0.71]
                                                         768 numbers
```

This is **d_model** — the width of the model's internal thought. Every token at every position is represented as 768 numbers throughout the entire model. You cannot read these numbers directly — they are learned, abstract features. Conceptually, different dimensions encode things like part of speech, animacy, topic domain, and grammatical role, but in a distributed way that no single dimension maps cleanly to one concept.

After embedding, the tensor flowing into the model has three dimensions:

| Axis | Size | Meaning |
|------|------|---------|
| 0 | 8 | **Batch**: 8 independent sequences processed in parallel |
| 1 | 1024 | **Sequence**: 1024 token positions per sequence |
| 2 | 768 | **Embedding**: 768 numbers describing each token |

The shape `(8, 1024, 768)` — 8 sequences of 1024 tokens, each represented by 768 numbers — is what enters the first transformer layer.

---

## 3. Self-Attention: Who Should I Listen To?

Consider: "The cat sat on the mat because it was tired." When the model reaches "it", it must figure out whether "it" refers to "cat" or "mat." Self-attention solves this by letting every token ask a question of every preceding token and blend the answers.

Each token's 768-d vector is projected into three separate vectors through learned weight matrices:

- **Query (Q)**: "What am I looking for?" — what information this position needs.
- **Key (K)**: "What do I contain?" — what information this position offers.
- **Value (V)**: "What should I hand over?" — the actual content to share if selected.

The projections are matrix multiplications:

```
Q = X × W_Q    →  (1024, 768) × (768, 768) = (1024, 768)
K = X × W_K    →  (1024, 768) × (768, 768) = (1024, 768)
V = X × W_V    →  (1024, 768) × (768, 768) = (1024, 768)
```

Then we compute how much each position should attend to each other by taking the dot product of every Query with every Key:

```
scores = Q × K^T / sqrt(768)

(1024, 768) × (768, 1024) = (1024, 1024)
```

Entry `[i, j]` in this matrix is: "how relevant is position j to position i?" The division by sqrt(768) ≈ 27.7 is critical — without it, the dot products grow proportionally to the dimension size, causing the softmax (next step) to saturate. When softmax saturates, one position gets probability ~1.0 and the rest get ~0.0, which kills gradients and stops learning.

Since this is autoregressive — the model cannot see the future — we set all entries where j > i to negative infinity before the next step. Position 3 can attend to positions 0, 1, 2, 3 but not 4, 5, 6, ...

### Softmax: Turning Scores into Weights

The raw attention scores are arbitrary numbers. We need probabilities — positive values that sum to 1. Softmax achieves this by exponentiating each score and normalizing.

A worked example with four positions:

```
Raw scores:     [2.0,  1.0,  0.1,  -1.0]
Exponentiate:   [7.39, 2.72, 1.11,  0.37]     sum = 11.59
Divide by sum:  [0.64, 0.23, 0.10,  0.03]     sums to 1.0
```

The exponentiation amplifies differences — a score of 2.0 vs 1.0 becomes a probability ratio of 7.39/2.72 ≈ 2.7x, not just 2x. The largest score dominates the distribution. You can think of this as a "temperature" effect: very large scores produce a near-one-hot distribution (the model is certain), while similar scores produce a flat distribution (the model is unsure).

These attention weights are then used to blend the Value vectors:

```
output = weights × V    →  (1024, 1024) × (1024, 768) = (1024, 768)
```

For position 3: output = 0.64 × V₀ + 0.23 × V₁ + 0.10 × V₂ + 0.03 × V₃. A weighted average of the content from all visible positions, where the weights come from Q/K compatibility.

### Multi-Head Attention

GPT-2 Small doesn't run one attention over all 768 dimensions. It splits them into **12 heads** of 64 dimensions each. Each head runs its own Q/K/V attention independently — perhaps head 1 learns syntactic relationships (subject-verb agreement), head 7 learns coreference ("it" → "cat"), and head 11 learns positional patterns (nearby words matter more). The 12 outputs are concatenated back to 768 dimensions and mixed through a final linear projection.

### Complexity

The attention score computation `Q × K^T` produces a (1024, 1024) matrix — one million entries per head, 12 million total. Attention scales as **O(seq_len²)**: double the context length and you quadruple the computation and memory. This is why LLMs have context limits. At 128K tokens, the attention matrix alone would require 128K × 128K × 2 bytes = 32 GB per layer.

---

## 4. FFN: Now Think About It

Attention mixes information **between** positions — "I know that 'it' refers to 'cat'." The **feed-forward network (FFN)** transforms information **within** each position independently — "Given that 'it' means 'cat', what does that imply about the next word?" This is where the model does its reasoning.

The FFN is a two-layer neural network applied to each position's 768-d vector:

```
Step 1:  Expand:   (768) × (768, 3072)  = (3072)     — project to 4x wider space
Step 2:  Activate: GELU(3072)            = (3072)     — non-linear transformation
Step 3:  Compress: (3072) × (3072, 768)  = (768)      — project back to model width
```

The inner dimension of 3072 (4 × d_model) acts as a "thinking space." The model expands into a larger representation where it can compute complex features, then compresses the useful ones back down.

### GELU: The Smooth Gate

If every operation were linear (just matrix multiplies and additions), stacking 12 layers would be mathematically identical to a single matrix multiply — no matter how deep the model, it would have the capacity of one layer. Non-linear activations are what make depth useful.

GELU (Gaussian Error Linear Unit) acts as a smooth gate: strong positive signals pass through nearly unchanged, strong negatives are suppressed to near-zero, and values near zero get a soft, gradual transition.

Concretely:

```
GELU( 2.0) =  1.96    Strong positive → passes through (almost identity)
GELU( 0.0) =  0.0     Zero → stays zero
GELU(-0.5) = -0.15    Small negative → dampened but not killed
GELU(-2.0) = -0.04    Large negative → nearly zeroed out
```

Compare this to the older ReLU activation, which uses a hard cutoff: every negative value becomes exactly 0. GELU's smooth curve avoids "dead neurons" — units that get stuck outputting zero and never recover — because small negative values still have a non-zero gradient to learn from.

### Where the Parameters Live

The FFN's two weight matrices have 768 × 3072 + 3072 × 768 = 4.7 million parameters per layer. Attention's Q/K/V matrices have 3 × 768 × 768 = 1.8 million per layer. Across 12 layers, FFNs account for roughly two-thirds of the model's 124 million total parameters. Attention gets the conceptual spotlight, but FFN is where most of the learned knowledge is stored.

---

## 5. The Prediction: Logits and Loss

After 12 layers of attention and FFN, each position holds a 768-d vector encoding its meaning in context. The final step is to convert this back to a prediction over the vocabulary.

The **LM head** is a linear projection that maps 768 dimensions to 50,257 scores (one per vocabulary token):

```
logits = hidden × W_lm_head    →  (1024, 768) × (768, 50257) = (1024, 50257)
```

A detail that is both elegant and practical: the LM head's weight matrix is the **same** matrix as the embedding table, transposed. This is called **weight tying**. The embedding maps token IDs into 768-d space; the LM head maps 768-d vectors back to token space. Mathematically, each logit is the dot product between the hidden state and a vocabulary token's embedding — "how similar is my prediction to the embedding of token X?"

### Softmax Over the Vocabulary

The same softmax from attention now applies over the vocabulary dimension, converting 50,257 raw scores into a probability distribution:

```
P(token_j) = exp(logit_j) / sum(exp(logit_k) for all k)
```

### Cross-Entropy: Measuring Prediction Quality

The loss function measures how much probability the model assigned to the **correct** next token. Cross-entropy loss is simply the negative log of that probability.

A worked example with a tiny vocabulary of 5 tokens, where the correct next token is token 2:

**Early in training** (model is confused):

```
Logits:    [1.0,  2.0,  0.5,  -1.0,  0.3]
Softmax:   [0.15, 0.40, 0.09,  0.02, 0.07]
                        ↑ correct token only gets 9%
Loss: -log(0.09) = 2.41  (high — bad prediction)
```

**After training** (model has learned):

```
Logits:    [0.2,  0.5,  3.5,  -1.0,  0.1]
Softmax:   [0.03, 0.04, 0.82,  0.01, 0.03]
                        ↑ correct token gets 82%
Loss: -log(0.82) = 0.20  (low — good prediction)
```

The logarithm creates an asymmetric penalty curve. When the model is right, the reward saturates — going from 80% to 90% confidence only reduces loss by 0.12. When the model is wrong, the penalty explodes:

```
P(correct) = 0.9    →  loss = 0.11   (confident and right: small loss)
P(correct) = 0.5    →  loss = 0.69   (coin flip: moderate loss)
P(correct) = 0.01   →  loss = 4.61   (confidently wrong: severe penalty)
P(correct) = 0.001  →  loss = 6.91   (catastrophically wrong: extreme penalty)
```

This exponential punishment for confident wrong answers is exactly what drives the model to spread probability mass wisely rather than gambling.

The final loss is averaged across all 1,024 positions and all 8 sequences in the batch, producing a single scalar number. This is what backpropagation starts from.

**Perplexity**, the standard metric for language models, is simply exp(loss). A loss of 7.13 means perplexity ≈ 1,245 — the model is as uncertain as choosing uniformly among 1,245 tokens. A well-trained GPT-2 reaches perplexity ~20-30, meaning it narrows the next token down to about 20-30 plausible candidates.

---

## 6. Backpropagation: Who's to Blame?

We have a single loss number. We need to figure out: for each of the 124 million parameters, how much did **that** parameter contribute to the error? Then nudge it in the direction that reduces the error.

The tool is the **chain rule** of calculus. If the loss is the result of a chain of operations — embedding → attention₁ → FFN₁ → attention₂ → FFN₂ → ... → LM head → loss — then the derivative of the loss with respect to any parameter deep in the chain is the product of derivatives along the path:

```
d(loss)/d(w) = d(loss)/d(layer12) × d(layer12)/d(layer11) × ... × d(layer1)/d(w)
```

Backpropagation walks this chain backwards. Starting from the loss, it computes the gradient at each layer by multiplying the incoming gradient (from the layer above) by the local derivative (of the current operation). Each step is a matrix multiply with transposed weight matrices — the same operation as the forward pass, just reversed.

For a concrete FFN layer, given the gradient flowing down from above:

```
Forward (computed earlier, stored in memory):
   a = x × W₁        → linear transformation
   h = GELU(a)        → non-linear activation
   y = h × W₂        → linear transformation

Backward (using stored activations):
   d(loss)/d(W₂) = h^T × d(loss)/d(y)         ← gradient for W₂'s update
   d(loss)/d(h)  = d(loss)/d(y) × W₂^T         ← pass error backward through W₂
   d(loss)/d(a)  = d(loss)/d(h) × GELU'(a)     ← pass error through activation
   d(loss)/d(W₁) = x^T × d(loss)/d(a)          ← gradient for W₁'s update
   d(loss)/d(x)  = d(loss)/d(a) × W₁^T         ← pass error to the previous layer
```

Notice: computing `d(loss)/d(W₂) = h^T × ...` requires `h` — the output of the forward pass. Every layer stores its intermediate results (activations) during the forward pass because the backward pass needs them to compute gradients. This is why **activations dominate GPU memory**, not the model weights themselves. The 124M parameter model occupies ~250 MB in bf16, but its activations during a forward pass occupy ~13 GB — fifty times more.

The backward pass does roughly 2x the computation of the forward pass because it computes two sets of gradients at each layer: one for the weight update (d(loss)/d(W)) and one for passing the error to the previous layer (d(loss)/d(x)).

---

## 7. The Optimizer Step: Nudge the Weights

Backpropagation produces a gradient for every parameter — a vector that says "moving in this direction increases the loss." To decrease the loss, move in the opposite direction. The most naive update possible is **Stochastic Gradient Descent (SGD)**:

```
weight = weight - learning_rate × gradient
```

For a parameter with value `0.5` and gradient `0.1`, with learning rate `0.001`:

```
new_weight = 0.5 - 0.001 × 0.1 = 0.4999
```

This works but has two serious problems in practice. First, every parameter gets the same step size, even though different parameters have wildly different gradient magnitudes and sensitivities. Second, each mini-batch is a random sample of the data, so the gradient is a noisy estimate — taking large steps on noisy gradients causes zigzagging rather than smooth convergence. AdamW solves both.

### AdamW: Adaptive Moments

AdamW maintains two **running statistics** per parameter, updated every step from the gradient history:

**First moment `m` — smooths gradient noise (momentum)**

```
m = 0.9 × m_prev + 0.1 × gradient_current
```

This is a weighted average of all past gradients, with recent ones weighted more. With `beta1=0.9`, each step's gradient contributes 10% to the average, and its influence decays by 10% every subsequent step.

Concrete example over five steps for one parameter:

```
Step 1: gradient =  0.10,  m = 0.9×0.00 + 0.1×0.10 = 0.010
Step 2: gradient =  0.12,  m = 0.9×0.01 + 0.1×0.12 = 0.021
Step 3: gradient =  0.09,  m = 0.9×0.02 + 0.1×0.09 = 0.028
Step 4: gradient = -0.05,  m = 0.9×0.03 + 0.1×(-0.05) = 0.020  ← noise barely dents it
Step 5: gradient =  0.11,  m = 0.9×0.02 + 0.1×0.11 = 0.029
```

The single negative gradient at step 4 only slightly reduces the momentum. If the gradient is genuinely reversing (the loss surface curves back), the momentum slows over several steps. If it's just noise, the consistent positive signal wins. **Crucially, `m` cannot be computed from the current gradient alone — it is the accumulated history. Throw it away and you lose everything the optimizer has learned about the direction of steepest descent.**

**Second moment `v` — enables per-parameter step sizes (variance)**

```
v = 0.95 × v_prev + 0.05 × gradient²
```

This tracks the **average magnitude of squared gradients** — how volatile this parameter's gradient has been historically. Large `v` means gradients have been large and unpredictable. Small `v` means gradients have been consistently small.

```
Noisy parameter (large swings):    v ≈ 0.25,   sqrt(v) ≈ 0.50
Quiet parameter (tiny gradients):  v ≈ 0.0004, sqrt(v) ≈ 0.02
```

**The actual update formula**

```
weight = weight - learning_rate × (m / sqrt(v + ε)) - learning_rate × weight_decay × weight
                                   ───────────────
                                   adaptive gradient
```

The `m / sqrt(v)` term is the normalized update. For the noisy parameter: `0.10 / 0.50 = 0.20` — scaled down. For the quiet parameter: `0.002 / 0.02 = 0.10` — a 5x larger relative step than its raw gradient suggests. Adam automatically amplifies learning for parameters that have historically been undertrained and dampens it for parameters that have been overtrained.

Concrete example with two parameters side by side:

```
Parameter A (large, volatile gradients):
  m = 0.10, v = 0.25, sqrt(v) = 0.50
  effective step = 3e-4 × (0.10 / 0.50) = 6e-5

Parameter B (small, consistent gradients):
  m = 0.002, v = 0.000004, sqrt(v) = 0.002
  effective step = 3e-4 × (0.002 / 0.002) = 3e-4
```

Parameter B takes a 5x larger step than A, even though its raw gradient is 50x smaller. This is the adaptive behaviour that makes Adam dramatically more efficient than plain SGD on heterogeneous loss landscapes.

### Weight Decay: The "W" in AdamW

Every step, regardless of the gradient, all weights are pulled slightly toward zero:

```
weight = weight - learning_rate × gradient_term - learning_rate × 0.1 × weight
                                                  ──────────────────────────────
                                                  decays weights by 0.01% per step
```

This regularizes the model — it requires the gradient signal to justify maintaining large weight values, preventing any single parameter from growing pathologically large. Without it, transformers develop very large weights that memorize training data rather than generalizing.

### Memory Cost

AdamW stores `m` and `v` in fp32 (to maintain precision across thousands of update steps) for every parameter. The full memory breakdown:

```
Model weights (bf16):  124M × 2 bytes = 248 MB
Gradients (bf16):      124M × 2 bytes = 248 MB
Optimizer m (fp32):    124M × 4 bytes = 496 MB
Optimizer v (fp32):    124M × 4 bytes = 496 MB
                                       ────────
Total:                                 ~1.5 GB
```

The optimizer states alone are 4x the size of the model in bf16. This is why optimizer state sharding is one of the biggest wins in distributed training — FSDP can split these across 8 GPUs, reducing the per-GPU footprint from 1.5 GB to 190 MB for our 124M parameter model.

---

## 7b. The Learning Rate Schedule

The learning rate is the global multiplier on all steps. AdamW handles per-parameter adaptation. The scheduler handles **when** it's safe to take large steps globally.

### The Problem: Training Is Not Stable at a Fixed Rate

Early in training, the model has random weights and produces wild, inconsistent gradients. The optimizer's `m` and `v` statistics are both zero — they have no history. The first gradient, divided by `sqrt(v + ε)` where `v ≈ 0`, produces enormous effective steps. A high learning rate at this stage can permanently destabilize the model — a parameter overshoots and lands in a bad region from which gradients keep getting worse.

Late in training, the model is close to a good solution. Large steps overshoot the minimum — jumping past a low-loss valley and landing somewhere worse. You need progressively finer adjustments.

### Phase 1: Warmup (steps 0-9)

The learning rate ramps linearly from 0 to the peak:

```
Step 0:  LR = 3e-4 × (0/10) = 0.0        (effectively zero)
Step 3:  LR = 3e-4 × (3/10) = 9e-5
Step 7:  LR = 3e-4 × (7/10) = 2.1e-4
Step 10: LR = 3e-4 × (10/10) = 3e-4      (peak reached)
```

By step 10, `m` and `v` have accumulated enough history to be reliable, and the model is past its most chaotic initialization phase. Only then does the learning rate reach full strength.

### Phase 2: Cosine Decay (steps 10-100)

The learning rate smoothly decreases from peak to a floor (10% of peak) following a half-cosine curve:

```
Step 10:  LR = 3e-4    (peak)
Step 30:  LR = 2.4e-4  (still large, model learning quickly)
Step 55:  LR = 1.65e-4 (midpoint — half the peak)
Step 80:  LR = 6e-5    (slowing down for fine-tuning)
Step 100: LR = 3e-5    (floor — gentle landing)
```

The cosine shape (rather than linear) is important: it decays slowly at first (the model still benefits from larger steps), accelerates through the middle, then slows near the floor. The floor of 3e-5 (not zero) keeps the optimizer making small refinements rather than freezing entirely.

### Gradient Clipping

Before every optimizer step, the total gradient magnitude is measured across all 124M parameters. If it exceeds 1.0, all gradients are scaled down proportionally:

```
grad_norm = sqrt(sum of all gradient² values)

if grad_norm > 1.0:
    gradients = gradients × (1.0 / grad_norm)
```

This bounds the maximum step the optimizer can take in a single update. Transformers are prone to gradient explosions: the chain rule multiplies gradients through 12 layers, and occasionally a bad batch produces gradients that compound multiplicatively into enormous values. Without clipping, one outlier batch can ruin a thousand steps of progress.

### Gradient Accumulation

A batch of 32 sequences produces more stable gradients than a batch of 8 — averaging over more samples reduces the noise. But 32 sequences of 1024 tokens don't fit in GPU memory simultaneously (activations alone would require ~42 GB for batch=32).

The solution: run 4 micro-batches of 8, accumulating gradients across all four, then apply one optimizer step. Each micro-batch's loss is divided by 4:

```
Micro-batch 1/4:  forward(8 seqs) → loss/4 → backward → grads += δ₁/4
Micro-batch 2/4:  forward(8 seqs) → loss/4 → backward → grads += δ₂/4
Micro-batch 3/4:  forward(8 seqs) → loss/4 → backward → grads += δ₃/4
Micro-batch 4/4:  forward(8 seqs) → loss/4 → backward → grads += δ₄/4
                                                          ────────────────
                  accumulated grad = mean gradient over 32 sequences → optimizer.step()
```

Only 8 sequences worth of activations (~10.5 GB) exist in VRAM at any moment. The gradients (248 MB) simply accumulate across all four passes. Mathematically identical to a true batch of 32, at 4x the wall-clock time per step.

---

## 8. Why GPUs: The Matrix Multiply Machine

Every operation described above — Q/K/V projections, attention scores, FFN layers, the LM head, and every step of backpropagation — is a **matrix multiplication**. A single training step for GPT-2 Small involves roughly 72 billion multiply-add operations.

A CPU has 8-16 powerful cores optimized for complex branching logic. It can do matrix multiplies, but processes maybe 16 operations per clock cycle in parallel. A good server CPU sustains ~1 TFLOP/s (one trillion floating-point operations per second).

A GPU has thousands of tiny cores designed for one thing: multiply two numbers and add the result. An NVIDIA L40S has 18,176 CUDA cores plus 568 **Tensor Cores** — specialized units that process 4×4 matrix blocks in a single cycle. At bf16 precision, the L40S sustains ~181 TFLOP/s.

The result:

```
72 billion operations per training step:

CPU (1 TFLOP/s):    72 seconds per step
GPU (181 TFLOP/s):  0.4 seconds per step    ← 180x faster
```

### The Memory Bandwidth Wall

The GPU's compute is so fast that the bottleneck shifts to **feeding it data**. The L40S performs 181 trillion operations per second but can only read from its memory (VRAM) at 864 GB/s. For large matrix multiplies — like the FFN's (1024, 768) × (768, 3072) — the compute-to-memory ratio is high: hundreds of operations per byte loaded. The GPU's cores stay busy. This is the **compute-bound** regime, and it's the ideal operating point.

For small operations — adding a bias vector, applying layer normalization, computing GELU — the ratio is low. You load a number, do one operation, write it back. The cores sit idle waiting for data. This is the **memory-bound** regime. `torch.compile` helps here by fusing multiple small operations into a single GPU kernel, reducing round-trips to memory.

### Mixed Precision

**bf16** (bfloat16) uses 16 bits per number instead of 32. This halves the memory for model weights and activations, and Tensor Cores process bf16 at 2x the throughput of fp32. The tradeoff is 3 fewer digits of mantissa precision — negligible for the matrix multiplies in attention and FFN, but potentially problematic for operations that accumulate many small values (reductions, layer norms). PyTorch's `autocast` handles this automatically: matrix multiplies run in bf16 for speed, while numerically sensitive operations stay in fp32 for accuracy.

### The Memory Hierarchy

Training data flows through a hierarchy of increasingly fast, increasingly small memories:

```
                    Speed         Size
                    ─────         ────
GPU Registers:      ~20 TB/s      256 KB per SM
GPU Shared Memory:  ~15 TB/s      128 KB per SM
GPU L2 Cache:       ~6 TB/s       48-96 MB
GPU HBM (VRAM):     864 GB/s      44 GB          ← weights + activations
───── PCIe bus ────  32 GB/s  ──────────────────
CPU RAM:            ~50 GB/s      124 GB          ← dataset tokens
Disk (NVMe):        ~3 GB/s       559 GB          ← cold storage
```

This is why `pin_memory=True` matters in the DataLoader: it allocates page-locked CPU memory that the GPU's DMA engine can read directly, bypassing the operating system's staging buffer. Combined with `non_blocking=True`, the CPU can prepare the next batch while the current one is still in transit across the PCIe bus. Without these, the GPU sits idle waiting for data — the exact pipeline bottleneck detailed in the companion analysis, *Feeding the Beast*.

---

## 9. Scaling Out: From One GPU to Many

A single GPU processes ~43,000 tokens per second on GPT-2 Small. To train on trillions of tokens in a reasonable time, you need many GPUs working together.

### DDP: Data Parallelism

**Distributed Data Parallel (DDP)** is the simplest strategy: put a complete copy of the model on every GPU, give each GPU different data, and average the gradients.

With 2 GPUs, the training step looks like this:

```
GPU 0: Forward on batch A → backward → gradients_A = [0.10, -0.04,  0.20, -0.08]
GPU 1: Forward on batch B → backward → gradients_B = [0.06, -0.12,  0.16,  0.02]

                    ┌─── All-Reduce (average) ───┐
                    │                             │
GPU 0 gets: [0.08, -0.08, 0.18, -0.03]    ← identical
GPU 1 gets: [0.08, -0.08, 0.18, -0.03]    ← identical

Both GPUs apply the same optimizer step → weights stay in sync.
```

This is mathematically equivalent to a single GPU processing both batches. The effective batch size doubles — twice the data throughput, same learning dynamics.

### All-Reduce: How Gradients Are Synchronized

**All-reduce** is the collective operation that takes a tensor from every GPU, computes the average, and distributes the identical result back to all GPUs. With 124 million parameters, you don't send everything at once.

**Ring all-reduce** splits the gradient tensor into N chunks (one per GPU) and passes them around a ring. In the first phase (reduce-scatter), each chunk travels around the ring accumulating partial sums, until one GPU holds the complete sum for its assigned chunk. In the second phase (all-gather), each GPU's completed chunk is broadcast to all others. Total data transferred per GPU: approximately 2 × (N-1)/N × tensor size — nearly constant regardless of the number of GPUs.

DDP overlaps this communication with the backward pass. As soon as a "bucket" (~25 MB of gradients) is computed, the all-reduce fires in the background while the remaining backward computation continues. By the time backward finishes, most gradients are already synchronized.

### FSDP: When the Model Doesn't Fit

DDP requires every GPU to hold the **full model** — weights, optimizer states, and gradients. For GPT-2 Small (124M params), that's manageable. For a 70-billion-parameter model, the optimizer states alone require ~560 GB — no single GPU has this.

**Fully Sharded Data Parallel (FSDP)** solves this by sharding the model across GPUs. Each GPU holds only 1/N of the parameters, 1/N of the gradients, and 1/N of the optimizer states. When a layer needs the full parameter tensor for a forward pass, it issues an **all-gather** to collect shards from other GPUs, computes, then discards the gathered parameters. During backward, a **reduce-scatter** simultaneously averages the gradients and distributes the sharded result.

The tradeoff: more communication per step (all-gather before every layer, reduce-scatter after), but dramatically less memory per GPU. A model that requires 560 GB splits across 8 GPUs as 70 GB each.

### NCCL: The Communication Engine

All of these collective operations — all-reduce, all-gather, reduce-scatter, barrier — are implemented by **NCCL** (NVIDIA Collective Communications Library). NCCL runs as GPU-side CUDA kernels: data flows directly from one GPU's VRAM to another's over NVLink (600 GB/s) or across machines via InfiniBand (200-400 Gb/s), without touching the CPU at all. It automatically detects the hardware topology — which GPUs share an NVLink switch, which are across PCIe, which are on different nodes — and computes optimal communication patterns.

The alternative backend, GLOO, routes everything through CPU memory and TCP sockets. It exists as a fallback for machines without NVIDIA GPUs and is 10-100x slower than NCCL for GPU training.

---

## The Complete Picture

One training step, end to end:

```
Text → Tokenizer → [464, 3797, 3332, ...] (integers)
  → Embedding lookup → (8, 1024, 768)
  → 12x [Self-Attention → FFN]
  → LM Head → (8, 1024, 50257) logits
  → Softmax → probabilities
  → Cross-entropy vs targets → scalar loss
  → Backpropagation → 124M gradients
  → [All-reduce across GPUs if distributed]
  → AdamW optimizer (adaptive m/v states) + LR schedule → updated weights
  → Repeat
```

The model starts with random weights and meaningless predictions — loss around 10.8, equivalent to guessing among 49,000 tokens. Each step sharpens the weights slightly. After 100 steps on WikiText, loss drops to 7.1 — guessing among ~1,200 tokens. After billions of tokens, a full-sized model reaches perplexity 20-30, narrowing each prediction to a handful of plausible continuations.

Every operation in this pipeline is a matrix multiply. Every matrix multiply is parallelizable across thousands of GPU cores. That is why language models and GPUs are inseparable — not because GPUs are "fast computers," but because the fundamental unit of computation in a transformer is the one operation GPUs were physically designed to accelerate.

---

*Training scripts: [`train_single_node_baseline.py`](train_single_node_baseline.py) (single GPU) and [`train_distributed_node.py`](train_distributed_node.py) (multi-GPU DDP/FSDP). Data pipeline analysis: [`FEEDING_THE_BEAST_COMPACT.md`](analysis/FEEDING_THE_BEAST_COMPACT.md).*
