# InfoTheory: Ultimate Technical Manual & Specification

`infotheory` is an industrial-strength suite for **Information-Theoretic Estimation**, **Sequential Prediction**, and **Universal AI (MC-AIXI)**. This document serves as a complete standalone specification of the codebase.

---

## 1. Core Architecture (`src/lib.rs`)

The library is centralized around the `InfotheoryCtx` struct, which manages the preferred backends for prediction and compression.

### 1.1 Predictive Backends (`RateBackend`)
Used to estimate entropy rates ($\hat{H}$) and symbol probabilities $P(s)$.
- **ROSA (RosaPlus)**: A Suffix Automaton model. Best for byte-level sequence modeling.
- **CTW (Context Tree Weighting)**: Binary predictor using KT-estimators and tree mixing.
- **RWKV7**: A recurrent neural network backend for high-density neural prediction.

### 1.2 Compression Backends (`NcdBackend`)
Used to estimate Kolmogorov Complexity ($C(x)$).
- **ZPAQ**: High-ratio Lempel-Ziv style compressor with arithmetic coding.
- **RWKV7**: Neural compression via predictive coding.

### 1.3 Feature Matrix
| Capability | Backend Support |
| :--- | :--- |
| **Entropy Rate** | ROSA, CTW, RWKV7 |
| **Cross Entropy** | ROSA, CTW, RWKV7 |
| **Compression** | ZPAQ, RWKV7 |
| **NCD** | ZPAQ, RWKV7 |

---

## 2. Mathematical Manual

### 2.1 Shannon Information Primitives
All measures are in **bits** (base 2).

- **Marginal Entropy ($H_0$)**: Treats data as a bag-of-bytes.
  $$H_0(X) = -\sum P(x) \log_2 P(x)$$
- **Entropy Rate ($\hat{H}$)**: Accounts for sequential dependencies.
  $$\hat{H}(X) = \lim_{n \to \infty} \frac{H(X_1...X_n)}{n}$$
- **Mutual Information ($I$)**:
  $$I(X;Y) = H(X) + H(Y) - H(X,Y)$$
- **Conditioned Cross-Entropy**:
  The bits required to code $X$ given that the model was first trained on $Y$.

### 2.2 Domain-Specific Metrics
- **Intrinsic Dependence (ID)**: Measures how much of the data's complexity is explained by its sequential structure vs its unigram distribution.
  $$ID(X) = \frac{H_0(X) - \hat{H}(X)}{H_0(X)}$$
- **Resistance to Transformation (R)**: Measures the information preserved across a transformation $T$.
  $$R(X, T) = \frac{I(X; T(X))}{H(X)}$$

### 2.3 Metric Distances
- **NCD (Normalized Compression Distance)**:
  $$NCD(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}$$
  Implemented in four variants: `Vitanyi`, `SymVitanyi`, `Cons`, and `SymCons`.
- **Total Variation Distance (TVD)**: L1-distance on the probability simplex $[0, 1]$.
- **Normalized Hellinger Distance (NHD)**:
  $$\text{NHD} = \sqrt{1 - \sum \sqrt{P(x)Q(x)}}$$

---

## 3. Sequence Prediction Backends

### 3.1 Context Tree Weighting (CTW)
Implemented in `src/ctw.rs`.
- **Tree Structures**: Each bit is pushed through a tree of depth $D$.
- **Mixing**: $P_{mixed} = \frac{1}{2} P_{KT} + \frac{1}{2} P_{children}$.
- **Consistency**: Uses a **Padded Context** (zero-padding) to ensure valid updates from the very first bit of history.
- **Log-Space Math**:
  $$\ln P_w = \ln P_{child\_sum} + \ln(1 + \exp(\ln P_{KT} - \ln P_{child\_sum})) - \ln 2$$

### 3.2 ROSA (Rapid Online Suffix Automaton)
Implemented in the `rosaplus` crate.
- Uses **Suffix Automaton (SAM)** for $O(1)$ state updates.
- **Rollback**: Supports atomic "transactions" to try hypothetical updates and revert them without rebuilding the tree.

---

## 4. Monte Carlo AIXI (MC-AIXI)

### 4.1 The AIXI Equation
Universal AI maximizes the expected reward $r$ over all computable environments.
$$v^* = \max_{a} \sum_{e} \max_{a'} \sum_{e'} ... (r + r' + ... ) P(e|ha)$$

### 4.2 MC-AIXI Implementation (`src/aixi/`)
1.  **Model**: A `Predictor` (CTW/ROSA) provides $P(e|ha)$.
2.  **Planner**: **MCTS** (Monte Carlo Tree Search) approximates the sum-max tree.
    - **UCT Formula**: Balances exploration and exploitation during tree traversal.
    - **Rollout**: Performs random "playouts" using the internal Predictor to sample future rewards.
3.  **Agent Loop**: The agent performs MCTS simulations, selects the best action, acts in the environment, and then updates its Predictor with the true outcome.

### 4.3 Symbol Encoding
AIXI predictors are binary.
- **Observations/Rewards**: Mapped to bit-streams using a fixed bit-width.
- **Actions**: Similarly mapped to bits.
The predictor sees a single unified sequence of bits: $a_1 a_2 ... a_k e_1 e_2 ... e_n a'_1 a'_2 ...$

---

## 5. Semantic Search Engine (`src/search.rs`)

A high-performance search pipeline using Information Theory as a ranking function.

### 5.1 Three-Stage Pipeline
1.  **Stage 0 (Prefilter)**: Rapid unigram cross-entropy filter.
2.  **Stage 1 (Conditional)**: Likelihood Gain ranking conditioned on a **Universal Prior**.
    $$\text{Score}(x) = H(\text{Query} | \text{Prior}) - H(\text{Query} | \text{Prior} + x)$$
3.  **Stage 2 (KMI)**: Precise reranking using **Kolmogorov Mutual Information**.
    $$KMI(q, x) = C(q) + C(x) - C(qx)$$

### 5.2 Search Options
- **Granularity**: `Snippet` (sliding windows) or `File`.
- **Universal Prior**: A corpus used to "normalize" the search by ignoring common patterns.
- **Prior Modes**: `UsePrior`, `NoPrior`, or `SummarizePrior` (recursively selects the best document from the prior as a context).

---

## 6. CLI Reference (`src/main.rs`)

The `infotheory` binary is the primary gateway to the library features.

### 6.1 Subcommands
- `ncd <file1> <file2>`: Compute distance between two files.
- `mi <file1> <file2>`: Compute Mutual Information.
- `entropy <file>`: Compute Entropy Rate.
- `search <query> <target>`: Run the semantic search engine.
- `aixi <config.json>`: Start an AIXI agent in a specified environment.
- `batch`: JSON-in/JSON-out REPL for high-throughput metric requests.

### 6.2 Global Flags
- `--rate-backend [ctw|rosa|rwkv]`: Selection of prediction model.
- `--ncd-backend [zpaq|rwkv]`: Selection of compression model.
- `--method <val>`: Specific settings for the backend (e.g., ZPAQ level `1`-`9`, CTW depth).
- `--max-order <n>`: Maximum context window for sequential models.

---

## 7. Quality Assurance (`src/axioms.rs`)

Every measurement is subjected to **Axiomatic Validation**:
- **Identity**: $NCD(x,x) \approx 0$
- **Symmetry**: $I(X;Y) = I(Y;X)$
- **Triangle Inequality**: $NCD(x,z) \le NCD(x,y) + NCD(y,z)$
- **Non-Negativity**: $I(X;Y) \ge 0$
- **Oracle Compliance**: Comparison against analytic Bernoulli and Markov ground truths.


## Potential new ideas:
The "Safe-World" Universal Prior (Or "Evil World")
Instead of just comparing File A to File B, you use your --prior flag to build a model of the "Set of All Legitimate Softwares" (The GNU C Library, Windows APIs, common frameworks).

The Groundbreaking Bit: You aren't searching for "Malware Signatures." You are quantifying Extrinsic Information Density (EID).
The Logic: If a 10KB block in a file has information that is Extremely Deviant from the "Safe Prior" but has High Intrinsic Dependence (meaning it's structured, not random noise), it is a novel exploit. Mathematically, you are detecting "Alien Logic."

