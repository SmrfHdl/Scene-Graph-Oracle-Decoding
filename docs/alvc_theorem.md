# ALVC: Information-Theoretic Lower Bound on Connector Length

> **Status:** Draft v0.1 — 2026-05-22. Derivation supersedes the original
> handoff statement `K* ≤ I/log(vocab)`, which was vacuous for continuous-valued
> vision tokens (an agent review on 2026-05-21 flagged this; see
> `HANDOFF_ALVC_PHASE0_SETUP_05_21_22_00.md`).
>
> **Audience:** Paper Section 3 (Theory). This document is the formal derivation
> of the converse bound that motivates adaptive K. Achievability — that ALVC
> + PonderNet training actually attains the bound to within a constant factor —
> is empirical and is the subject of Phase 1 experiments (G1–G4).

---

## 1. Setup

Let $(X, Q, Y) \sim \mathcal{D}$ be a random triple drawn from the data
distribution:

- $X \in \mathcal{X}$ — an image (raw pixels, or pre-vision-encoder features).
- $Q \in \mathcal{Q}$ — a text query.
- $Y \in \mathcal{Y}$ — the target response (token sequence or task-specific label).

The VLM pipeline decomposes as

$$
X \xrightarrow{\;\;f\;\;} Z \xrightarrow{\;\;g(\cdot,\,Q)\;\;} \hat{Y},
$$

where:

- $f: \mathcal{X} \times \mathcal{Q} \to \mathbb{R}^{K \times d}$ is the
  **connector** (linear projector, Q-Former, ALVC, …). It emits $K$ continuous
  vision tokens $Z = (Z_1, \dots, Z_K)$ each in $\mathbb{R}^{d}$.
- $g: \mathbb{R}^{K \times d} \times \mathcal{Q} \to \mathcal{Y}$ is the
  **frozen language model**.
- $K$ may depend on $(X, Q)$ — this is the ALVC degree of freedom. Fixed-K
  baselines (linear projector, Q-Former) have $K$ constant.

We measure reconstruction quality with a distortion
$\mathrm{d}: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\ge 0}$ (e.g.,
token-level cross-entropy) and define the achievable distortion at length $K$
as

$$
D(K) \;=\; \inf_{f,\,g \text{ s.t. output length}=K} \;
\mathbb{E}\bigl[\mathrm{d}(Y, g(f(X,Q),\,Q))\bigr].
$$

**Assumption A1 (per-token capacity).** There exists a finite constant
$C > 0$ such that, for any encoder $f$ producing tokens within the operating
manifold of the LM's input space,

$$
H(Z_k \mid Q) \;\le\; C \quad \text{for all } k = 1, \dots, K \text{ (in nats)}.
$$

Throughout, $H$ denotes differential entropy. We discuss how to estimate $C$
in §5; for the bound to be meaningful, $C$ must be finite, which it is whenever
the per-token distribution has bounded second moment (the maximum-entropy
distribution under a second-moment constraint is Gaussian, with finite
differential entropy).

**Assumption A2 (well-defined source rate).** $I(X; Y \mid Q) < \infty$. The
rate-distortion function $R_{Y \mid Q}(D)$ (defined below) is non-increasing
in $D$ and convex.

---

## 2. Main theorem

**Theorem (Converse bound on connector length).** *Under A1–A2, any VLM
pipeline that achieves expected distortion $\le D$ must use, on average over
$(X, Q)$,*

$$
\mathbb{E}[K] \;\ge\; \frac{R_{Y \mid Q}(D)}{C},
$$

*where the conditional rate-distortion function is*

$$
R_{Y \mid Q}(D)
\;=\; \inf_{P_{\hat Y \mid X, Q}: \, \mathbb{E}\mathrm{d}(Y,\hat Y) \le D}
I(X; \hat Y \mid Q).
$$

*In particular, when lossless reconstruction is required* ($D = 0$, with
$\mathrm{d}$ the discrete 0–1 loss on a finite alphabet), $R_{Y\mid Q}(0) =
I(X; Y \mid Q)$ *and*

$$
\boxed{\; \mathbb{E}[K] \;\ge\; \left\lceil \frac{I(X; Y \mid Q)}{C} \right\rceil. \;}
$$

This is the formal statement of the bound cited in the handoff.

---

## 3. Proof

We split the chain $X \to Z \to \hat Y$ and apply (i) data-processing
inequality, then (ii) entropy subadditivity, then (iii) the rate-distortion
converse.

### Step 1 — Data processing inequality.

Conditional on $Q$, the Markov chain

$$
X \;\to\; Z \;\to\; \hat Y
$$

holds (the LM only sees $Z$ and $Q$). Hence

$$
I(X; \hat Y \mid Q) \;\le\; I(X; Z \mid Q). \tag{1}
$$

### Step 2 — Bounding $I(X;Z \mid Q)$ by total entropy of $Z$.

By definition $I(X; Z \mid Q) = H(Z \mid Q) - H(Z \mid X, Q) \le H(Z \mid Q)$,
since differential entropy can be negative but
$H(Z \mid X, Q) \ge -\infty$… more carefully, in our setting $Z = f(X, Q)$ is a
deterministic function of $(X, Q)$, so $H(Z \mid X, Q) = -\infty$ in the
continuous case. To avoid the degeneracy, treat $f$ as a noisy channel: add a
vanishing noise floor so $Z = f(X, Q) + \epsilon$ with $\epsilon$ of variance
$\sigma_0^2 \to 0$. Then $H(Z \mid X, Q) = \frac{Kd}{2}\log(2\pi e \sigma_0^2)$
is finite, and the mutual information limit is the appropriate
information-theoretic quantity used in lossy compression literature
(Cover & Thomas, Ch. 10). In that limit,

$$
I(X; Z \mid Q) \;\le\; H(Z \mid Q) \;\le\; \sum_{k=1}^{K} H(Z_k \mid Q) \tag{2}
$$

by **subadditivity of entropy** (conditional version). Equality holds iff the
tokens are conditionally independent given $Q$ — they generally are not, since
both encode the same image; hence the bound has slack at high $K$.

### Step 3 — Per-token capacity (A1).

Applying A1 termwise:

$$
\sum_{k=1}^{K} H(Z_k \mid Q) \;\le\; K \cdot C. \tag{3}
$$

### Step 4 — Source-rate lower bound (rate-distortion converse).

The standard rate-distortion converse (Shannon 1959; Cover & Thomas Thm 10.2.1),
applied conditionally on $Q$, states that any encoder–decoder pair achieving
distortion $\le D$ satisfies

$$
I(X; \hat Y \mid Q) \;\ge\; R_{Y \mid Q}(D). \tag{4}
$$

(Intuition: the reproduction $\hat Y$ must retain at least $R_{Y\mid Q}(D)$ nats
of information about $X$, conditional on the query, to bring task loss below $D$.)

### Step 5 — Chaining.

Combining (1)–(4):

$$
R_{Y \mid Q}(D)
\;\overset{(4)}{\le}\; I(X; \hat Y \mid Q)
\;\overset{(1)}{\le}\; I(X; Z \mid Q)
\;\overset{(2,3)}{\le}\; K \cdot C.
$$

Taking expectation over $(X, Q)$ (both sides are random because $K$ may depend
on $(X, Q)$):

$$
R_{Y \mid Q}(D) \;\le\; \mathbb{E}[K] \cdot C \;\;\Longleftrightarrow\;\;
\mathbb{E}[K] \ge \frac{R_{Y \mid Q}(D)}{C}.
$$

For pointwise (per-sample) interpretation: if we require *pointwise* lossless
reconstruction at each $(x,q)$, the integer-K version $K(x,q) \ge \lceil
I(X;Y\mid Q=q) \cdot \mathbb{1}[X=x] / C \rceil$ holds in expectation; the
ceiling appears because $K$ must be a non-negative integer.

$\blacksquare$

---

## 4. When is the bound tight?

The chain has slack at three places:

- **Step 2 (subadditivity):** tight iff tokens are conditionally independent
  given $Q$. Real connectors output highly correlated tokens — Q-Former
  attention pools across all patches and Phi-3.5-V's projector preserves
  spatial correlations.
- **Step 3 (per-token capacity):** tight iff each token saturates capacity,
  i.e., its conditional distribution matches the entropy-maximizing
  distribution under the operating constraint. In practice, learned
  projector outputs cluster on a low-dimensional manifold of the ambient
  space and waste capacity.
- **Step 4 (rate-distortion):** tight only at the Shannon-optimal coding
  scheme. Frozen LM decoders are not optimal channel decoders.

**Practical consequence.** The bound is a converse (necessary condition).
ALVC's job is achievability: get within a constant factor of the bound. We
expect the bound to be loose by 2–10×, which is fine — the qualitative claim
is

$$
\mathbb{E}[K \mid \text{simple } (X,Q)] \;<\; \mathbb{E}[K \mid \text{complex } (X,Q)],
$$

driven by $I(X; Y \mid Q)$ varying across the data distribution. This is
testable empirically (gate G5).

---

## 5. Estimating $C$ in practice

$C$ is the maximum per-token entropy under operational constraints. Three
estimators, in order of looseness:

### 5a. Gaussian upper bound (loose, fast).

Under a second-moment constraint $\mathbb{E}\|Z_k\|^2 \le P d$ per token, the
max-entropy distribution is $\mathcal{N}(0, P I_d)$ with differential entropy

$$
H(Z_k) \;\le\; \frac{d}{2}\log(2\pi e P) \qquad \text{(nats)}.
$$

For Phi-3.5-V ($d = 3072$, $P \approx 1$ after layer-norm): $C \approx
\frac{3072}{2} \log(2\pi e) \approx 2{,}900$ nats. Use as a numerator
upper bound: $C_{\text{Gauss}} = (d/2)\log(2\pi e \hat\sigma^2)$ where
$\hat\sigma^2$ is estimated from connector outputs on a calibration set.

### 5b. Empirical Gaussian per-layer (tighter).

Fit a full multivariate Gaussian to the connector output distribution on a
calibration corpus, take $\frac{1}{2}\log\det(2\pi e \hat\Sigma)$.
$\hat\Sigma \succeq 0$ is the empirical covariance of $Z_k$ across $(X, Q)$ and
positions $k$. This captures anisotropy of the learned manifold and is
typically smaller than 5a by a factor of 2–5×.

### 5c. KSG / k-NN entropy estimator (tightest, slow).

For dimensions up to a few hundred, Kraskov–Stögbauer–Grassberger entropy
estimators (Kraskov et al. 2004) give consistent non-parametric estimates.
At $d = 3072$ this is statistically infeasible without a learned projection
first. **Strategy for paper:** project $Z_k$ to a $d' = 32$ subspace via
random projection (Johnson-Lindenstrauss preserves entropy up to a
constant) and estimate $H$ in the low-dim space.

### Numerator: estimating $I(X; Y \mid Q)$.

We don't need the absolute value; we need the *variability* across
$(x, q)$. For G5 we estimate the conditional via

$$
\widehat{I}(X; Y \mid Q = q) \;\approx\; \mathbb{E}_X[-\log p_g(Y \mid X, Q=q)]
\;-\; \mathbb{E}_X[-\log p_g(Y \mid Q=q)]
$$

(difference between LM with-image and without-image NLL on the same target).
This is a lower bound on $I$ since the frozen LM is a suboptimal decoder, but
is sufficient to rank queries by complexity.

---

## 6. ALVC as achievability

The theorem is a *converse*. ALVC is the corresponding *achievability* scheme:

- **Encoder side:** $f$ is learned (importance scorer + projector), so that
  the top-$K$ tokens by importance saturate the per-token capacity as much
  as possible.
- **Halting side:** $K$ is chosen per-sample by a PonderNet head conditioned
  on $(X, Q)$, trained with the rate-distortion training objective

$$
\mathcal{L}(\phi)
\;=\; \mathbb{E}\Bigl[\sum_n p_n(\phi) \, \mathcal{L}_\text{task}(K=n)\Bigr]
\;+\; \beta \, \mathrm{KL}\bigl(p_n \;\|\; \mathrm{Geom}(\lambda_p)\bigr),
$$

where $p_n$ is the halting distribution output by `HaltingHead` and the
geometric prior prevents collapse to $K=K_{\max}$. See
[sgod/policies/alvc/halting.py](../sgod/policies/alvc/halting.py).

**Open empirical question (G1):** does the trained $\mathbb{E}[K]$ track
$I(X;Y\mid Q)/C$ across samples, or does halting collapse to a constant?
A negative answer means the achievability gap is large and the paper's
qualitative claim fails. This is the week-1 gate.

---

## 7. Caveats & open questions

1. **Sequence-level vs per-patch halting.** Theorem assumes a single $K$ per
   sample. Per-patch halting (Phase 2) gives a tighter achievability, but
   the converse bound is unchanged ($\mathbb{E}[K]$ is still the relevant
   quantity).

2. **Token correlation is not modelled.** Subadditivity is loose by exactly
   the *total correlation* of $Z$: $C(Z) = \sum_k H(Z_k) - H(Z)$. A tighter
   theorem would replace $K \cdot C$ with $H(Z \mid Q)$ directly and
   estimate the latter. Defer to v2 of the paper if needed; the loose form
   is sufficient to motivate adaptive K.

3. **$C$ is operating-point dependent.** A frozen LM imposes constraints
   on the input distribution that are non-trivial to characterize (the
   "manifold" of valid soft tokens). Our estimators assume operating
   *inside* that manifold; outside, the LM's behavior is undefined and
   capacity arguments break.

4. **$I(X; Y \mid Q)$ depends on the target $Y$.** For free-form generation,
   $Y$ is a long token sequence and $I$ blows up; for single-answer QA
   (POPE, MMBench), $Y$ is short and $I$ is small. Choose evaluation
   tasks where $Y$ is single-answer to keep the bound informative.

5. **The bound is asymptotic.** Strictly speaking, rate-distortion is a
   block-coding statement (large i.i.d. blocks of $(X,Q,Y)$). For a single
   instance, the gap between the random-coding achievability and the
   one-shot regime is $O(1)$ — fine for our qualitative argument, not
   fine for tight numerical predictions.

---

## 8. Relation to prior work

| Method | $K$ | Query-conditioned? | Theoretical bound? |
|---|---|---|---|
| Linear projector (LLaVA-1.5) | fixed, $=N$ | no | — |
| Q-Former (BLIP-2) | fixed, $=32$ | no (image only) | — |
| Perceiver IO | fixed | no | — |
| LLaVA-PruMerge (2403.15388) | adaptive | no (image only) | heuristic |
| Matryoshka M3 (2405.17430) | adaptive | no (image only) | empirical |
| **ALVC (ours)** | **adaptive** | **yes** | **§2 above** |

The query-conditioned dimension is the single point of theoretical novelty —
no prior work motivates adaptive $K$ through $I(X; Y \mid Q)$ rather than
$I(X)$ (the image entropy). This is what G5 tests.

---

## 9. References

- Cover & Thomas, *Elements of Information Theory*, 2nd ed., Chapters 8, 10.
- Kraskov, Stögbauer & Grassberger 2004, "Estimating Mutual Information,"
  *Phys. Rev. E*.
- Shannon 1959, "Coding theorems for a discrete source with a fidelity
  criterion."
- Banino et al. 2021, "PonderNet: Learning to Ponder," arXiv:2107.05407.
- Cha et al. 2024, "Honeybee: Locality-enhanced Projector for Multimodal LLM" —
  closest fixed-$K$ ablation; orthogonal to ALVC.

---

*Last edited: 2026-05-22. Next revision after Phase 0 G5 measurement lands —
will replace §5 numerical placeholders with calibrated values from Phi-3.5-V.*
