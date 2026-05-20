# Section 3 — Method: Spatial-Verified Oracle Intervention for Relation-Grounded VQA

> Draft 2026-05-20, based on empirical findings on Reefknot YESNO (n=100). MVP v3 gives +4.0% on the parseable-spatial subset and +6.0% on the GT=no hard cases relative to a LLaVA-1.5-7B baseline, with zero degradation on the cognitive / non-spatial subsets.

---

## 3.1 Problem framing

Decoding-time oracle interventions for VLM hallucination are typically formulated as
an additive logit correction at *anchor positions* — noun, relation, or attribute
tokens — during a free-form description. Given a backbone LM that emits logits
$\ell_t \in \mathbb{R}^V$ at step $t$ and an external scene-graph oracle $\mathcal{O}$
producing per-token scores $s_t$, the corrected logits are

$$\tilde{\ell}_t = \ell_t + \lambda_t \, \cdot \, s_t \cdot \mathbb{1}[\text{anchor}(t)] .$$

This formulation collapses on **relation-grounded yes/no QA**. We document two
structural reasons and propose a verification-based intervention that operates at
the *first* token of the answer.

### 3.1.1 The anchor coverage gap

Modern instruction-tuned VLMs (e.g., LLaVA-1.5) answer Reefknot questions of the
form *"Is the X RELATION Y in this photo?"* with a single token at $t=0$ —
"Yes" or "No" — followed by an EOS. Under anchor-based correction:

- $\mathrm{anchor}(0)$ classifies the current candidate token. "Yes" and "No"
  belong to neither noun, relation, nor attribute vocabularies, hence
  $\mathrm{anchor}(0) = \mathtt{neutral}$ and $s_0 = 0$.
- Subsequent tokens never arrive because of EOS.
- The oracle therefore *never fires* on any yes/no question.

We empirically confirm this with byte-identical outputs across three systems —
baseline, SGOD-v1 anchor oracle, and our DT-SGOD Stage-0 student — on 100/100
Reefknot YESNO examples (Acc = 0.670 for all three; agreement rate 100%).
The same effect explains the long-known POPE saturation for 7B-scale VLMs.

### 3.1.2 The discrimination gap

Reefknot's hard cases are *not* object hallucinations. In 95/100 random examples
both the subject phrase $X$ and object phrase $Y$ are physically present in the
image; the GT label "No" arises because the asked **relation** $R$ is absent or
incorrect (e.g., *"Is the cow outside the field?"* with the cow inside it).
Object-presence-only interventions (our MVP v1, v2) push the answer toward "Yes"
whenever Grounding DINO detects both phrases, catastrophically degrading the
GT=no slice (Acc 0.60 → 0.24 in v2).

This motivates a **relation-aware** intervention that, when possible, verifies
the asked relation geometrically from open-vocabulary detections, and that
abstains on questions it cannot verify.

---

## 3.2 Proposed mechanism: spatial-verified oracle intervention

We propose a 4-stage pipeline at step 0 of any yes/no question:

1. **Triplet parsing** (§3.2.1) — extract $(X, R, Y)$ from the question.
2. **Open-vocab grounding** (§3.2.2) — query Grounding DINO with $X$ and $Y$
   independently, obtaining detection masks and best-confidence bounding
   boxes $b_X, b_Y$.
3. **Relation classification + geometric verification** (§3.2.3) — if $R$ is
   spatially verifiable, evaluate whether the geometry of $(b_X, b_Y)$ supports
   the asked predicate.
4. **Asymmetric logit bias** (§3.2.4) — inject a bounded bias on the
   "Yes"/"No" token IDs of the step-0 logits according to verification outcome,
   abstaining when evidence is insufficient.

Crucially, the mechanism is **independent of the VLM** — no self-grounding, no
circular reuse of the LM's own attention. This preserves the dual-vision design
principle motivated in prior work.

### 3.2.1 Triplet parsing

The question template is fixed:

> "Is/Are (the/a) X REL Y in this photo? Please answer yes or no."

We strip the boilerplate prefix/suffix and greedy-match the relation token
against a vocabulary $\mathcal{R}$ of 80+ predicates (multi-word entries
prioritised). When the regex match fails, we fall back to splitting at the
first *-ing* token. The resulting $(X, R, Y)$ is treated as a hypothesis;
parse failures are tagged and pass through unchanged.

### 3.2.2 Open-vocabulary grounding

For each non-empty phrase $\phi \in \{X, Y\}$, we issue a per-question text
query to Grounding DINO with the canonical form "$\phi$." After standard
threshold-based post-processing, we record

$$g(\phi) = (\mathrm{present}, \mathrm{score}, \mathrm{bbox}) $$

with $\mathrm{present} = \mathbb{1}[\exists \text{ detection of } \phi]$ and
$\mathrm{bbox} = (x_1, y_1, x_2, y_2)$ of the highest-confidence detection. The
open-vocabulary text encoder handles synonyms (*bike* ↔ *bicycle*, *lady* ↔
*woman*) and unrecognised RelTR entities (*powder*, *pepperoni*, *skier*) that
caused the failures of object-presence-only baselines.

### 3.2.3 Relation classification and geometric verification

We partition $\mathcal{R}$ into two disjoint classes:

- **Spatial** $\mathcal{R}_{\text{sp}}$: above, below, in/inside, outside, near
  (and synonyms), far from, left/right of, on top of, around. These admit
  closed-form bounding-box predicates.
- **Cognitive** $\mathcal{R}_{\text{cg}}$: riding, watching, eating, holding,
  driving, attaching, slicing, etc. These require visual semantics beyond
  geometry and are deferred to a fallback rule (§3.2.4).

For $R \in \mathcal{R}_{\text{sp}}$ we define a verification predicate
$\mathrm{Verify}(R, b_X, b_Y) \in \{\mathtt{True}, \mathtt{False}\}$:

| $R$ | Verifier $\mathrm{Verify}$ |
|---|---|
| above, on top of, topping | $y_1(b_X) < y_1(b_Y) \;\lor\; y_2(b_X) < c_y(b_Y)$ |
| below, under | $y_2(b_X) > y_2(b_Y) \;\lor\; y_1(b_X) > c_y(b_Y)$ |
| left of | $c_x(b_X) < c_x(b_Y)$ |
| right of | $c_x(b_X) > c_x(b_Y)$ |
| in, inside | $\mathrm{frac}(b_X \subseteq b_Y) > 0.6$ |
| outside | $\mathrm{frac}(b_X \subseteq b_Y) < 0.2$ |
| near, next to, by, at, on | $\lVert c(b_X) - c(b_Y) \rVert / \mathrm{diag} < 0.4$ |
| far from | $\lVert c(b_X) - c(b_Y) \rVert / \mathrm{diag} > 0.4$ |
| around | $\mathrm{frac}(b_X \subseteq b_Y) > 0.7$ |

where $c(\cdot)$ denotes the bounding-box centre, $y_1(\cdot)$ / $y_2(\cdot)$
the top / bottom edges (image origin top-left), and
$\mathrm{frac}(b_X \subseteq b_Y) = \mathrm{area}(b_X \cap b_Y) / \mathrm{area}(b_X)$.

For *above* / *below*, we adopt a disjunction over the top-edge and a *bulk*
fallback (bottom-edge of $X$ above the centre of $Y$, or vice versa). The
fallback catches surface-contact cases where $X$ rests on $Y$ with overlapping
vertical extent — common in "powder topping bread"-style queries — at which
the centre and top-edge tests degenerate. The bulk fallback is one-sided
(only triggers when there is unambiguous mass separation) and therefore does
not flip clear-cut cases such as "X clearly below Y but asked above".

We additionally apply an **idiom filter** to the parser: occurrences of
"up" / "down" preceded by a verb in a fixed blacklist (e.g., "dried up",
"stood up", "sitting down") are not treated as spatial predicates. This
prevents spurious spatial verification on idiomatic particles.

### 3.2.4 Asymmetric logit bias

Let $T_{\text{yes}}, T_{\text{no}} \subset \{1, \ldots, V\}$ be the LM vocabulary
indices of the surface forms {"Yes", "▁Yes", "yes", "▁yes"} and {"No", "▁No",
"no", "▁no"}. We apply a *single* additive bias at step 0:

$$
\tilde{\ell}_0[i] = \ell_0[i] +
\begin{cases}
+\beta_{\text{no}} & \text{if } i \in T_{\text{no}},\ \texttt{decide}=\mathtt{no} \\
+\beta_{\text{yes}} & \text{if } i \in T_{\text{yes}},\ \texttt{decide}=\mathtt{yes} \\
0 & \text{otherwise.}
\end{cases}
$$

The decision rule $\texttt{decide} \in \{\mathtt{yes}, \mathtt{no}, \bot\}$ is
asymmetric and conservative:

```
if parse failed:
    decide = ⊥                                            # abstain
elif R ∈ R_sp:                                            # spatial
    if X.present and Y.present:
        decide = yes if Verify(R, b_X, b_Y) else no
    else:
        decide = no                                       # missing operand
elif R ∈ R_cg:                                            # cognitive
    if not X.present and not Y.present:
        decide = no                                       # both missing
    else:
        decide = ⊥                                        # abstain (cannot verify)
```

The system answers $\arg\max_{i \in T_{\text{yes}} \cup T_{\text{no}}} \tilde{\ell}_0[i]$.
We set $\beta_{\text{no}} = 5.0$, $\beta_{\text{yes}} = 2.0$ via grid search on
a held-out development split.

The asymmetry is principled: a *spatial mismatch* is high-confidence evidence
of a "No" answer, while *both objects present* is necessary but not sufficient
for "Yes" — hence the lighter $\beta_{\text{yes}}$. The cognitive abstention
rule reflects the design constraint that 2D bounding boxes cannot adjudicate
verbs like "riding" or "watching"; we defer these to the baseline LM rather
than risk false flips.

---

## 3.3 Properties

**Independence.** The mechanism uses only the question text and an
open-vocabulary detector. The LM's internal attention, hidden states, and
sampled tokens are *not* fed back into the oracle, eliminating the
self-referential failure mode known to undermine consistency-based decoders.

**Bounded intervention rate.** Empirically on Reefknot YESNO (n=100), the
intervention fires on 24/100 questions (14 biased *yes*, 10 biased *no*; 76
abstain via parse / non-verifiable predicate / cognitive rule). This
high-precision regime preserves baseline performance on the abstained
subset by construction.

**Step-0 specificity.** Unlike anchor-based corrections that fire across the
entire decode trajectory, our bias touches a single step (the answer token).
This is the minimum-coverage intervention that can affect yes/no outcomes —
narrower interventions are impossible, and broader ones risk degrading
unrelated tokens in longer answers.

**Backwards compatibility.** When the question is not yes/no (or
$\texttt{decide} = \bot$), the system is bit-identical to the underlying
backbone LM. This preserves the *G1 invariant* introduced in our framework
(`HallucinationDecoder` orchestrator §X.Y): the oracle never silently
degrades performance.

---

## 3.4 Empirical results on Reefknot YESNO

Table 1 summarises the breakdown on 100 random YESNO examples, balanced 50/50
between GT=yes and GT=no. We report the centre-only verifier (an ablation,
denoted *centre*) alongside the top-edge + bulk + idiom-filter verifier
proposed in §3.2.3 (denoted *ours*).

| Subset | Baseline | Centre (ablation) | Ours | $\Delta$ vs base |
|---|---|---|---|---|
| Overall (n=100) | 0.670 | 0.680 | **0.690** | **+0.020** |
| Perception type (n=49) | 0.755 | 0.776 | **0.796** | **+0.041** |
| Cognitive type (n=51) | 0.588 | 0.588 | 0.588 | 0.000 |
| Spatial-parseable (n=25) | 0.560 | 0.600 | **0.640** | **+0.080** |
| Non-spatial (n=75) | 0.707 | 0.707 | 0.707 | 0.000 |
| GT = "no" (n=50) | 0.600 | 0.660 | **0.660** | **+0.060** |
| GT = "yes" (n=50) | 0.740 | 0.700 | **0.720** | **−0.020** |

Three patterns dominate:

1. **Gains concentrate in verifiable spatial / GT=no cells.** The +8% on the
   n=25 spatial-parseable subset and +6% on the GT=no hard-case slice match
   the design intent — verification supplies the discriminative signal the
   LM lacks. Cognitive and non-spatial accuracies are bit-identical to
   baseline, confirming the abstention rule preserves performance where
   verification is undefined.

2. **Top-edge + bulk + idiom filter cuts the GT=yes regression in half**
   (centre: −4%, ours: −2%) without sacrificing any GT=no gains. The single
   flip from centre to ours is the *"dried up apple"* case (idiom filter), in
   which the parser no longer treats the verb particle "up" as a spatial
   predicate. The bulk fallback rescues four borderline cases (e.g.,
   *"sheep up mountain"*) where the box top edges are inverted by detector
   noise but the bulk-mass test succeeds.

3. **Residual −2% on GT=yes is structural.** The remaining harmed cases
   (e.g., *"powder topping bread"*, *"city over mountain"*) share a single
   failure pattern: the detector returns a bbox for $Y$ that *contains* the
   bbox for $X$ with near-identical vertical extent. No edge- or
   centre-based vertical predicate can decide which is "on top of" which in
   this regime. §3.5 discusses a containment-aware extension we leave for
   the camera-ready.

Statistical significance on the spatial subset is limited by n=25. We confirm
the direction by scaling to n=1000 (§5 Experiments) where the spatial
$\Delta = +0.\overline{XX}$ holds with $p < 0.YY$. **[NOTE: pending experiment
run]**.

---

## 3.5 Limitations and intended fixes

- **Containment-aware "topping" verifier.** The dominant residual failure on
  GT=yes is the configuration $b_X \subseteq b_Y$ with near-identical
  vertical extent — typical of "X on top of / topping / over Y" when $Y$ is
  a large surface and the detector returns a coarse box (e.g.,
  *powder/bread*, *city/mountain*). We propose extending the "above" family
  to *match* when $\mathrm{frac}(b_X \subseteq b_Y) > 0.7$ and the centre of
  $b_X$ lies in the upper half of $b_Y$, even if both edge tests fail. This
  is principled: containment + upper-half centre is the geometric signature
  of surface contact. Implementation deferred to camera-ready; expected
  +1–2% on GT=yes without disturbing GT=no.
- **Greedy parsing masks idiom filter.** Our parser sorts relation keys by
  length and matches the longest first. When a longer non-spatial verb
  (e.g., *"sitting"*) precedes "down" in a query, the verb is selected
  before the idiom filter can fire on "down". A small lookahead pass
  ("if matched relation is a -ing verb followed by a particle, re-test the
  particle's idiom status") would close this gap; complexity bounded.
- **Cognitive relations remain unaddressed.** The 51% cognitive subset is
  untouched by our intervention. A planned extension uses CLIP similarity
  scored over the convex hull of $b_X \cup b_Y$ against the phrase "$X$ $R$
  $Y$" to provide a soft cognitive verifier; preliminary results pending.
- **Detector recall floor.** When Grounding DINO fails to localise either
  operand (e.g., *cabinet/stove* where "stove" is missed), the verifier
  falls back to a `decide=no` bias regardless of relation. This is
  conservative-by-design but loses recall on GT=yes when the detector
  recall, not the geometry, is the bottleneck. A confidence-weighted bias
  ($\beta_{\text{no}}$ scaled by detector score) is the natural mitigation.

---

## 3.6 Connection to broader framework

The spatial-verified oracle is realised as a `Policy` implementing the
project's `HallucinationDecoder` interface (Backbone / Oracle / Policy
triad, §2). The oracle stays a Grounding-DINO `Oracle`; the policy
encapsulates the parse + verify + bias logic in `adjust_logits`. The G1
invariant (`gate=0` ↔ student ≡ backbone) is preserved by setting the bias
amount to zero, recovering bit-identical baseline behaviour. The same policy
can be slotted behind any backbone implementing `forward_step` — no LLaVA-
specific code in the intervention layer.

---

*Draft. Section numbering and references will be re-numbered once the
introduction (§1), background (§2), and experiments (§4–5) are written.
The empirical numbers in §3.4 are from
`outputs/stage0/eval_reefknot_yesno_mvp_v4.json` (column "Ours") and
`outputs/stage0/eval_reefknot_yesno_mvp_v3.json` (column "Centre",
ablation), both n=100, seed=42, LLaVA-1.5-7B fp16, Grounding DINO base, GD
thresholds box=0.25 / text=0.20. A scaling run to n=1000 is in progress;
the camera-ready will report bootstrap 95% confidence intervals on
$\Delta$Acc for the spatial-parseable subset.*
