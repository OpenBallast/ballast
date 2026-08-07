# Retrieval: the realized band, and the two-pass architecture

*Part of the [OpenBallast working notes](../THESIS.md). Two measurement regimes live here. Part 1 de-oracles the matrix headline: logprob MC probes over the T0 triple corpus, with a real entity linker in the loop. Part 2 is the prose-retrieval research line: open-ended generation over a full-body Wikipedia prose corpus on natural benchmarks (SimpleQA, NQ-open, PopQA), where retrieval is genuinely hard, and where the two-pass architecture was found and validated.*

## Part 1: Realized retrieval, de-oracling the headline

*(Correction, 2026-08-01: an earlier version of this section reported that wrong entity links were as harmful as right links were helpful, and on that basis ranked precision-first linkers highest at ~34% realization. That harm number came from a padding-misaligned scoring harness, caught within hours by cross-checking against the banked per-probe passes, and reversed on re-measurement. The corrected numbers below supersede it; the audit trail lives in the research repo.)*

The matrix assumes perfect entity resolution. We measured what a real, non-generative linker (capitalized-span mining + normalized name index) actually realizes on the same 50,147 probes, and priced its failure modes directly:

- A linked probe resolving to the **right** entity realizes that probe's grounded outcome (+17 accuracy points on the hit slice, hallucination halved). A **miss** links nothing and falls back to ungrounded (free).
- A **wrong link** attaches a homonym's facts; measured on a dedicated GPU pass over 1,025 wrong-linked probes, it is **approximately neutral** (accuracy 0.311 → 0.325): the model largely ignores evidence irrelevant to the question. Hedged framing ("may be unrelated, ignore if so") changes nothing in either direction.
- Because wrong links are cheap and misses forgo gain, **hit-rate is the objective**, and recall machinery pays. Composing per-probe over seven linker variants (E2B bf16, wrong links conservatively scored as exactly neutral): the naive precise linker realizes **52%** of the oracle gain; adding context disambiguation, lowercase fallback mining, and FTS5 fuzzy matching lifts the band to **63–66%**. The shipped CLI and demo endpoint run a mid-band configuration (~63%).

**What the floor does to the headline crossings.** At ~63% realization, E2B + the full 1.51 GB ballast lands at ≈0.77 end-to-end against the 12B's raw 0.683. The full-corpus crossing holds comfortably. The sharper oracle claim "E2B + 180 MB beats the 12B raw" moves to **E2B + ~470 MB (L5)** at the realized floor (L3 composes to ≈0.67, just under; L5 to ≈0.71). The byte advantage versus the ~19.4 GB bf16 parameter route remains ~40×, or ~15× if the 12B is instead charged at Q4_K_M, the cheapest quant that leaves it intact ([quantization results](results-quantization.md)). Two further qualifications, in opposite directions: the probe setting (mining mentions from raw natural-language trivia) is close to worst-case retrieval, and the agent path (the model calls `resolve("Douglas Adams")` with a clean mention, resolved at ~88%) sits near the ceiling, not the floor; conversely, these compositions inherit the harm sample's CIs and the probe set's popularity mix.

This turns the former "oracle retrieval" caveat into a measured band: the corpus's value is the ceiling, a weekend-grade linker holds about two thirds of it, and the remaining gap is an engineering ladder (typed disambiguation, retrieval-quality levels R0–R7) with a per-rung measurement in place, not an assumption.

## Part 2: Prose retrieval and the two-pass support-routed architecture

### The instrument: three arms and a delivery ratio

Every result below is read through a three-arm decomposition, run per probe on the same model and corpus:

- **U** (ungrounded): the model alone;
- **R** (realized): the model with the production retrieval pipeline;
- **S** (saturated): the model handed the true source passage directly.

The **delivery ratio DR = (R − U) / (S − U)** measures how much of the knowledge gap the retrieval stack actually delivers: S − U is what the corpus could be worth, R − U is what the pipeline realizes. The decomposition separates "the model can't use the fact" (rare) from "retrieval never delivered it" (the dominant failure), and it is what makes every kill below mechanically attributable.

Protocol and anchors: the generator throughout Part 2 is **Qwen3.5-9B-Base (int8)**, answering open-ended in ≤24 tokens, graded by normalized alias containment (SQuAD/NQ-open convention), a different and harder instrument than Part 1's logprob multiple choice, so numbers do not compare across parts. On the 2,000-probe development population (G2 = 1,000 SimpleQA + 1,000 NQ-open): **U = 0.1540**, single-pass R = 0.2885, two-pass R = 0.3315, and the saturated ceiling S = 0.6805, depressed by benchmark gold noise (roughly a third of official gold passages do not support their official answers as graded), not by the model.

### The transfer law: ten uniform pack policies, ten kills

Stage F asked whether smarter packing (graph-neighbor expansion, typed Wikidata statements, semantic (cosine) ordering, margin gates) could deliver evidence to the probes the baseline retrieval misses (the "gap" population) without harming the probes it already serves (the "covered" population). Every screen was registered with bands and kill lines before its run. **Ten registered screens were killed on that joint contract**, and the pattern is one law measured from ten angles:

| screen (uniform policy) | gap gain | covered cost |
|---|---|---|
| S6 competitive admission | +12.70pp | −11.89pp |
| S6b two-block cap | +10.98pp | −9.54pp |
| S6c margin gate m=1.5 (calibrated) | +5.61pp | −0.83pp |
| S8 cosine admission (δ=0) | +17.16pp | −9.82pp |
| S8 cosine reorder-only | +0.69pp | −9.13pp |
| recovery-v2 neighbor widening | +0.02pp (projected) | — |
| recovery-v2 + statement blocks | +0.18pp (projected) | — |

**Uniform admission is a ~1:1 wealth transfer** (+12.7 gap / −11.9 covered; exchange rate 1.07:1). The best pack-time separator found, the per-chunk score margin, improves the exchange rate to 6.8:1, but the registered contract point (+8pp gain at ≤1pp cost) lies just outside its achievable frontier. The reason is structural: **no pack-time observable knows whether the current evidence already suffices.** On covered probes retrieval was already right, so every added candidate is pure risk; on gap probes the same admission rule starves. The kills are first-class results: they establish that the wall is real, not that the experiments were unlucky.

### The signal that lives, and the architecture built on it

The entity that does know whether the evidence sufficed is the model at generation time, read not from its confidence but from its transcript. **Is the pass-1 answer supported (as a normalized substring) by the evidence it was generated from?** That string test separates confabulations from grounded answers at **AUC 0.830**, where mean token logprob manages **0.561**, barely above coin.

The composed policy: pass-1 runs the production pack untouched; support-miss rows regenerate once against a max-recall *recovery pack* (whose dilution is irrelevant on rows whose pass-1 already failed); a second support test accepts pass-2 only when the recovery pack supports it. The wealth transfer disappears because the two packs never compete for the same probe.

| cell | composite accuracy | registered | guard (pass-1-correct net loss) |
|---|---|---|---|
| G2 (SimpleQA + NQ-open, developed) | 0.2885 → **0.3315 (+4.30pp)** | band [+3.5, +7.0], kill < +2.5: **PASS** | 0.50pp against ≤ 1.0pp, held |
| PopQA (held-out, frozen recipe, zero adaptation) | 0.7874 → **0.8236 (+3.62pp)** | band [+1.2, +3.7], kill < +1.0: **PASS** | 0.25pp against ≤ 0.5pp, held |

On G2 the router sends 46.8% of rows to one extra generation; on PopQA, 12.8% (oracle-quality evidence keeps most answers supported). Serving cost is one conditional regeneration on routed rows; retrieval stays CPU throughout. **The architecture now has two independent population results, one of them held-out.**

### The equal-VRAM verdict

The 9B model plus corpus plus two-pass composite (0.3315) against the Gemma-12B running alone (0.1565): **+17.50pp at less VRAM** (widened from +13.20pp by the two-pass result). Spending the marginal gigabytes on a corpus and a second conditional pass beats spending them on parameters, on the population where both were measured.

### The closure ledger

Stage G swept the remaining headroom levers. Every one died measured, which is itself the map:

- **Fresh gold converts; re-presented gold re-fails.** On routed rows with gold in the recovery pack, delivery failures (gold newly delivered) convert at **0.63**; rows where pass-1 evidence already contained gold and the model still answered wrong convert at **0.18**. The same asymmetry replicated on PopQA (0.593 vs 0.343). A second read fixes delivery failures well and model failures poorly.
- **Every cheap confidence channel is dead on this model**: token logprobs AUC **0.561**; semantic answer-to-evidence similarity AUC **0.539**; letting the recovery pack arbitrate replacements nets **−0.10pp**. A universal second pass delivered its projected gross (+1.55pp) and died on its guard: 50 of 492 supported-correct rows flipped, −2.50pp against a ≤0.5pp guard. A gold-rich recovery pack supports nearly any fluent answer.
- **The encoder ceiling is closed.** A 4B bi-encoder (Qwen3-Embedding-4B) recovers **0.180** of pool-gold rows the production ordering misses; a cross-encoder (MiniLM, full query–chunk attention) recovers **0.173**: two architecturally different scorers converging on the same ~0.17–0.18 recoverable lift, and both damage the packed controls (0.900 and 0.750 retention against 1.000). The residual gold is not mis-ranked; it is semantically invisible to the query at the chunk level (enumeration prose, flattened tables, mid-article micro-facts).
- **Pack geometry is within half a point of its optimum**: the best deployable shape change (lead truncation to 200B) projects +0.45pp against a +0.75 kill line; subject-first packing is fatal (−2.17pp).

Two later censuses closed the cheapest remaining candidates. Answer stability under evidence-block permutation (the behavioral form of sample consistency) separates wrong from correct at only AUC 0.53: wrong answers are slightly less stable, but 93 of 150 confabulations never waver across three orderings. And hypothesis-guided ordering (re-ranking the recovery pool by the embedding of question + pass-1 answer, the HyDE shape) recovers 8% of the missing gold while unpacking 30% of the gold already held: a wrong hypothesis is a confident pointer toward its own confabulation, net −0.33pp.

**Substring support is a delivery detector, not a truth detector.** It routes brilliantly where its false positives are rare (unsupported answers over thin packs) and it is blind exactly where the pack is rich enough to support anything. The supported-but-wrong population (~4pp of gross headroom) stays closed until a genuinely new verification signal exists (a *trained* verifier, not another behavioral readout of the same model): four such readouts (logprobs, semantic similarity, pack arbitration, order-stability) now sit between AUC 0.47 and 0.56. The other open lines are index-time context enrichment (the one face of the ordering problem no scorer swap or query rewrite can reach) and corpus coverage expansion. Within the measured class (this generator, bge-class retrieval, 4KB packs, behavioral signals only), the two-pass architecture above is the optimum.
