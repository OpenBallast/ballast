# Model-aware selection: a negative result with a measured ceiling

*Part of the [OpenBallast working notes](../THESIS.md). The boost design is described in [methodology](methodology.md).*

## The verdict: tuned loses to generic at equal bytes

The boost design asked whether a corpus selected for a specific model beats the generic notability prefix at equal bytes. For the E2B→E4B pair, it does not:

- **Generic selection alone crosses the target**: the rank prefix closes 103% of the E2B→E4B raw gap at L2 (107 MB) and 395% at the full corpus. The "boost past the larger model" effect needs no model-awareness at all.
- **Both tuned arms lose to generic at every equal-bytes level** (profile: Δ −0.013…−0.137; delta: Δ −0.013…−0.131; McNemar p ≈ 0 throughout). The result is decisive, not marginal.
- **The oracle ceiling is nonetheless enormous**: an identity-keyed selection of the 1,885 entities the model actually misses closes 213% of the gap in 0.9 MB (three orders of magnitude less than the generic corpus needs).

## It replicates on two further arms, chosen to stress it in opposite directions

*E2B→31B* is the widest capability gap in the ladder (+0.143 on the EVAL split): generic closes 156% of it at full corpus, the tuned arm 141%, and tuned loses at every equal-bytes level (Δ −0.014…−0.131, McNemar p ≈ 0). *12B@Q4_K_M→12B@Q6_K* is the **quant-damage buyback** arm (can targeted ballast repurchase what quantization deleted?), and it fails the same way (Δ −0.009…−0.109, p ≈ 0). Three arms, spanning a 0.9-point gap and a 14-point one, all decisively negative: the result is not an artifact of the pair it was discovered on.

The gap the buyback arm is trying to close is tiny: 12B loses only 0.9 points to Q4_K_M ([quantization results](results-quantization.md)), while the generic corpus adds ~20 points on top, so generic ballast overshoots the quantization damage by **~24×** (gap_closed 23.7 at full corpus). For a model that rides its quant cleanly, worrying about recovering quantization loss is the wrong frame entirely. The corpus dwarfs it.

## The failure mechanism

The failure is visible in the coverage column: at L0 the tuned selection grounds **1.5%** of probe subjects against the generic prefix's **20.4%**. Every ingredient of tuned selection works in isolation: competence models pass their gates on every cell (AUC 0.66–0.81; see [equal-bytes results](results-equal-bytes.md)), and miss-sets are genuinely model-specific (below); but the selection objective is *expected query mass × predicted ignorance*, and corpus-native features estimate only the second factor. Predicted ignorance concentrates in the deep tail, where per-entity query mass is smallest; buying it evicts head entities that the query distribution actually touches. The oracle succeeds precisely because identity keys encode the query distribution. Parametric knowledge tracks popularity and is therefore predictable from corpus structure, but *demand* (which facts get asked) is a property of the query distribution, not the corpus, and no function of corpus-native features recovers it. This sharpens, rather than contradicts, the selection problem Mallen et al. (2023) pose: knowing *where the model is ignorant* is not sufficient; selection needs an independent estimate of what will be asked.

Scope note: this verdict is relative to the benchmark-derived query distribution of the probe set. A query distribution concentrated in the deep tail would shift the comparison; the oracle row bounds how much.

## Miss-set overlap: knowledge holes are lineage-specific

Cohen's κ between miss indicators (beyond accuracy-marginal chance), 41,381 shared probes:

- **Cross-family κ 0.27–0.55** vs within-family 0.67–0.90. On 39% of probes, Qwen3.5-0.8B misses what Gemma-4-12B knows; the converse set is 2.5%. Different training mixtures leave substantially different holes.
- **Weight quantization preserves the miss-set**: same-model pairs at different quants agree at κ 0.84–0.90; quantization mostly shrinks knowledge along the same frontier rather than moving it.
- **Except Gemma-4-E4B at nf4**: κ 0.54 against its own bf16; nf4 does not merely shrink E4B's knowledge, it *scrambles* it. This is a third independent signature of the [quant cliff](results-quantization.md), after the raw floor and the grounded ceiling.

Together these two results form a consistent picture: holes are model-specific and predictable, yet a single generic corpus serves every model measured here better than corpora tuned to each, because the generic ranking already approximates the one quantity tuning cannot observe.
