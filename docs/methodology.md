# Methodology

*Part of the [OpenBallast working notes](../THESIS.md). Full citations are in the overview's [references](../THESIS.md#references).*

## Probes

Knowledge is measured by **logprob choice scoring**: for each question, score the length-normalized answer logprob of the gold answer against 7 type-matched distractors under the same prompt; prediction = argmax; confidence = softmax mass on the argmax. No generation, no judge model. This is the standard multiple-choice evaluation protocol from GPT-3-era benchmarking (Brown et al., 2020) as implemented in lm-evaluation-harness (Gao et al., 2023); probing knowledge via KB-triple-derived questions goes back to LAMA (Petroni et al., 2019). Abstention threshold 0.5 on candidate probability yields the triad *(correct / hallucinated / not attempted)*, adopting SimpleQA's grading taxonomy (Wei et al., 2024): hallucination rate = wrong per attempt.

Probe set: **50,147 questions** across PopQA (Mallen et al., 2023), a self-generated Wikidata benchmark (rank-stratified, uncontaminated by construction), SimpleQA (Wei et al., 2024), Natural Questions (open; Kwiatkowski et al., 2019), and TriviaQA (Joshi et al., 2017), with the latter three entity-linked to the corpus by normalized label/alias match. Full corpus grounds 90.5% of subjects; of grounded probes, 97.7% carry the gold answer inside the rendered evidence block (instrumented per probe, not assumed).

## The two-pass composition trick

The rate–distortion curve wants accuracy at every level, but a probe's evidence at cutoff k is determined by its subject's bucket. So **two GPU passes per model** (no corpus, full corpus) determine every point *exactly*:

```
accuracy(Lk) = mean_i [ grounded_i  if bucket(subject_i) ≤ k  else ungrounded_i ]
```

The one approximation (evidence content also thins under truncation) is checked, not assumed: re-running a 1,000-probe subsample with evidence physically rendered at each cutoff agrees with the composition to ≈0.002 accuracy. This trick is why the whole experiment matrix fits on one 32 GB workstation card: every corpus level, selection scheme, and budget arm is arithmetic over two passes.

## The matrix

Three ways to spend bytes on knowledge (parameter count, parameter precision, corpus) measured on one probe set:

- **Size ladders at bf16**: Gemma-4 E2B / E4B / 12B; Qwen3.5 0.8B / 2B / 4B / 9B.
- **Gemma span at nf4** including the 31B (the only quant where it fits 32 GB), kept on its own axis so nf4 damage is never silently compared to bf16.
- **Quant sweep** on pivots (Gemma-4 E4B, 12B; Qwen3.5 4B, 9B): bf16 / fp8 / nf4 (Dettmers et al., 2023) / Q6_K / Q4_K_M (llama.cpp GGUF K-quants; Gerganov et al.). GGUF K-quants are scored via dequantization so the quantization error is preserved verbatim while deployment bytes are charged at the .gguf size. No public *base* K-quants exist for either family, so all GGUFs are self-converted from the base checkpoints; the Qwen3.5 ones additionally need a hand-written reverse of llama.cpp's conversion (transformers has no `qwen35` GGUF architecture), verified by tensor-wise roundtrip before use.

## Model-aware ballast (the boost design)

Beyond the generic level prefix, a **competence model** per (model, quant) cell predicts P(model already knows | entity) from corpus-native features (rank, sitelinks, claim count). That such prediction should work at all is prior art: parametric knowledge tracks entity popularity (Mallen et al., 2023; Kandpal et al., 2023), and models' self-knowledge is itself well-calibrated (Kadavath et al., 2022). We use external corpus features rather than model introspection so selection can run without GPU passes. The competence model is fit on half the probes (split by subject hash), evaluated on the other half, with a hard gate: if the model can't rank its own misses better than AUC 0.58, selection would be noise and the arm is abandoned. Four selection arms per model pair: **generic** (rank prefix), **profile** (weight by predicted ignorance), **delta** (weight by what a reference model knows and this one doesn't), **oracle** (identity-keyed ceiling, fit-side only). Selected corpora are written as *real artifacts* with label closure (every referenced object keeps its label row), so composition stays exact. Verdicts: [model-aware selection](results-boost.md).

## Beyond recall

A second probe family tests whether ballast reduces hallucination when copy-extraction cannot work: 2-hop composition chains whose answer appears only via a join (with an adversarial subset where the competing join path's answer is also present in evidence, distractor-in-context stress in the spirit of Shi et al., 2023), unanswerable/false-premise probes where every candidate is false (fabrication = any confident pick; the unanswerable-question design follows SQuAD 2.0, Rajpurkar et al., 2018, and false-premise framing follows FreshQA, Vu et al., 2023), 2WikiMultiHopQA (Ho et al., 2020), and TruthfulQA MC1 (Lin et al., 2022) as a falsification control (misconception-driven questions that grounding should *not* improve); if it does, gains are prompt artifact. Verdicts: [hallucination results](results-hallucination.md).

## Registered bands

Every experimental cell in the retrieval research line (and, increasingly, the matrix) runs under a pre-registration discipline:

- **Bands and kill lines are fixed before the run.** Each screen registers a predicted success band, a kill threshold, and any guard (e.g. "covered-population regression ≤ 1pp") before any result is seen. A result below the kill line kills the mechanism as registered: no post-hoc grid refinement, no rescue tuning.
- **Components are measured on the deployment population.** Bands built from component products (capture × survival × ceiling) proved unreliable whenever a component was measured on a different distribution than the one being deployed to; the standing rule is that every component of a projection must be measured on the target population itself before a band is registered. Adopting this rule is what turned a run of structural band misses into consecutive in-band predictions.
- **Results are reported however they land.** Kills are first-class results with mechanical autopsies, and every dated correction note stays in the public record.
