# Ballast: Quantize the Weights, Ballast the Knowledge

*OpenBallast — working notes, v0.1 (July 2026). Numbers below are measured, not projected; the experiment matrix is still running and this document will be updated as verdicts land.*

---

## 1. Thesis

**Parameters are the most expensive place to store facts.**

A language model's weights conflate two different things: a *reasoning engine* and a *knowledge inventory*. Modern small models (2–4B) have surprisingly capable engines — they read, compose, and follow instructions well — but their inventories are hard-capped by parameter count (empirically ~2 bits of recallable fact per parameter; Allen-Zhu & Li, 2024), and long-tail facts are precisely what scaling struggles to buy (Kandpal et al., 2023). Every method labs use to make small models — smaller pretrains, distillation (Hinton et al., 2015), pruning, nested elastic architectures (Devvrit et al., 2023) — compresses the engine efficiently and *loses the tail of the inventory by construction*.

The community's verdict on small models ("brain damaged," "unusable") is mostly a knowledge complaint wearing a general verdict. Our first measurement makes the split visible: **Gemma-4-E2B answers 60.8% of factual probes from its weights, and 86.8% when handed compact evidence it has never seen** — the capability to *use* facts is present; the facts are not.

So the claim: knowledge should ship as a **separate, versioned, compressed, quantizable artifact** — a *ballast* — that any model loads next to its weights:

- **~100× cheaper per fact than parameters** (measured below),
- stored in flash/RAM instead of VRAM (~1000× cheaper medium),
- updateable monthly without retraining anything,
- quantizable in nested levels exactly like weight quants (pick your L2 like you pick your Q4),
- CC0, reproducible from public dumps by anyone.

One sentence: **quantize the weights, ballast the knowledge.**

### 1.1 What's borrowed, what's new

The core mechanism — augmenting a parametric model with an external datastore at
inference time — is the semiparametric/retrieval-augmented line: kNN-LM
(Khandelwal et al., 2020), REALM (Guu et al., 2020), RAG (Lewis et al., 2020),
RETRO (Borgeaud et al., 2022), Atlas (Izacard et al., 2022). RETRO and Atlas in
particular established that a small model plus retrieval can match a much larger
one. Mallen et al. (2023) established the specific fact this project leans
hardest on: parametric memory tracks entity *popularity*, retrieval helps
exactly where popularity runs out — and can even *hurt* at the head, which is
why selection (not just retrieval) is the interesting problem.

The packaging idea follows two existing format precedents. Weights have **GGUF**
(llama.cpp; Gerganov et al.): one file, quantization levels, works everywhere.
Offline reading has **ZIM** (openZIM/Kiwix), which spent two decades proving a
reference corpus can ship as one portable, versioned, offline artifact.
Knowledge for models has no equivalent — every retrieval corpus is a bespoke
pipeline and somebody's Docker Compose file. Ballast is an attempt at that
missing third format, and takes the artifact model directly from those two.

The compression and access layer is borrowed whole from the data-engineering
stack: **Apache Parquet** columnar storage under **zstd** compression (Collet &
Kucherawy, RFC 8878), **Hive-style partitioning** (Thusoo et al., 2009) — which
is what makes levels real: buckets are directory prefixes any engine prunes
without reading, so truncation happens at download time — and **DuckDB**
(Raasveldt & Mühleisen, 2019) as the in-process query engine, so the artifact is
readable on a laptop or an edge worker with zero servers. None of that is ours;
the contribution is noticing this stack is the right shape for quantizable model
knowledge.

What this project adds is artifact discipline on top of that literature: the
datastore as a *versioned, CC0, standalone release* rather than a lab-internal
index; nested rank-bucket **levels** that make knowledge/bytes a user-facing
knob analogous to weight quants; a measured **corpus-bytes vs parameter-bytes
exchange rate** on one probe set across model sizes and weight quants; and
per-model **tuned ballasts** selected by a competence model. The numbers below
measure those additions, not retrieval augmentation per se.

## 2. The artifact

**Ballast T0** is the reference artifact: cleaned Wikidata (Vrandečić & Krötzsch, 2014) triples, published as hive-partitioned Parquet. Canonical distribution: [huggingface.co/datasets/OpenBallast/ballast-t0](https://huggingface.co/datasets/OpenBallast/ballast-t0).

| property | value |
|---|---|
| entities | 25.4M (every Wikidata item with a resolvable label, minus Wikimedia-internal pages) |
| triples | 197.4M statements (external-identifier, media, and URL datatypes excluded at publish) |
| properties | 13,704 (labels shipped with every level) |
| published size | **1.51 GB** (zstd parquet; 8.67 GB as raw TSV) |
| license | CC0 (Wikidata) |
| layout | `entities/…/rank_bucket=k/`, `triples/…/rank_bucket=k/`, `properties.parquet`, `manifest.json` |

### Corpus quantization

Entities are ranked by a composite of notability signals (log sitelinks + log claim count) and assigned to **8 nested rank buckets** at fractions 0.5%, 1%, 2%, 4%, 8%, 16%, 32%, 100%. A *level* Lk is the union of buckets 0..k — a byte-budget prefix of the corpus:

| level | cumulative size | contains |
|---|---|---|
| L0 | 36 MB | top 0.5% of entities |
| L1 | 62 MB | top 1% |
| L2 | 107 MB | top 2% |
| L3 | 179 MB | top 4% |
| L4 | 288 MB | top 8% |
| L5 | 466 MB | top 16% |
| L6 | 756 MB | top 32% |
| L7 | 1,507 MB | everything |

Truncation is not a filter — deeper buckets are simply never downloaded. It degrades evidence honestly in two ways: a subject outside the cutoff has no evidence at all, and a triple whose *object* entity falls outside the cutoff is dropped rather than rendered as a bare Q-id. Both behaviors are preserved end-to-end, including in the demo endpoint.

## 3. Methodology

### 3.1 Probes

Knowledge is measured by **logprob choice scoring**: for each question, score the length-normalized answer logprob of the gold answer against 7 type-matched distractors under the same prompt; prediction = argmax; confidence = softmax mass on the argmax. No generation, no judge model. This is the standard multiple-choice evaluation protocol from GPT-3-era benchmarking (Brown et al., 2020) as implemented in lm-evaluation-harness (Gao et al., 2023); probing knowledge via KB-triple-derived questions goes back to LAMA (Petroni et al., 2019). Abstention threshold 0.5 on candidate probability yields the triad *(correct / hallucinated / not attempted)*, adopting SimpleQA's grading taxonomy (Wei et al., 2024) — hallucination rate = wrong per attempt.

Probe set: **50,147 questions** across PopQA (Mallen et al., 2023), a self-generated Wikidata benchmark (rank-stratified, uncontaminated by construction), SimpleQA (Wei et al., 2024), Natural Questions (open; Kwiatkowski et al., 2019), and TriviaQA (Joshi et al., 2017) — the latter three entity-linked to the corpus by normalized label/alias match. Full corpus grounds 90.5% of subjects; of grounded probes, 97.7% carry the gold answer inside the rendered evidence block (instrumented per probe, not assumed).

### 3.2 The two-pass composition trick

The rate–distortion curve wants accuracy at every level, but a probe's evidence at cutoff k is determined by its subject's bucket. So **two GPU passes per model** (no corpus, full corpus) determine every point *exactly*:

```
accuracy(Lk) = mean_i [ grounded_i  if bucket(subject_i) ≤ k  else ungrounded_i ]
```

The one approximation — evidence content also thins under truncation — is checked, not assumed: re-running a 1,000-probe subsample with evidence physically rendered at each cutoff agrees with the composition to ≈0.002 accuracy. This trick is why the whole experiment matrix fits on one 32 GB workstation card: every corpus level, selection scheme, and budget arm is arithmetic over two passes.

### 3.3 The matrix

Three ways to spend bytes on knowledge — parameter count, parameter precision, corpus — measured on one probe set:

- **Size ladders at bf16**: Gemma-4 E2B / E4B / 12B; Qwen3.5 0.8B / 2B / 4B / 9B.
- **Gemma span at nf4** including the 31B (the only quant where it fits 32 GB), kept on its own axis so nf4 damage is never silently compared to bf16.
- **Quant sweep** on pivots (E4B, 12B): bf16 / fp8 / nf4 (Dettmers et al., 2023) / Q6_K / Q4_K_M (llama.cpp GGUF K-quants; Gerganov et al.) — GGUF K-quants scored via dequantization so the quantization error is preserved verbatim while deployment bytes are charged at the .gguf size.

### 3.4 Model-aware ballast (the boost design)

Beyond the generic level prefix, a **competence model** per (model, quant) cell predicts P(model already knows | entity) from corpus-native features (rank, sitelinks, claim count). That such prediction should work at all is prior art: parametric knowledge tracks entity popularity (Mallen et al., 2023; Kandpal et al., 2023), and models' self-knowledge is itself well-calibrated (Kadavath et al., 2022) — we use external corpus features rather than model introspection so selection can run without GPU passes. The competence model is fit on half the probes (split by subject hash), evaluated on the other half, with a hard gate: if the model can't rank its own misses better than AUC 0.58, selection would be noise and the arm is abandoned. Four selection arms per model pair: **generic** (rank prefix), **profile** (weight by predicted ignorance), **delta** (weight by what a reference model knows and this one doesn't), **oracle** (identity-keyed ceiling, fit-side only). Selected corpora are written as *real artifacts* with label closure (every referenced object keeps its label row), so composition stays exact.

### 3.5 Beyond recall

A second probe family tests whether ballast reduces hallucination when copy-extraction cannot work: 2-hop composition chains whose answer appears only via a join (with an adversarial subset where the competing join path's answer is also present in evidence — distractor-in-context stress in the spirit of Shi et al., 2023), unanswerable/false-premise probes where every candidate is false (fabrication = any confident pick; the unanswerable-question design follows SQuAD 2.0, Rajpurkar et al., 2018, and false-premise framing follows FreshQA, Vu et al., 2023), 2WikiMultiHopQA (Ho et al., 2020), and TruthfulQA MC1 (Lin et al., 2022) as a falsification control — misconception-driven questions that grounding should *not* improve; if it does, gains are prompt artifact. Verdicts pending (in the running matrix).

## 4. Numbers so far

### 4.1 The ladder, raw vs ballasted (bf16, 50,147 probes, abstention 0.5)

| model | raw accuracy | + full ballast (1.51 GB) | hallucination raw → ballasted |
|---|---|---|---|
| Gemma-4-E2B | 0.608 | **0.868** | 0.242 → **0.074** |
| Gemma-4-E4B | 0.662 | **0.910** | 0.197 → 0.042 |
| Gemma-4-12B | 0.683 | **0.910** | 0.207 → 0.049 |

Grounded ceilings compress (E4B ≈ 12B once ballasted); raw floors spread. Generations differ in what they know far more than in what they can read.

### 4.1b Second family: Qwen3.5 (bf16, same probes)

| model | raw accuracy | + full ballast (1.51 GB) | hallucination raw → ballasted |
|---|---|---|---|
| Qwen3.5-0.8B | 0.324 | **0.784** | 0.601 → **0.114** |
| Qwen3.5-2B | 0.363 | 0.770 | 0.552 → 0.137 |
| Qwen3.5-4B | 0.434 | **0.831** | 0.474 → **0.095** |
| Qwen3.5-9B | 0.542 | 0.819 | 0.329 → 0.113 |

The pattern replicates across an unrelated family, and sharpens: raw floors
spread 0.32–0.54 while grounded ceilings land in a 0.77–0.83 band — and the
**ballasted 4B beats the ballasted 9B outright** (0.831 vs 0.819). Crossings:
0.8B + 62 MB (L1) exceeds the 4B raw (0.443 vs 0.434); 0.8B + 180 MB (L3)
exceeds the 9B raw (0.552 vs 0.542) — the parameter route to that gain is
~16 GB of weights, a ~90× byte disadvantage. The 0.8B's hallucination rate
falls more than 5× (0.601 → 0.114). Families do differ in grounded ceiling
(Gemma ~0.91 vs Qwen ~0.82 on the same evidence) — reading ability varies
across lineages — but the within-family structure (floors spread, ceilings
compress, head-heavy value-per-byte) is identical.

### 4.2 Equal-bytes crossings — the headline

| comparison | corpus route | parameter route | byte advantage |
|---|---|---|---|
| E2B reaches E4B's raw accuracy | **+110 MB** of ballast (L2: 0.672 > 0.662) | +2.2B params ≈ 4.4 GB bf16 | **~40×** |
| E2B reaches 12B's raw accuracy | **+180 MB** of ballast (L3: 0.712 > 0.683) | +9.7B params ≈ 19.4 GB bf16 | **~100×** |

### 4.3 Rate–distortion is smooth and concave

Marginal value per gigabyte falls ~14× from the head of the corpus to the tail (E2B: 0.71 accuracy-points/GB at L1 → 0.05 at L7), with the same shape across model sizes. Truncation is a real quality knob — graceful degradation, no cliff — which is what makes corpus quantization levels meaningful.

### 4.4 The boundary is predictable (boost precondition)

Competence models hit **AUC 0.774 (E2B), 0.807 (E4B), 0.807 (12B)** on held-out subjects against a 0.58 gate — each model's knowledge holes are strongly predictable from corpus-native features alone. Model-aware tuned ballasts therefore have signal to exploit; gap-closed verdicts are in the running matrix.

### 4.5 Live replication

The entire loop runs end-to-end on consumer hardware: a 16 GB desktop GPU holding both Gemma members, grounded by the corpus served from a free-tier edge endpoint. First 4-probe tail-PopQA spot check through open-ended generation: raw **0/4 and 0/4**; ballasted **2/4 and 3/4** (both remaining misses are grader strictness, e.g. "Eastern Orthodox Christianity" vs gold "Eastern Orthodox Church"). The raw models hallucinate *differently* (London vs Surrey for Douglas Adams's birthplace); one ~440 ms edge lookup corrects both.

### 4.6 Weight quantization interacts — the nf4 axis

Same ladder at nf4 (own axis, never silently compared to bf16):

| model @ nf4 | raw (Δ vs bf16) | + full ballast (Δ vs bf16) |
|---|---|---|
| Gemma-4-E2B | 0.587 (−0.021) | 0.842 (−0.026) |
| Gemma-4-E4B | 0.485 (**−0.177**) | 0.650 (**−0.260**) |
| Gemma-4-12B | 0.673 (−0.010) | 0.903 (−0.007) |

The full pivot sweep (E4B, 12B at bf16 / fp8 / nf4) shows it as a dose-response:

| quant | E4B raw → ballasted | 12B raw → ballasted |
|---|---|---|
| bf16 | 0.662 → 0.910 | 0.683 → 0.910 |
| fp8 | 0.632 → 0.841 | 0.677 → 0.906 |
| nf4 | 0.485 → **0.650** | 0.673 → 0.903 |

The 12B's grounded ceiling is untouched across the entire sweep (0.910 → 0.903);
E4B's collapses monotonically with quantization depth (0.910 → 0.841 → 0.650).
(GGUF K-quant cells pending.)

Two lessons. First, nf4 damage is architecture-dependent, not size-monotonic:
E2B and 12B ride it nearly free while E4B falls off a cliff. Second — and the
important one for this thesis — when quantization damages the *engine*, ballast
cannot buy it back: E4B@nf4's grounded ceiling collapses along with its raw
floor, i.e. nf4 broke its ability to read evidence, and no amount of corpus
fixes a broken reader. The community's "Q4 made it unusable" experience and the
"Q4 is free" experience are both real; they happen on different models, and the
grounded ceiling is the diagnostic that tells them apart. Corollary: at nf4 the
*smaller* Gemma is strictly better once ballasted (0.842 vs 0.650), and
E2B@nf4 + full ballast — under 3 GB of total footprint — beats the raw
12B bf16 (0.842 vs 0.683 on ~24 GB).

### 4.7 Realized retrieval — de-oracling the headline

The matrix assumes perfect entity resolution. We measured what a real,
non-generative linker (capitalized-span mining + normalized name index)
actually realizes on the same 50,147 probes, and priced its failure modes:

- A linked probe resolving to the **right** entity realizes that probe's
  grounded outcome. A **miss** (nothing linked) falls back to ungrounded —
  free. A **wrong link** attaches a homonym's facts, and a dedicated GPU pass
  (854 wrong-linked probes, E2B) shows this is *actively harmful*: accuracy
  0.423 → 0.159, hallucination-per-attempt 0.345 → 0.536. Wrong evidence costs
  roughly what right evidence gains.
- Composing per-probe over seven linker variants (E2B bf16): the naive precise
  linker realizes **26.7%** of the oracle gain; adding context disambiguation
  (question-token overlap against candidate evidence, blended with notability)
  lifts it to **34.1%**. Every recall-raising variant tested — lowercase
  fallback mining, FTS5 fuzzy matching, gated combinations — *reduced* realized
  gain (to 15–20%), because they convert free misses into costly wrong links.
- The thesis's erasure-vs-substitution asymmetry (§3.3) therefore applies to
  the retriever itself: **a linker that shrugs beats a linker that guesses.**
  The shipped CLI and demo endpoint run the 34% configuration; recall additions
  are opt-in.

**What the floor does to the headline crossings — stated plainly.** At 34%
realization, E2B + the full 1.51 GB ballast lands at ≈0.70 end-to-end against
the 12B's raw 0.683: the full-corpus crossing survives, but barely — within
noise. The sharper claim, **"E2B + 180 MB beats the 12B raw," does not hold at
today's retrieval floor** (≈0.64 vs 0.683); it is an oracle-ceiling statement.
Two qualifications keep this honest in both directions. First, the probe
setting — mining entity mentions out of raw natural-language trivia questions —
is close to worst-case retrieval; when the *model* supplies the mention (the
MCP tool-calling path: `resolve("Douglas Adams")`), the name index resolves
clean mentions at ~88%, putting agent-mode deployments near the ceiling, not
the floor. Second, the floor is one afternoon old and moved 7 points in that
afternoon.

This turns the former "oracle retrieval" caveat into a measured band: the
corpus's value is the ceiling, a trivial linker holds a third of it, and the
remaining gap is an engineering ladder (typed disambiguation, retrieval-quality
levels R0–R7) with a per-rung measurement already in place — not an assumption.

## 5. Honest caveats

- **Retrieval realization is a band, not a point**: matrix numbers are the
  perfect-resolution ceiling; the measured trivial linker realizes ~34% of the
  grounding gain end-to-end (§4.7). Retrieval-side improvements move within
  that band without touching the artifact.
- **Coverage cap**: the corpus grounds 90.5% of probe subjects; the remainder bounds grounded accuracy.
- **Abstention-thresholded metrics**: raw forced-choice gaps are smaller than triad numbers.
- **Contamination**: models have seen Wikipedia/Wikidata in pretraining. The self-generated benchmark is uncontaminated by construction; external sets serve as robustness checks; and the interesting quantity (grounded − ungrounded delta) is contamination-*conservative* — pretraining exposure inflates the raw floor, not the gain.
- **Generative grading** (demo bench) is weaker than logprob scoring — demo readout, not measurement.
- Facts absent from Wikidata are absent from the ballast; T0 measures the triple-shaped slice of knowledge.

## 6. What's running now / next

- Boost verdicts: gap-closed curves for E2B→E4B, E2B→31B, and the **quant-damage buyback** arm (12B Q4_K_M → Q6_K: what fraction of quantization's knowledge loss does targeted ballast recover, per MB).
- Hallucination-beyond-recall verdicts (composition, fabrication, control).
- Gemma-4-31B; cross-family miss-set overlap — if families miss *different* facts, a shared ballast is worth more than either family's tuning. (Qwen3.5 ladder: done, §4.1b.)
- Legacy revival cells (Mistral-7B-class): grounding as a generation equalizer.
- Tooling for third parties to build and tune their own ballasts (to be open-sourced separately).

## 7. Where things live

| surface | role |
|---|---|
| [huggingface.co/OpenBallast](https://huggingface.co/OpenBallast) | **canonical artifacts** — the T0 corpus and eval sets, versioned |
| [github.com/OpenBallast/ballast-cli](https://github.com/OpenBallast/ballast-cli) | **local tooling** — `ballast pull/serve`: OpenAI grounding proxy + MCP server over the corpus, offline |
| [mcp.openballast.org](https://mcp.openballast.org) | live **demo endpoint** (MCP + HTTP) — levels L0–L5 served from a $0/month free-tier stack |
| [github.com/OpenBallast](https://github.com/OpenBallast) | docs, spec, this document |
| [openballast.org](https://openballast.org) | landing |

Data: Wikidata contributors, CC0.

## 8. References

- Allen-Zhu, Z. & Li, Y. (2024). *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.* ICLR 2025. [arXiv:2404.05405](https://arxiv.org/abs/2404.05405)
- Apache Parquet. *Columnar storage format.* [parquet.apache.org](https://parquet.apache.org)
- Borgeaud, S. et al. (2022). *Improving Language Models by Retrieving from Trillions of Tokens* (RETRO). [arXiv:2112.04426](https://arxiv.org/abs/2112.04426)
- Collet, Y. & Kucherawy, M. (2021). *Zstandard Compression and the application/zstd Media Type.* [RFC 8878](https://www.rfc-editor.org/rfc/rfc8878)
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners* (GPT-3). [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs* (NF4). [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Devvrit et al. (2023). *MatFormer: Nested Transformer for Elastic Inference.* [arXiv:2310.07707](https://arxiv.org/abs/2310.07707)
- Gao, L. et al. (2023). *A framework for few-shot language model evaluation* (lm-evaluation-harness). [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Gerganov, G. et al. *llama.cpp / GGUF quantization formats.* [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- Guu, K. et al. (2020). *REALM: Retrieval-Augmented Language Model Pre-Training.* [arXiv:2002.08909](https://arxiv.org/abs/2002.08909)
- Hinton, G., Vinyals, O. & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- Ho, X. et al. (2020). *Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps* (2WikiMultiHopQA). [arXiv:2011.01060](https://arxiv.org/abs/2011.01060)
- Izacard, G. et al. (2022). *Atlas: Few-shot Learning with Retrieval Augmented Language Models.* [arXiv:2208.03299](https://arxiv.org/abs/2208.03299)
- Joshi, M. et al. (2017). *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.* [arXiv:1705.03551](https://arxiv.org/abs/1705.03551)
- Kadavath, S. et al. (2022). *Language Models (Mostly) Know What They Know.* [arXiv:2207.05221](https://arxiv.org/abs/2207.05221)
- Kandpal, N. et al. (2023). *Large Language Models Struggle to Learn Long-Tail Knowledge.* [arXiv:2211.08411](https://arxiv.org/abs/2211.08411)
- Khandelwal, U. et al. (2020). *Generalization through Memorization: Nearest Neighbor Language Models* (kNN-LM). [arXiv:1911.00172](https://arxiv.org/abs/1911.00172)
- Kwiatkowski, T. et al. (2019). *Natural Questions: A Benchmark for Question Answering Research.* TACL 7. [doi:10.1162/tacl_a_00276](https://doi.org/10.1162/tacl_a_00276)
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (RAG). [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Lin, S., Hilton, J. & Evans, O. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* [arXiv:2109.07958](https://arxiv.org/abs/2109.07958)
- Mallen, A. et al. (2023). *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories* (PopQA). [arXiv:2212.10511](https://arxiv.org/abs/2212.10511)
- openZIM contributors. *The ZIM file format* (Kiwix offline content archives). [openzim.org](https://openzim.org)
- Petroni, F. et al. (2019). *Language Models as Knowledge Bases?* (LAMA). [arXiv:1909.01066](https://arxiv.org/abs/1909.01066)
- Raasveldt, M. & Mühleisen, H. (2019). *DuckDB: an Embeddable Analytical Database.* SIGMOD 2019. [doi:10.1145/3299869.3320212](https://doi.org/10.1145/3299869.3320212)
- Rajpurkar, P., Jia, R. & Liang, P. (2018). *Know What You Don't Know: Unanswerable Questions for SQuAD* (SQuAD 2.0). [arXiv:1806.03822](https://arxiv.org/abs/1806.03822)
- Shi, F. et al. (2023). *Large Language Models Can Be Easily Distracted by Irrelevant Context.* [arXiv:2302.00093](https://arxiv.org/abs/2302.00093)
- Thusoo, A. et al. (2009). *Hive: A Warehousing Solution Over a Map-Reduce Framework.* VLDB 2(2). [doi:10.14778/1687553.1687609](https://doi.org/10.14778/1687553.1687609)
- Vrandečić, D. & Krötzsch, M. (2014). *Wikidata: A Free Collaborative Knowledgebase.* CACM 57(10). [doi:10.1145/2629489](https://doi.org/10.1145/2629489)
- Vu, T. et al. (2023). *FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation* (FreshQA). [arXiv:2310.03214](https://arxiv.org/abs/2310.03214)
- Wei, J. et al. (2024). *Measuring Short-Form Factuality in Large Language Models* (SimpleQA). [arXiv:2411.04368](https://arxiv.org/abs/2411.04368)
