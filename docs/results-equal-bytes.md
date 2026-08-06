# Equal-bytes results: ladders, crossings, rate–distortion

*Part of the [OpenBallast working notes](../THESIS.md). Protocol: [methodology](methodology.md). All numbers measured under logprob choice scoring with abstention 0.5 unless stated.*

## The ladder, raw vs ballasted (bf16, 50,147 probes)

| model | raw accuracy | + full ballast (1.51 GB) | hallucination raw → ballasted |
|---|---|---|---|
| Gemma-4-E2B | 0.608 | **0.868** | 0.242 → **0.074** |
| Gemma-4-E4B | 0.662 | **0.910** | 0.197 → 0.042 |
| Gemma-4-12B | 0.683 | **0.910** | 0.207 → 0.049 |

Grounded ceilings compress (E4B ≈ 12B once ballasted); raw floors spread. Generations differ in what they know far more than in what they can read.

## Second family: Qwen3.5 (bf16, same probes)

| model | raw accuracy | + full ballast (1.51 GB) | hallucination raw → ballasted |
|---|---|---|---|
| Qwen3.5-0.8B | 0.324 | **0.784** | 0.601 → **0.114** |
| Qwen3.5-2B | 0.363 | 0.770 | 0.552 → 0.137 |
| Qwen3.5-4B | 0.434 | **0.831** | 0.474 → **0.095** |
| Qwen3.5-9B | 0.542 | 0.819 | 0.329 → 0.113 |

The pattern replicates across an unrelated family, and sharpens: raw floors spread 0.32–0.54 while grounded ceilings land in a 0.77–0.83 band — and the **ballasted 4B beats the ballasted 9B outright** (0.831 vs 0.819). Crossings: 0.8B + 62 MB (L1) exceeds the 4B raw (0.443 vs 0.434); 0.8B + 180 MB (L3) exceeds the 9B raw (0.552 vs 0.542) — both at ideal entity resolution (the realization band, measured on the Gemma family, is in [realized retrieval](results-retrieval.md)) and on margins of ~0.01. The parameter route to that gain is ~16 GB of bf16 weights, a ~90× byte disadvantage. The 0.8B's hallucination rate falls more than 5× (0.601 → 0.114). Families do differ in grounded ceiling (Gemma ~0.91 vs Qwen ~0.82 on the same evidence) — reading ability varies across lineages — but the within-family structure (floors spread, ceilings compress, head-heavy value-per-byte) is identical.

## Equal-bytes crossings — the headline

*(Ideal entity resolution; the realized band is in [realized retrieval](results-retrieval.md).)*

| comparison | corpus route | parameter route | byte advantage |
|---|---|---|---|
| E2B reaches E4B's raw accuracy | **+110 MB** of ballast (L2: 0.672 > 0.662) | +2.2B params ≈ 4.4 GB bf16 | **~40×** |
| E2B reaches 12B's raw accuracy | **+180 MB** of ballast (L3: 0.712 > 0.683) | +9.7B params ≈ 19.4 GB bf16 | **~100×** |
| E2B@nf4 reaches 31B@nf4's raw accuracy | **+466 MB** of ballast (L5: 0.769 > 0.732) | +28.7B params ≈ 16.3 GB nf4 | **~35×** |

The last row is the ladder top, added once the 31B cell became runnable ([quantization results](results-quantization.md)): **a 2.3B model carrying 466 MB of corpus beats a 31B model on 7.2 GB of total footprint against 17.8 GB** — the same crossing as the first two rows, now against a model 13× its size and priced at nf4 on both sides rather than against full-precision weights.

![Accuracy vs total on-disk bytes for the Gemma ladder. Each model's corpus spending rises near-vertically while the parameters-only line crawls: bytes of context buy far more accuracy than bytes of parameters.](../assets/figures/equal_bytes.png)

*Each solid line is one model spending additional bytes on corpus; the dashed line is the same budget spent on parameters instead. Corpus spend is near-vertical at this scale — the crossings in the table are visible as each small model's line climbing past the larger models' starting points.*

Pricing note: the parameter route above is charged at bf16. The [quant sweep](results-quantization.md) shows the 12B rides Q4_K_M nearly intact (raw 0.673 vs 0.683), so the cheapest honest parameter route is ~7.3 GB on disk — against which the full-corpus advantage is roughly 15× at the realized retrieval floor. The 40–100× figures hold against full-precision weights; both pricings are stated so the comparison never rests on the flattering one.

## Rate–distortion is smooth and concave

Marginal value per gigabyte falls ~14× from the head of the corpus to the tail (E2B: 0.71 accuracy-points/GB at L1 → 0.05 at L7), with the same shape across model sizes. Truncation is a real quality knob — graceful degradation, no cliff — which is what makes corpus quantization levels meaningful.

## The boundary is predictable (boost precondition)

Competence models hit **AUC 0.774 (E2B), 0.807 (E4B), 0.807 (12B)** on held-out subjects against a 0.58 gate — each model's knowledge holes are strongly predictable from corpus-native features alone. Model-aware tuned ballasts therefore have signal to exploit — yet the measured verdict ([model-aware selection](results-boost.md)) is that selection built on this signal loses to the generic prefix at equal bytes: predictable ignorance without demand-awareness is not enough.

## Live replication

The entire loop runs end-to-end on consumer hardware: a 16 GB desktop GPU holding both Gemma members, grounded by the corpus served from a free-tier edge endpoint. First 4-probe tail-PopQA spot check through open-ended generation: raw **0/4 and 0/4**; ballasted **2/4 and 3/4** (both remaining misses are grader strictness, e.g. "Eastern Orthodox Christianity" vs gold "Eastern Orthodox Church"). The raw models hallucinate *differently* (London vs Surrey for Douglas Adams's birthplace); one ~440 ms edge lookup corrects both.
