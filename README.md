# ⚓ OpenBallast

**Quantize the weights. Ballast the knowledge.**

## What this is

A language model is a reasoning engine fused to an encyclopedia. Most of what a
bigger model buys you is the encyclopedia, and parameters are an expensive place
to store facts.

Ballast separates the two. The knowledge ships as a versioned, CC0 file that
sits next to the model. It comes in nested sizes (L0 = 36 MB up to L7 = 1.5 GB
of Parquet; the CLI's prebuilt SQLite levels run somewhat larger), so you pick a
knowledge level the same way you pick a quant. The model reads from it at answer
time.

Measured result: a 2B model plus a 470 MB ballast file exceeds a 12B model's
factual accuracy — end to end, with a real entity linker doing the lookups.
Under perfect retrieval it takes 180 MB. Either way, the same gain through
parameters costs ~19 GB of weights, and ballast bytes sit in flash/RAM rather
than VRAM.

## Pain points

**"Small models are brain-damaged."** Gemma-4-E2B answers 60.8% of factual
probes from its weights, and 86.8% when handed a short evidence snippet. It can
use facts it's given; it doesn't have room to memorize them. The problem is
missing knowledge, not missing capability.

**"It makes things up."** Same model, same probes: hallucination drops from 0.24
to 0.07 with ballast attached. In the live demo, two raw models gave different
wrong birthplaces for Douglas Adams (London, Surrey); one corpus lookup
corrected both. It holds where copying can't explain it — on two-hop questions
whose answer only exists by joining two evidence lines, hallucination falls
3–20×.

The boundary is measured too, and it cuts the other way: on questions with **no**
true answer (false premises, properties nobody recorded), evidence suppresses
correct abstention and fabrication *rises* — 0.18–0.27 ungrounded to 0.31–0.44
at the full corpus, in every model and quant we tested. Grounding helps when the
question has an answer. It does not teach a model to say "I don't know."

**"My GPU has 8/12/16 GB."** VRAM spent memorizing the long tail of Wikipedia is
VRAM unavailable for context or a better engine. Ballast moves those facts to
disk. A 16 GB desktop card runs the full demo: two models plus corpus lookups.

**"The model's knowledge is stale."** Weights update at training time only. A
ballast rebuilds monthly from public dumps, without retraining or re-downloading
the model.

**"We can't send data to anyone's API."** The common hallucination fixes — web
search, hosted RAG, embedding APIs — need a network. A ballast is a static
file: copy it across the air gap once and the knowledge layer runs offline.
Relevant for defense, healthcare, ships, factory floors, and data-residency
regimes. Because it's a single versioned artifact, you can also audit exactly
what the model has access to.

**"Serving costs."** If a 2B engine with a knowledge sidecar matches a 12B on
factual work, that's ~6× fewer parameters to run per token — partly offset by
the evidence block, which adds a few hundred prefill tokens per grounded
question. The sidecar itself serves from commodity storage; the demo endpoint
runs on a $0/month free tier. Knowledge becomes one shared copy per cluster
instead of a per-GPU parameter cost.

## Who this is for

- **Local runners** — pair a small model with the level that fits your disk.
  L3 (179 MB) already lifts E2B past E4B's raw accuracy with real linking (L2,
  107 MB, under perfect retrieval).
- **Edge & embedded** — engine in silicon, facts in flash, knowledge updated
  over the air without touching weights.
- **Air-gapped operators** — the knowledge layer crosses the gap as one
  auditable file and runs offline.
- **Fleet operators** — serve smaller engines, mount one shared artifact per
  cluster, refresh it monthly.
- **Old-model owners** — once ballasted, E4B ≈ 12B. Models age faster in what
  they know than in what they can read, so older models may come back further
  than expected.
- **Researchers** — 50k linked probes, exact composition methodology, caveats
  documented. Start with [THESIS.md](THESIS.md).

## "Isn't this just RAG?"

Mechanically, yes: retrieval at answer time. The differences:

- **Nested levels** (L0–L7) make knowledge-per-byte a tunable knob like
  Q4/Q6/Q8. The value-per-byte curve is measured, smooth, and concave — the
  first 100 MB is worth ~14× the last.
- **One canonical CC0 artifact** anyone can pin, cite, diff, and rebuild from
  public dumps, instead of a private vector DB and embedding pipeline.
- **The measured quantity is the exchange rate** between corpus bytes and
  parameter bytes, on the same probes, across model sizes and weight quants.
  That rate (~40–100×) is the finding, not "retrieval helps."
- **One artifact is the right shape, and we tried to disprove it.** Models from
  different lineages miss *different* facts (agreement between miss-sets: κ
  0.27–0.55 across families, 0.67–0.90 within one), so per-model corpora looked
  promising. They lose. Ballasts selected for a specific model — using
  competence models that predict its gaps at AUC ~0.8 — were beaten by the plain
  generic prefix at every equal-byte size. Corpus features can tell you what a
  model doesn't know, but not what anyone will ask, and selection needs both.
  Details, including the oracle ceiling that says the idea isn't dead:
  [THESIS.md §4.8](THESIS.md).

## Numbers

Measured on 50,147 factual probes (PopQA, SimpleQA, Natural Questions, TriviaQA,
plus an uncontaminated self-generated set). Methodology and full tables in
[THESIS.md](THESIS.md).

| model (bf16) | raw accuracy | + full ballast (1.51 GB) | hallucination raw → ballasted |
|---|---|---|---|
| Gemma-4-E2B | 0.608 | **0.868** | 0.242 → **0.074** |
| Gemma-4-E4B | 0.662 | **0.910** | 0.197 → 0.042 |
| Gemma-4-12B | 0.683 | **0.910** | 0.207 → 0.049 |

| to match… | ballast route | parameter route | advantage |
|---|---|---|---|
| E4B's raw accuracy (from E2B) | **+110 MB** | +4.4 GB of weights | ~40× |
| 12B's raw accuracy (from E2B) | **+180 MB** | +19.4 GB of weights | ~100× |

(Crossing sizes here assume perfect entity resolution — what that costs in
practice is measured two sections down.)

It replicates on a second, unrelated family:

| model (bf16) | raw accuracy | + full ballast (1.51 GB) | hallucination raw → ballasted |
|---|---|---|---|
| Qwen3.5-0.8B | 0.324 | **0.784** | 0.601 → **0.114** |
| Qwen3.5-2B | 0.363 | 0.770 | 0.552 → 0.137 |
| Qwen3.5-4B | 0.434 | **0.831** | 0.474 → **0.095** |
| Qwen3.5-9B | 0.542 | 0.819 | 0.329 → 0.113 |

Raw accuracy spreads 0.32–0.54; ballasted lands in a 0.77–0.83 band. The
**ballasted 4B beats the ballasted 9B outright**, and the 0.8B plus 180 MB
passes the 9B's raw accuracy — a gain that costs ~16 GB of weights to buy with
parameters. The 0.8B's hallucination rate falls more than 5×.

### What real retrieval costs you

The tables above assume perfect entity resolution — every question finds the
right entity. So we measured a real one: a non-generative linker (span mining +
name index, no embeddings) on the same 50,147 probes. It realizes **63–66%** of
the oracle gain (52% before disambiguation and fuzzy matching); wrong links turn
out to be roughly harmless, because models ignore evidence that doesn't fit the
question. At that floor the full-corpus crossing holds comfortably, and the
sharp "+180 MB beats a 12B" claim becomes +470 MB. Both numbers are in
[THESIS.md §4.7](THESIS.md); caveats in [§5](THESIS.md).

### If you run quantized weights, read this one

Weight quantization interacts with ballast in a way that isn't size-monotonic:

| model @ nf4 | raw (Δ vs bf16) | + full ballast (Δ vs bf16) |
|---|---|---|
| Gemma-4-E2B | 0.587 (−0.021) | 0.842 (−0.026) |
| Gemma-4-E4B | 0.485 (**−0.177**) | 0.650 (**−0.260**) |
| Gemma-4-12B | 0.673 (−0.010) | 0.903 (−0.007) |

"Q4 ruined this model" and "Q4 is free" are both true — of different models. The
12B rides the whole bf16 → fp8 → nf4 sweep almost untouched (grounded 0.910 →
0.903); the E4B falls off a cliff (0.910 → 0.841 → 0.650). And when
quantization damages the engine, no amount of corpus buys it back: the E4B's
grounded ceiling collapses with its raw floor, meaning nf4 broke its ability to
*read* evidence.

The useful part: the grounded ceiling is a diagnostic that tells the two cases
apart, and it inverts the usual ordering — at nf4 the smaller Gemma is strictly
better once ballasted (0.842 vs 0.650), and E2B@nf4 plus the full corpus, under
3 GB total, beats the raw 12B at bf16 on ~24 GB.

## Get it

| what | where |
|---|---|
| **`ballast` CLI** (pull the corpus, ground Ollama / any MCP client) | [github.com/OpenBallast/ballast-cli](https://github.com/OpenBallast/ballast-cli) |
| **T0 corpus** (25.4M entities, 197M triples, levels L0–L7) | [huggingface.co/datasets/OpenBallast/ballast-t0](https://huggingface.co/datasets/OpenBallast/ballast-t0) |
| **Eval sets** (50k linked probes) | [huggingface.co/datasets/OpenBallast/ballast-evalsets](https://huggingface.co/datasets/OpenBallast/ballast-evalsets) |
| **Live demo endpoint** (MCP + HTTP, L0–L5) | [mcp.openballast.org](https://mcp.openballast.org) — see [docs/mcp.md](docs/mcp.md) |

## Try it

Locally, against Ollama (this is the plug-and-play path):

```bash
uvx --from git+https://github.com/OpenBallast/ballast-cli openballast pull --level 3
uvx --from git+https://github.com/OpenBallast/ballast-cli openballast serve
# point your client at http://localhost:11435/v1 instead of :11434 — done
```

(PyPI package coming; it becomes just `uvx openballast`.)

First live A/B on a 0.5B model: raw answer to "Where was Douglas Adams born?"
was *Dublin*; through the proxy, *Cambridge*.

Or without installing anything:

```bash
curl "https://mcp.openballast.org/lookup?question=Where+was+Douglas+Adams+born%3F&level=5"
```

The hosted endpoint is a demo; canonical downloads live on Hugging Face.

## Docs

- [THESIS.md](THESIS.md) — thesis, methodology, measured numbers, caveats
- [docs/artifact.md](docs/artifact.md) — corpus layout, quantization levels, how to load
- [docs/mcp.md](docs/mcp.md) — demo endpoint reference

## Status

Research phase. Done and published: two model families end to end, the
bf16/fp8/nf4 quant sweep, the realized-retrieval measurement, model-aware
selection (a negative result), cross-family miss-set overlap, and the
hallucination probes beyond recall. Still running: GGUF K-quant cells (Q6_K,
Q4_K_M — whether integer quantization moves these curves) and models above 12B.
Results land here as they complete. Tooling for building and tuning your own
ballasts will be open-sourced separately.

Docs are CC-BY-4.0. Corpus data is CC0 (Wikidata contributors).
