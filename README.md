# ⚓ OpenBallast

**Quantize the weights. Ballast the knowledge.**

## What this is

A language model is a reasoning engine fused to an encyclopedia. Most of what a
bigger model buys you is the encyclopedia, and parameters are an expensive place
to store facts.

Ballast separates the two. The knowledge ships as a versioned, CC0 file that
sits next to the model. It comes in nested sizes (L0 = 36 MB up to L7 = 1.5 GB),
so you pick a knowledge level the same way you pick a quant. The model reads
from it at answer time.

Measured result: a 2B model plus a 180 MB ballast file exceeds a 12B model's
factual accuracy. The same gain through parameters costs ~19 GB of weights —
roughly 100× more bytes, and ballast bytes sit in flash/RAM rather than VRAM.

## Pain points

**"Small models are brain-damaged."** Gemma-4-E2B answers 60.8% of factual
probes from its weights, and 86.8% when handed a short evidence snippet. It can
use facts it's given; it doesn't have room to memorize them. The problem is
missing knowledge, not missing capability.

**"It makes things up."** Same model, same probes: hallucination drops from 0.24
to 0.07 with ballast attached. In the live demo, two raw models gave different
wrong birthplaces for Douglas Adams (London, Surrey); one corpus lookup
corrected both.

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
factual work, that's ~6× less compute per token, and the sidecar serves from
commodity storage — the demo endpoint runs on a $0/month free tier. Knowledge
becomes one shared copy per cluster instead of a per-GPU parameter cost.

## Who this is for

- **Local runners** — pair a small model with the level that fits your disk.
  L2 (107 MB) already lifts E2B past E4B's raw accuracy.
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
- **Model-aware tuning** (in progress) — a model's knowledge gaps are
  predictable from corpus features (AUC ~0.8), so a ballast can target what a
  specific model doesn't know.

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

Matrix numbers use oracle-grade entity resolution — an upper bound on retrieval.
The live demo uses real linking and shows the same effect. Caveats:
[THESIS.md §5](THESIS.md).

## Get it

| what | where |
|---|---|
| **T0 corpus** (25.4M entities, 197M triples, 1.51 GB, levels L0–L7) | [huggingface.co/datasets/OpenBallast/ballast-t0](https://huggingface.co/datasets/OpenBallast/ballast-t0) |
| **Eval sets** (50k linked probes) | [huggingface.co/datasets/OpenBallast/ballast-evalsets](https://huggingface.co/datasets/OpenBallast/ballast-evalsets) |
| **Live demo endpoint** (MCP + HTTP, L0–L5) | [mcp.openballast.org](https://mcp.openballast.org) — see [docs/mcp.md](docs/mcp.md) |

## Try it

```bash
curl "https://mcp.openballast.org/lookup?question=Where+was+Douglas+Adams+born%3F&level=5"
```

Or add `https://mcp.openballast.org/mcp` to any MCP client and call `lookup`
before answering factual questions. The endpoint is a demo; canonical downloads
live on Hugging Face.

## Docs

- [THESIS.md](THESIS.md) — thesis, methodology, measured numbers, caveats
- [docs/artifact.md](docs/artifact.md) — corpus layout, quantization levels, how to load
- [docs/mcp.md](docs/mcp.md) — demo endpoint reference

## Status

Research phase. The experiment matrix (model size × weight quant × corpus level,
plus model-aware tuned ballasts and hallucination probes beyond recall) is
running; results land here as they complete. Tooling for building and tuning
your own ballasts will be open-sourced separately.

Docs are CC-BY-4.0. Corpus data is CC0 (Wikidata contributors).
