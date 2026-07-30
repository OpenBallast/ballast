# ⚓ OpenBallast

**Quantize the weights. Ballast the knowledge.**

## The 30-second version

A language model is two things fused together: a **reasoning engine** and an
**encyclopedia**. When you buy a bigger model, you're mostly buying a bigger
encyclopedia — and parameters are just about the most expensive medium ever
invented for storing facts.

Ballast splits them apart. The encyclopedia becomes a **separate file** — a
versioned, CC0, compressed knowledge artifact you download next to your model,
the way you download a GGUF. It comes in nested sizes (L0 = 36 MB → L7 = 1.5 GB),
so you pick your knowledge level like you pick your quant. Your model reads from
it at answer time instead of guessing from memory.

The result, measured: **a 2B model + a 180 MB ballast file beats a 12B model's
factual accuracy.** Getting that gain through parameters costs ~19 GB of VRAM.
That's a **~100× byte advantage** — and the ballast bytes live in flash/RAM, not
VRAM.

## The pain points this addresses

**"Small models are brain-damaged."** The most common r/LocalLlama verdict on
2–4B models. Our measurement says the diagnosis is wrong: Gemma-4-E2B answers
**60.8%** of factual questions from its own weights, but **86.8%** when handed a
compact evidence snippet. The reasoning engine is fine. The encyclopedia is
missing. That's not brain damage — that's a library card problem, and a library
card is a lot cheaper than a brain transplant.

**"It keeps making things up."** Same model, same questions: hallucination rate
drops **0.24 → 0.07** with ballast attached. A model that can *look at* the fact
doesn't need to invent it. (Two raw models hallucinated *different* wrong
birthplaces for Douglas Adams — London and Surrey. One 440 ms lookup corrected
both.)

**"My GPU only has 8/12/16 GB."** Every GB of VRAM you spend on parameters that
exist to memorize the long tail of Wikipedia is a GB you can't spend on context,
speed, or a better engine. Ballast moves those facts to the cheapest storage you
own. A 16 GB desktop card running a 2B/4B pair plus ballast reproduces our
results end-to-end — that's the live demo.

**"The model's knowledge is frozen in 2025."** Facts baked into weights update
never. A ballast file updates monthly from public dumps, with zero retraining,
zero fine-tuning, zero re-download of the model. Version the knowledge like
software, not like a $10M training run.

**"Serving costs don't scale."** For anyone thinking about unit economics: if a
2B-engine + sidecar-knowledge setup matches a 12B on factual work, that's ~6×
less compute per token on the dominant cost line, and the knowledge sidecar
serves from commodity storage/CDN — our demo endpoint runs the entire corpus
lookup layer on a **$0/month free tier**. Knowledge stops being a per-GPU cost
and becomes a cached, shared, one-copy-per-cluster asset.

## Who this is for

- **Local runners** — pair any small model with a ballast level that fits your
  disk. L2 is 107 MB and already lifts E2B past E4B's raw accuracy.
- **Edge & embedded** — phones, robots, offline appliances: engine in silicon,
  facts in flash, updates over the air without touching weights.
- **Fleet operators** — serve smaller engines, mount one shared knowledge
  artifact per cluster, refresh it monthly.
- **Old-model owners** — grounded ceilings compress across generations (once
  ballasted, E4B ≈ 12B). Models age in what they *know* far faster than in what
  they can *read*, which means yesterday's models may revive better than you'd
  guess.
- **Researchers** — everything is measured, versioned, and reproducible from
  public dumps: 50k linked probes, exact composition methodology, honest
  caveats. Start with [THESIS.md](THESIS.md).

## "Isn't this just RAG?"

Mechanically, yes — retrieval at answer time. What's new is treating the corpus
as a **first-class, quantizable artifact** with the same discipline weights get:

- **Nested levels** (L0–L7) so knowledge/bytes is a knob you can turn, exactly
  like Q4/Q6/Q8 — and the value-per-byte curve is measured, smooth, and concave
  (the first 100 MB matter ~14× more than the last).
- **Versioned and CC0** — one canonical artifact anyone can pin, cite, diff, and
  rebuild from public dumps. Not "your vector DB, your embeddings, your chunking
  luck."
- **Measured against parameters** — the point isn't "retrieval helps" (known);
  it's *how many bytes of corpus equal how many bytes of parameters*, on the
  same probes, across model sizes and weight quants. That exchange rate
  (~40–100×) is the finding.
- **Model-aware tuning** (in progress) — each model's knowledge holes are
  predictable (AUC ~0.8 from corpus-native features), so a ballast can be tuned
  to what *your* model doesn't know.

## The numbers so far

Measured on 50,147 factual probes (PopQA, SimpleQA, Natural Questions, TriviaQA,
plus an uncontaminated self-generated set). Full tables, methodology, and
caveats in [THESIS.md](THESIS.md).

| model (bf16) | raw accuracy | + full ballast (1.51 GB) | hallucination raw → ballasted |
|---|---|---|---|
| Gemma-4-E2B | 0.608 | **0.868** | 0.242 → **0.074** |
| Gemma-4-E4B | 0.662 | **0.910** | 0.197 → 0.042 |
| Gemma-4-12B | 0.683 | **0.910** | 0.207 → 0.049 |

| to match… | ballast route | parameter route | advantage |
|---|---|---|---|
| E4B's raw accuracy (from E2B) | **+110 MB** | +4.4 GB of weights | ~40× |
| 12B's raw accuracy (from E2B) | **+180 MB** | +19.4 GB of weights | ~100× |

One honest flag: matrix numbers use oracle-grade entity resolution (an upper
bound on retrieval); the live demo uses real linking and still lands the
headline effect. Caveats are a first-class section of the thesis, not a
footnote.

## Get it

| what | where |
|---|---|
| **T0 corpus** (25.4M entities, 197M triples, 1.51 GB, levels L0–L7) | [huggingface.co/datasets/OpenBallast/ballast-t0](https://huggingface.co/datasets/OpenBallast/ballast-t0) |
| **Eval sets** (50k linked probes) | [huggingface.co/datasets/OpenBallast/ballast-evalsets](https://huggingface.co/datasets/OpenBallast/ballast-evalsets) |
| **Live demo endpoint** (MCP + HTTP, L0–L5) | [mcp.openballast.org](https://mcp.openballast.org) — see [docs/mcp.md](docs/mcp.md) |

## Try it in 10 seconds

```bash
curl "https://mcp.openballast.org/lookup?question=Where+was+Douglas+Adams+born%3F&level=5"
```

Or add `https://mcp.openballast.org/mcp` to any MCP client and call `lookup`
before answering factual questions. (The endpoint is a demo; canonical downloads
live on Hugging Face.)

## Docs

- [THESIS.md](THESIS.md) — thesis, methodology, measured numbers, caveats
- [docs/artifact.md](docs/artifact.md) — corpus layout, quantization levels, how to load
- [docs/mcp.md](docs/mcp.md) — demo endpoint reference

## Status

Research phase. The boundary-condition experiment matrix (model size × weight
quant × corpus level, plus model-aware tuned ballasts and
hallucination-beyond-recall probes) is running; results land here as they
complete. Tooling for building and tuning your own ballasts will be open-sourced
separately.

Docs are CC-BY-4.0. Corpus data is CC0 (Wikidata contributors).
