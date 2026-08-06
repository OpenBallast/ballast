# ⚓ OpenBallast

## TL;DR

We measured how much of a bigger model's factual advantage is just memorized
trivia, and whether you can buy that back with a file instead of with
parameters. You can, and it's 40–100× cheaper per byte.

**What we found, on 50,147 factual questions plus a 43,000-probe hallucination
suite, across two model families:**

- A 2B model with a 470 MB fact file beside it beats a 12B model on its own —
  measured with a real lookup in the loop, not an idealized one. Buying that
  accuracy with parameters costs ~19 GB of weights at full precision; even
  charging the 12B at the cheapest quantization that leaves it intact (~7 GB),
  the file route is still ~15× cheaper per byte.
- Answering from memory alone, a 2B, a 4B and a 12B from the same family get
  61%, 66% and 68% of those questions right. Give all three the same file to
  look facts up in and they land at 87%, 91% and 91%. What separates a small
  model from a big one, factually, is mostly what it memorized, not what it can
  do.
- Made-up answers drop from 24% to 7%. But on questions with no true answer,
  evidence makes fabrication *worse* (24% to 41% on the 12B). This
  fixes answerable questions; it does not teach a model to abstain.
- 4-bit quantization damages some models and not others, unpredictably from
  size. Accuracy *with* evidence tells you which case you're in: a forgetful
  model recovers, a damaged one doesn't.
- Building each model a personalized corpus of what it doesn't know loses to
  one generic corpus, every time. Knowing a model's gaps isn't enough. You'd
  also need to know what people will ask, and the corpus can't tell you that.

**Why this matters if you deploy models.** The default answer to "it gets facts
wrong" is a bigger model, and you pay for that in VRAM on every GPU you run.
This says the cheaper move is a small engine plus a file. That file is static,
versioned, auditable, works offline, and is rebuilt from public dumps whenever
the knowledge needs refreshing — you swap a file and retrain nothing. It also
gives you a diagnostic for whether your quantized model is actually intact.

So go ahead. Download more VRAM.

## Core idea

A language model is two things fused together: something that can reason and
read, and something that has memorized an encyclopedia. Most of what you get
from a bigger model is the encyclopedia, and parameters are an expensive place
to keep facts.

Ballast splits them apart. The facts ship as a plain file that sits next to the
model, free for anyone to copy. It comes in sizes, from 36 MB up to 1.5 GB (the
CLI's ready-to-serve database builds run somewhat larger), so you pick how much
world knowledge you want the same way you pick a quant. The model looks things
up in it while answering.

The result we measured: a 2B model with a 470 MB file beside it answers factual
questions more accurately than a 12B model does on its own — through a real
lookup, not an oracle. Buying that same accuracy with parameters costs about
19 GB of weights at full precision, or about 7 GB at the kindest quantization
that doesn't damage the 12B. And those 470 MB sit on disk, not in VRAM.

## Problems we're trying to address

### *"Small models are dumb."*

Not exactly. They're *ignorant*.

Gemma-4-E2B gets 61% of factual questions right from memory. Hand it a short
block of relevant evidence and it gets 87%. It could always use the fact; it
just never had room to store it.

That gap is what we're trying to measure and exploit.

### *"It makes things up."*

Attaching the corpus cuts hallucination on that same model from **24% to 7%**.

It isn't just copying answers out of the text, either. On questions that need
two facts chained together (*"where was the director of this film born?"*, where
neither line contains the answer alone), made-up answers drop by 3–20×.

However, we found that when a question has **no** true answer, meaning a false
premise or a detail nobody ever recorded, evidence makes things *worse*.
Fabrication climbs by half to three-quarters (24% → 41% on the 12B). Handing a
model a page about the right person seems to read as permission to answer, even
when the page doesn't contain the answer.

So this fixes questions that have answers. It does not teach a model to say "I
don't know." Full write-up: [docs/results-hallucination.md](docs/results-hallucination.md).

### *"It answered wrong — can it notice?"*

Sometimes, and cheaply: after the model answers, a plain string test checks
whether the answer is actually supported by the evidence it was given;
unsupported answers get one retry against a deeper, recall-first evidence pack.
That two-pass check added **+4.30** accuracy points on the population it was
developed on and **+3.62** on a held-out one, with the retry almost never
breaking an already-correct answer. The honest limit: it catches answers the
evidence never supported, not answers that are supported *and* wrong — details
in [docs/results-retrieval.md](docs/results-retrieval.md).

### *"My GPU only has 8/12/16 GB."*

Every gigabyte of weights spent memorizing obscure facts is a gigabyte not spent
on context or a better engine. Ballast moves those facts to disk, where they're
cheap.

The whole demo, two models plus lookups, runs on one 16 GB desktop card (a 4070
Ti SUPER was used).

### *"The model's knowledge is stale."*

Weights only learn at training time. The corpus is built from public dumps and
rebuilt as those dumps land.

You update what the model knows by swapping a file, with no retraining and no
redownloading the model.

### *"We can't send data to anyone's API."*

Every common fix for hallucination (web search, hosted RAG, embedding APIs)
needs a network connection. This is a static file. Copy it across the air gap
once and the knowledge layer works offline, forever.

Useful for defense, healthcare, ships, factory floors, and anywhere data isn't
allowed to leave. It's also one versioned file, so you can say exactly what the
model does and doesn't have access to.

### *"Serving is too expensive."*

If a 2B engine with a knowledge file matches a 12B on factual work, you're
running 6× fewer parameters per token. (The evidence does spend context: a
grounded question carries a few hundred extra prompt tokens.)

The file itself serves from ordinary storage. One copy of the knowledge per
cluster, instead of paying for it in every GPU.

## Who this is for

- **Local runners.** Pair a small model with whatever level fits your disk.
  Through the real lookup, 470 MB lifts a 2B past a 12B's factual accuracy —
  and the curve is steep at the start, so smaller levels buy most of the
  benefit.
- **Edge and embedded.** Engine in silicon, facts in flash, knowledge updated
  over the air without touching the model.
- **Air-gapped operators.** The knowledge layer crosses the gap as one file you
  can audit, and runs with no network at all.
- **Fleet operators.** Smaller engines, one shared file mounted per cluster,
  refreshed on whatever schedule you want.
- **Owners of older models.** Once both can look things up, a 4B matches a 12B.
  Models age much faster in what they know than in what they can do, so an old
  model may have more life left in it than you'd think.
- **Researchers.** 50k probes, full methodology, caveats documented. Start with
  [THESIS.md](THESIS.md), then the topic write-ups under [docs/](docs/).

## "Isn't this just RAG?"

Mechanically, yes: look something up, put it in the prompt. What's different is
what's being offered.

- **It comes in sizes.** L0 through L7 make knowledge-per-byte a dial, like
  Q4/Q6/Q8. The curve is measured and it's steep at the start: the first 100 MB
  is worth about 14× as much as the last 100 MB. Most of the benefit is cheap.
- **It's one public file.** Anyone can pin it, cite a version, or diff two
  versions — and once the build tooling ships (see roadmap), rebuild it from
  the same public dumps. Not a private vector database and an embedding
  pipeline you have to maintain.
- **The thing being measured is an exchange rate.** How many bytes of corpus buy
  what a byte of parameters buys, across model sizes and quantization levels.
  That number (40–100× against full-precision weights, ~15× against the
  cheapest intact quant) is the finding. "Retrieval helps" isn't news.
- **We tried to build a better, model-specific version and failed.** Details
  below; it's one of the more useful things we learned.

## Numbers

Measured on 50,147 factual questions drawn from PopQA, SimpleQA, Natural
Questions, TriviaQA, and a set we generated ourselves to sidestep
contamination. One honest note on protocol: these are 8-way multiple-choice
scores read from the model's probabilities (no generation, no judge model),
with an abstain option — open-ended generative use scores lower across the
board. Full tables in [docs/results-equal-bytes.md](docs/results-equal-bytes.md),
method in [docs/methodology.md](docs/methodology.md), caveats in
[THESIS.md](THESIS.md).

| model | on its own | with the full 1.5 GB file | made-up answers, before → after |
|---|---|---|---|
| Gemma-4-E2B | 61% | **87%** | 24% → **7%** |
| Gemma-4-E4B | 66% | **91%** | 20% → 4% |
| Gemma-4-12B | 68% | **91%** | 21% → 5% |

The interesting part is the shape. On their own the three models are spread
apart, and that spread is the memorization gap. Once all three can look things
up, they land in the same place. What separates a 2B from a 12B, factually, is
mostly what it memorized, and that part can be bought back cheaply.

Same experiment on a completely unrelated model family:

| model | on its own | with the full 1.5 GB file | made-up answers, before → after |
|---|---|---|---|
| Qwen3.5-0.8B | 32% | **78%** | 60% → **11%** |
| Qwen3.5-2B | 36% | 77% | 55% → 14% |
| Qwen3.5-4B | 43% | **83%** | 47% → **10%** |
| Qwen3.5-9B | 54% | 82% | 33% → 11% |

The pattern holds, and gets blunter: the **ballasted 4B edges past the
ballasted 9B** (83% vs 82%). With an ideal lookup, a 180 MB file lifts the
0.8B past the raw 9B — a jump that costs about 16 GB of extra full-precision
weights to buy the normal way. (The real-lookup measurement below was done on
the Gemma family, so we don't quote a realized crossing for Qwen yet.) Either
way, the 0.8B's rate of making things up falls by more than five times.

### What it costs to look things up for real

Headline tables like these usually assume the lookup always finds the right
entity. In practice, that is hopeful optimism. So we built a caveman one (no
embeddings, no model calls, just capitalized-phrase matching against a name
index) and measured what it actually delivers: **about two thirds** of the
ideal benefit. The crossings quoted at the top of this page — 470 MB to beat
the raw 12B — already have that real lookup priced in; with an ideal one,
180 MB would do it.

Interestingly, looking up the *wrong* entity was nearly harmless in our
measurements (one model, 1,025 wrong-linked questions). Feed a model facts
about the wrong Douglas Adams and it mostly ignores them. So the thing to
optimize is finding *something*, not being careful. That's a much easier
engineering problem.

With the real lookup in the loop, the headline holds: a 2B with the full file
beats a 12B on its own, comfortably — still roughly 40× cheaper per byte than
the full-precision parameter route, ~15× against the quantized one. Full
measurement: [docs/results-retrieval.md](docs/results-retrieval.md).

### If you run quantized models, this part is for you

Squeezing a model to 4-bit does wildly different things to different models, and
it isn't about size:

| model at 4-bit | on its own | with the full file |
|---|---|---|
| Gemma-4-E2B | 59% (barely changed) | 84% (barely changed) |
| Gemma-4-E4B | 49% (**down from 66%**) | 65% (**down from 91%**) |
| Gemma-4-12B | 67% (barely changed) | 90% (barely changed) |

"Q4 ruined this model" and "Q4 is basically free" are both true, of different
models, and you can't guess which from the parameter count. The middle model
here is the one that breaks.

What ballast adds is a way to tell them apart. When a model is merely *forgetful*
after quantization, the file gives the knowledge back. When quantization has
damaged its ability to read and follow evidence, the score stays low no matter
how much corpus you hand it. That is what a broken model looks like, and its
accuracy *with* evidence is what's interesting.

Practical consequence: at 4-bit the smaller Gemma is the better model once
ballasted (84% vs 65%), which reverses the usual ordering. And a 4-bit 2B plus
the entire corpus, under 3 GB all in, beats the full-precision 12B on ~24 GB.

The table above is bitsandbytes nf4. The formats people actually download, the
GGUF K-quants, behave the same way. On the two models we swept across GGUF
levels:

| at GGUF quant | Q6_K | Q4_K_M |
|---|---|---|
| Gemma-4-E4B | 66% / 91% (intact) | 50% / 72% (**broken**) |
| Gemma-4-12B | 68% / 91% | 67% / 91% |

(raw / with the full file.) Q6_K was free on every model we measured, identical
to full precision. Q4_K_M broke the same model nf4 breaks and left the other
untouched. So the cliff for a fragile model sits between 6-bit and 4-bit, it
follows the model rather than the quantization method, and accuracy *with*
evidence is still the test that tells you which case you have.

![Chart: raw and ballasted accuracy across bf16, fp8, Q6_K, Q4_K_M and nf4. The 12B lines are flat everywhere; the E4B lines plunge between Q6_K and the two 4-bit formats.](assets/figures/quant_cliff.png)

The full sweep, including a second model family where the ~4-bit story inverts,
is in [docs/results-quantization.md](docs/results-quantization.md).

### The thing that didn't work

The obvious next idea: instead of one corpus for everybody, build each model a
corpus of the facts *it personally* doesn't know. We built the machinery. It
works, in the sense that we can predict a given model's blind spots from the
corpus alone with decent accuracy, and different model families genuinely do
have different blind spots (models from the same family miss largely the same
facts; models from different families overlap far less).

It lost anyway. At every size, the plain generic corpus beat the personalized
one, decisively.

The reason is the useful part. Picking facts for a model needs two things: which
facts it doesn't know, and which facts someone is going to ask about. The corpus
can tell you the first. It cannot tell you the second, which is a property of
the people asking, not of the data. Personalized selection spent its budget on
obscure facts the model didn't know and nobody asks about, while dropping common
facts it also didn't know. When we cheated and used the actual questions to
select, 0.9 MB closed twice the model-to-model gap that 107 MB of generic
corpus closed. So the ceiling is real. It just isn't reachable from the corpus
side. Full write-up: [docs/results-boost.md](docs/results-boost.md).

## Get it

| what | where |
|---|---|
| **`ballast` CLI** (`pull`/`serve` the corpus into Ollama or any MCP client; `build` your own corpus, `profile` a model, `eval` the three-arm benchmark) | [github.com/OpenBallast/ballast-cli](https://github.com/OpenBallast/ballast-cli) |
| **The corpus** (25.4M entities, 197M facts, levels L0–L7) | [huggingface.co/datasets/OpenBallast/ballast-t0](https://huggingface.co/datasets/OpenBallast/ballast-t0) |
| **The question sets** (50k probes) | [huggingface.co/datasets/OpenBallast/ballast-evalsets](https://huggingface.co/datasets/OpenBallast/ballast-evalsets) |
| **Live demo endpoint** (MCP + HTTP) | [mcp.openballast.org](https://mcp.openballast.org), reference in [docs/mcp.md](docs/mcp.md) |

## Try it

Against a local Ollama. This is the whole setup:

```bash
uvx openballast pull --level 3
uvx openballast serve
# point your client at http://localhost:11435/v1 instead of :11434, and that's it
```

The CLI also covers the rest of the loop:

```bash
ballast build <dir>      # turn your own documents into a ballast (bring-your-own-corpus)
ballast profile -m <model>   # profile a model: what it knows raw vs what it reads from evidence
ballast eval -m <model>      # the three-arm benchmark (ungrounded / realized / saturated)
```

(Or `pip install openballast`. Source: [github.com/OpenBallast/ballast-cli](https://github.com/OpenBallast/ballast-cli).)

First live A/B on a 0.5B model, asked "Where was Douglas Adams born?" On its
own: *Dublin*. Through the proxy: *Cambridge*.

Or without installing anything:

```bash
curl "https://mcp.openballast.org/lookup?question=Where+was+Douglas+Adams+born%3F&level=5"
```

The hosted endpoint is a demo. It runs on Cloudflare's free tier, and you can
host your own for $0. The real downloads live on Hugging Face.

## Docs

- [THESIS.md](THESIS.md): the research overview — thesis, headline numbers,
  every caveat
- [docs/results-equal-bytes.md](docs/results-equal-bytes.md): the accuracy
  ladders and the corpus-vs-parameters exchange rate
- [docs/results-quantization.md](docs/results-quantization.md): what
  quantization does to recall vs reading, and the Q6_K verdict
- [docs/results-boost.md](docs/results-boost.md): the personalized-corpus
  negative result
- [docs/results-hallucination.md](docs/results-hallucination.md): where
  grounding cuts hallucination and where it makes fabrication worse
- [docs/results-retrieval.md](docs/results-retrieval.md): the real-lookup
  measurement and the two-pass support check
- [docs/methodology.md](docs/methodology.md): probes, the composition trick,
  registered bands
- [docs/artifact.md](docs/artifact.md): what's in the corpus and how to load it
- [docs/mcp.md](docs/mcp.md): demo endpoint reference

## Roadmap

- **More corpora on Hugging Face.** T0 is Wikidata triples, so anything that
  isn't a structured fact is currently missing. Next up: encyclopedia prose,
  open textbooks for the explanatory and procedural knowledge triples can't
  hold.
- **License as a first-class knob.** CC0 was the first choice of corpus because it has
  no strings attached. Most other good sources come with terms (share-alike,
  attribution, non-commercial), and mixing them into one blob may make the whole thing unusable.
  Licenses are tracked per record, and you'll be
  able to say what you're willing to accept at download time, for example
  "commercial use only, no share-alike," and get an artifact that satisfies it.
- **Bring your own corpus.** `ballast build <dir>` already turns a directory of
  your own documents into a ballast; publishing the format and loader contract
  properly — so anyone can build one with no part of our tooling in the loop —
  is the remaining step.

## Status

Research phase, published as it lands. Done: two model families end to end
(through the Qwen3.5 quant grid), the full quantization sweep on the pivots
(bf16, fp8, GGUF Q6_K and Q4_K_M, nf4), the real-lookup measurement, the
personalized-corpus attempt (which failed, informatively), the hallucination
experiments, and the retrieval research line — concluded with the two-pass
support check validated on a held-out population. Open lines: a verification
signal for supported-but-wrong answers, match-side retrieval work, and corpus
coverage beyond structured facts. We intend to keep profiling
state-of-the-art open-source models as they come out.

Docs are CC-BY-4.0. The corpus is CC0 (thanks to Wikidata's contributors).
