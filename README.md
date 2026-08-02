# ⚓ OpenBallast

## TL;DR

We measured how much of a bigger model's factual advantage is just memorized
trivia, and whether you can buy that back with a file instead of with
parameters. You can, and it's 40–100× cheaper per byte.

**What we found, on 50,147 factual questions across two model families:**

- A 2B model with a 470 MB fact file beside it beats a 12B model on its own.
  Getting there with parameters costs ~19 GB of weights.
- Answering from memory alone, a 2B, a 4B and a 12B from the same family get
  61%, 66% and 68% of those questions right. Give all three the same file to
  look facts up in and they land at 87%, 91% and 91%. What separates a small
  model from a big one, factually, is mostly what it memorized, not what it can
  do.
- Made-up answers drop from 24% to 7%. But on questions with no true answer,
  evidence makes fabrication *worse* (22% to 37%). This fixes answerable
  questions; it does not teach a model to abstain.
- 4-bit quantization damages some models and not others, unpredictably from
  size. Accuracy *with* evidence tells you which case you're in: a forgetful
  model recovers, a damaged one doesn't.
- Building each model a personalized corpus of what it doesn't know loses to
  one generic corpus, every time. Knowing a model's gaps isn't enough. You'd
  also need to know what people will ask, and the corpus can't tell you that.

**Why this matters if you deploy models.** The default answer to "it gets facts
wrong" is a bigger model, and you pay for that in VRAM on every GPU you run.
This says the cheaper move is a small engine plus a file. That file is static,
versioned, auditable, works offline, and can be rebuilt from public dumps
whenever you want the knowledge refreshed, without retraining anything. It also
gives you a diagnostic for whether your quantized model is actually intact.

So go ahead. Download more VRAM.

## Core idea

A language model is two things fused together: something that can reason and
read, and something that has memorized an encyclopedia. Most of what you get
from a bigger model is the encyclopedia, and parameters are an expensive place
to keep facts.

Ballast splits them apart. The facts ship as a plain file that sits next to the
model, free for anyone to copy or rebuild. It comes in sizes, from 36 MB up to
1.5 GB, so you pick how much world knowledge you want the same way you pick a
quant. The model looks things up in it while answering.

The result we measured: a 2B model with a 470 MB file beside it answers factual
questions more accurately than a 12B model does on its own. Buying that same
accuracy with parameters costs about 19 GB of weights. And those 470 MB sit on
disk, not in VRAM.

## Problems we're trying to address

**"Small models are dumb."** Not exactly. They're *ignorant*. Gemma-4-E2B gets
61% of factual questions right from memory. Hand it a couple of relevant lines
of evidence and it gets 87%. It could always use the fact; it just never had
room to store it. That gap is what we're trying to measure and exploit.

**"It makes things up."** Attaching the corpus cuts hallucination on that same
model from 24% to 7%. It isn't just copying answers out of the text either. On
questions that require chaining two facts together (*"where was the director of
this film born?"*, where neither line contains the answer alone), made-up
answers drop by 3–20×.

However we found that when a question has **no** true answer, 
meaning a false premise or a detail nobody ever recorded,
evidence makes things *worse*. Fabrication goes from ~22% to ~37%. Handing a
model a page about the right person seems to read as permission to answer, even
when the page doesn't contain the answer. So this fixes questions that have
answers. It does not teach a model to say "I don't know."

**"My GPU only has 8/12/16 GB."** Every gigabyte of weights spent memorizing
obscure facts is a gigabyte not spent on context or a better engine.
Ballast moves those facts to disk, where they're cheap. The whole demo, two
models plus lookups, runs on one 16 GB desktop card (a 4070 TI Super was used).

**"The model's knowledge is stale."** Weights only learn at training time. The
corpus is built from public dumps and can be rebuilt as often as those dumps
land. You update what the model knows by swapping a file, with
no retraining and no redownloading the model.

**"We can't send data to anyone's API."** Every common fix for hallucination
(web search, hosted RAG, embedding APIs) needs a network connection. This is a
static file. Copy it across the air gap once and the knowledge layer works
offline, forever. Useful for defense, healthcare, ships, factory floors, and
anywhere data isn't allowed to leave. It's also one versioned file, so you can
say exactly what the model does and doesn't have access to.

**"Serving is too expensive."** If a 2B engine with a knowledge file matches a
12B on factual work, you're running 6× fewer parameters per token. (Note: We are
using a bit of context as the evidence adds a few hundred tokens to each grounded question.) The
file itself serves from ordinary storage; our demo endpoint costs $0/month. One
copy of the knowledge per cluster, instead of paying for it in every GPU.

## Who this is for

- **Local runners.** Pair a small model with whatever level fits your disk.
  179 MB is enough to push a 2B past a 4B's factual accuracy.
- **Edge and embedded.** Engine in silicon, facts in flash, knowledge updated
  over the air without touching the model.
- **Air-gapped operators.** The knowledge layer crosses the gap as one file you
  can audit, and runs with no network at all.
- **Fleet operators.** Smaller engines, one shared file mounted per cluster,
  refreshed on whatever schedule you want.
- **Owners of older models.** Once ballasted, a 4B matches a 12B. Models age
  much faster in what they know than in what they can do, so an old model may
  have more life left in it than you'd think.
- **Researchers.** 50k probes, full methodology, caveats documented. Start with
  [THESIS.md](THESIS.md).

## "Isn't this just RAG?"

Mechanically, yes: look something up, put it in the prompt. What's different is
what's being offered.

- **It comes in sizes.** L0 through L7 make knowledge-per-byte a dial, like
  Q4/Q6/Q8. The curve is measured and it's steep at the start: the first 100 MB
  is worth about 14× as much as the last 100 MB. Most of the benefit is cheap.
- **It's one public file.** Anyone can pin it, cite a version, diff two
  versions, or rebuild it from the same public dumps. Not a private vector
  database and an embedding pipeline you have to maintain.
- **The thing being measured is an exchange rate.** How many bytes of corpus buy
  what a byte of parameters buys, across model sizes and quantization levels.
  That number (40–100×) is the finding. "Retrieval helps" isn't news.
- **We tried to build a better, model-specific version and failed.** Details
  below; it's one of the more useful things we learned.

## Numbers

Measured on 50,147 factual questions drawn from PopQA, SimpleQA, Natural
Questions, TriviaQA, and a set we generated ourselves to rule out
contamination. Full tables and method in [THESIS.md](THESIS.md).

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

The pattern holds, and gets blunter: the **ballasted 4B beats the ballasted
9B**. The 0.8B with a 180 MB file overtakes the raw 9B, a jump that costs about
16 GB of extra weights to buy the normal way. And its rate of making things up
falls by more than five times.

### What it costs to look things up for real

The tables above assume the lookup always finds the right entity. Real software
doesn't. So we built a deliberately unglamorous one (no embeddings, no model
calls, just capitalized-phrase matching against a name index) and measured what
it actually delivers: **about two thirds** of the ideal benefit.

One surprise worth knowing: looking up the *wrong* entity is nearly harmless.
Feed a model facts about the wrong Douglas Adams and it mostly ignores them. So
the thing to optimize is finding *something*, not being careful. That's a much
easier engineering problem.

With the real lookup in the loop, the headline holds: a 2B with the full file
beats a 12B on its own, comfortably. The sharper version of the claim, that
180 MB is enough, needs 470 MB once you account for imperfect lookups. Still
roughly 40× cheaper than the parameter route.

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

We intend to continue profiling state of the art open source models as they come out.

### The thing that didn't work

The obvious next idea: instead of one corpus for everybody, build each model a
corpus of the facts *it personally* doesn't know. We built the machinery. It
works, in the sense that we can predict a given model's blind spots from the
corpus alone with decent accuracy, and different model families genuinely do
have different blind spots (models from the same family miss the same facts;
models from different families don't).

It lost anyway. At every size, the plain generic corpus beat the personalized
one, decisively.

The reason is the useful part. Picking facts for a model needs two things: which
facts it doesn't know, and which facts someone is going to ask about. The corpus
can tell you the first. It cannot tell you the second, which is a property of
the people asking, not of the data. Personalized selection spent its budget on
obscure facts the model didn't know and nobody asks about, while dropping common
facts it also didn't know. When we cheated and used the actual questions to
select, 0.9 MB outperformed the generic 1.5 GB corpus. So the ceiling is real.
It just isn't reachable from the corpus side.

## Hardware this ran on

No cluster, no rented A100s. Two desktops and a free-tier CDN account.

**Build and evaluation box.** Every number in this README was produced here.

| | |
|---|---|
| GPU | NVIDIA RTX PRO 4500 Blackwell, 32 GB VRAM |
| CPU | Intel Core i9-12900K, 16 cores / 24 threads |

**Demo and live-replication machine.** The end-to-end demo, the one where a
model answers questions through the corpus, runs on ordinary gaming hardware.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB VRAM |
| CPU | AMD Ryzen 7 7800X3D, 8 cores / 16 threads |

**Hosting.** The public endpoint runs entirely on Cloudflare's free tier, which
you can reproduce yourself for $0. The corpus files are served from Hugging
Face.

## Get it

| what | where |
|---|---|
| **`ballast` CLI** (download the corpus, plug it into Ollama or any MCP client) | [github.com/OpenBallast/ballast-cli](https://github.com/OpenBallast/ballast-cli) |
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

(Or `pip install openballast`. Source: [github.com/OpenBallast/ballast-cli](https://github.com/OpenBallast/ballast-cli).)

First live A/B on a 0.5B model, asked "Where was Douglas Adams born?" On its
own: *Dublin*. Through the proxy: *Cambridge*.

Or without installing anything:

```bash
curl "https://mcp.openballast.org/lookup?question=Where+was+Douglas+Adams+born%3F&level=5"
```

The hosted endpoint is a demo. It runs on Cloudflare Free Tier that you can host yourself. The real downloads live on Hugging Face.

## Docs

- [THESIS.md](THESIS.md): the research version, with methodology, every number
  and every caveat
- [docs/artifact.md](docs/artifact.md): what's in the corpus and how to load it
- [docs/mcp.md](docs/mcp.md): demo endpoint reference

## Roadmap

- **More corpora on Hugging Face.** T0 is Wikidata triples, so anything that
  isn't a structured fact is currently missing. Next up: encyclopedia prose,
  open textbooks for the explanatory and procedural knowledge triples can't
  hold, non-English builds, and domain sets.
- **License as a first-class knob.** CC0 was the right MVP choice because it has
  no strings at all. Most other good sources come with terms (share-alike,
  attribution, non-commercial), and mixing them into one blob makes the whole
  thing unusable for somebody. Licenses are tracked per record, and you'll be
  able to say what you're willing to accept at download time, for example
  "commercial use only, no share-alike," and get an artifact that satisfies it.
- **Bring your own corpus.** The format and the loader contract get published so
  anyone can turn their own documents, or their national statistics office, into
  a ballast without waiting for us.
- **Better lookups.** The current one delivers about two thirds of what perfect
  lookup would. Closing that gap is ordinary engineering and it lifts every
  number above it.
- **Knowing when there's no answer.** The fabrication result is the clearest
  open problem here: the system needs to tell the model when the evidence
  doesn't contain an answer, instead of handing over a page and hoping.
- **Multi-hop questions.** Questions that need two facts joined together already
  work when the evidence is handed over whole. Doing that lookup automatically
  needs a query planner, which the data supports and the software doesn't yet.
- **Staying current.** Right now a refresh means rebuilding. Incremental updates
  are the goal, so a new build is a small patch rather than a new download.
- **Checking output, not just feeding input.** Same corpus, opposite direction:
  take what a model wrote and flag the claims the corpus doesn't support.
- **A citable release.** A pinned version with a DOI, so results can be
  referenced and reproduced against exactly the bytes they were measured on.

## Status

Research phase, published as it lands. Done: two model families end to end, the
quantization sweep, the real-lookup measurement, the personalized-corpus attempt
(which failed, informatively), and the hallucination experiments. Still running:
GGUF quantizations and models above 12B. Tooling for building your own corpus
will be open-sourced separately.

Docs are CC-BY-4.0. The corpus is CC0 (thanks to Wikidata's contributors).
