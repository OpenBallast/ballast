# ⚓ OpenBallast

**Quantize the weights. Ballast the knowledge.**

Small models reason fine — they just don't *know* much. Ballast is a versioned, CC0,
rank-quantized knowledge artifact you pair with any local model: pick your knowledge
level like you pick a GGUF quant, and a 2B model stops hallucinating facts it never
had room to memorize.

**Measured so far** (full methodology and tables in [THESIS.md](THESIS.md)):

- **+180 MB of ballast lifts Gemma-4-E2B past a 12B's factual accuracy** — the same
  gain via parameters costs ~19 GB. ~100× byte advantage.
- **Hallucination 0.24 → 0.07** on the same model, same probes.
- Grounded ceilings compress across model sizes: once ballasted, E4B ≈ 12B.
  Generations differ in what they *know*, far less in what they can *read*.

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
before answering factual questions.

## Docs

- [THESIS.md](THESIS.md) — thesis, methodology, measured numbers, caveats
- [docs/artifact.md](docs/artifact.md) — corpus layout, quantization levels, how to load
- [docs/mcp.md](docs/mcp.md) — demo endpoint reference

## Status

Research phase. The boundary-condition experiment matrix (model size × weight quant ×
corpus level, plus model-aware tuned ballasts and hallucination-beyond-recall probes)
is running; results land here as they complete. Tooling for building and tuning your
own ballasts will be open-sourced separately.

Docs are CC-BY-4.0. Corpus data is CC0 (Wikidata contributors).
