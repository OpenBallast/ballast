# Ballast: Quantize the Weights, Ballast the Knowledge

*OpenBallast — working notes, v0.3 (August 2026). Numbers below are measured, not projected. This document is the overview; the full result write-ups live in [docs/](docs/).*

---

## 1. Thesis

**Parameters are the most expensive place to store facts.**

A language model's weights conflate two different things: a *reasoning engine* and a *knowledge inventory*. Modern small models (2–4B) have surprisingly capable engines — they read, compose, and follow instructions well — but their inventories are hard-capped by parameter count (empirically ~2 bits of recallable fact per parameter; Allen-Zhu & Li, 2024), and long-tail facts are precisely what scaling struggles to buy (Kandpal et al., 2023). The community's verdict on small models ("brain damaged," "unusable") is mostly a knowledge complaint wearing a general verdict: **Gemma-4-E2B answers 60.8% of factual probes from its weights, and 86.8% when handed compact evidence it has never seen.** The capability to *use* facts is present; the facts are not.

So the claim: knowledge should ship as a **separate, versioned, compressed, quantizable artifact** — a *ballast* — that any model loads next to its weights: ~40–100× cheaper per fact than full-precision parameters (measured; ~40× with the measured real-world linker, ~15× against the larger model's cheapest intact quant), stored in flash/RAM instead of VRAM, updateable monthly without retraining, quantizable in nested levels like weight quants, and CC0, rebuilt from public dumps.

One sentence: **quantize the weights, ballast the knowledge.**

### What's borrowed, what's new

The mechanism is the semiparametric/retrieval-augmented line — kNN-LM (Khandelwal et al., 2020), REALM (Guu et al., 2020), RAG (Lewis et al., 2020), RETRO (Borgeaud et al., 2022), Atlas (Izacard et al., 2022) — with Mallen et al. (2023) supplying the fact this project leans hardest on: parametric memory tracks entity *popularity*, and retrieval helps exactly where popularity runs out. The packaging follows two format precedents — **GGUF** for weights, **ZIM** for offline reading — and the storage layer is borrowed whole from data engineering: Parquet under zstd, Hive-style partitioning (which is what makes levels real: buckets are directory prefixes pruned at download time), DuckDB as the in-process engine.

What this project adds is artifact discipline on top of that literature: the datastore as a *versioned, CC0, standalone release*; nested rank-bucket **levels** that make knowledge/bytes a user-facing knob analogous to weight quants; a measured **corpus-bytes vs parameter-bytes exchange rate** across model sizes and weight quants; and a measured verdict on per-model **tuned ballasts** (negative — generic selection wins at equal bytes). The numbers measure those additions, not retrieval augmentation per se.

## 2. The artifact in brief

**Ballast T0**: cleaned Wikidata triples as hive-partitioned Parquet. Canonical distribution: [huggingface.co/datasets/OpenBallast/ballast-t0](https://huggingface.co/datasets/OpenBallast/ballast-t0).

| property | value |
|---|---|
| entities / triples / properties | 25.4M / 197.4M / 13,704 |
| published size | **1.51 GB** (zstd parquet; 8.67 GB as raw TSV) |
| license | CC0 (Wikidata) |
| levels | L0 (36 MB, top 0.5% of entities) … L7 (1,507 MB, everything) — nested byte-budget prefixes |

Truncation is not a filter — deeper buckets are simply never downloaded, and evidence degrades honestly (a subject outside the cutoff has no evidence; a triple whose object falls outside is dropped rather than rendered as a bare Q-id). Full layout, schemas, and provenance: [docs/artifact.md](docs/artifact.md).

## 3. Headline numbers

| finding | number | details |
|---|---|---|
| Accuracy ladder (bf16, 50,147 probes) | E2B 0.608 → **0.868** ballasted; E4B 0.662 → 0.910; 12B 0.683 → 0.910 — floors spread, ceilings compress | [equal-bytes](docs/results-equal-bytes.md) |
| Second family replicates | Qwen3.5-0.8B 0.324 → 0.784; ballasted 4B beats ballasted 9B (0.831 vs 0.819) | [equal-bytes](docs/results-equal-bytes.md) |
| Equal-bytes crossings | E2B→E4B **~40×**; E2B→12B **~100×**; E2B@nf4→31B@nf4 **~35×** cheaper per byte than parameters | [equal-bytes](docs/results-equal-bytes.md) |
| Realized retrieval | a real linker holds **~63%** of the oracle gain; wrong links ≈ neutral; full-corpus crossing holds end-to-end | [retrieval](docs/results-retrieval.md) |
| Two-pass support routing | +4.30pp (developed) / **+3.62pp held-out**, registered bands, guards held | [retrieval](docs/results-retrieval.md) |
| Equal-VRAM verdict | 9B + corpus + two-pass 0.3315 vs 12B alone 0.1565 = **+17.50pp** at less VRAM | [retrieval](docs/results-retrieval.md) |
| Quantization | Q6_K free in all six cells; E4B collapses at ~4-bit (ceiling 0.910 → 0.650) while 12B rides everything; recall and reading are separately damageable | [quantization](docs/results-quantization.md) |
| Tuned ballasts | negative — generic beats model-tuned at every equal-bytes level; oracle closes 213% of the gap in 0.9 MB | [boost](docs/results-boost.md) |
| Hallucination | composition PASS all 36 readings; unanswerable/false-premise FAIL all 12 cells — fabrication rises with grounding | [hallucination](docs/results-hallucination.md) |

## 4. Result write-ups

- [docs/results-equal-bytes.md](docs/results-equal-bytes.md) — the ladders, the three crossings, rate–distortion, competence AUC, live replication
- [docs/results-quantization.md](docs/results-quantization.md) — the nf4 cliff, the dose–response sweep, the cross-family double dissociation, the Q6_K verdict
- [docs/results-boost.md](docs/results-boost.md) — model-aware selection: three decisively negative arms, the oracle ceiling, miss-set overlap
- [docs/results-hallucination.md](docs/results-hallucination.md) — composition vs fabrication, the TruthfulQA control, the answerability signal
- [docs/results-retrieval.md](docs/results-retrieval.md) — realized retrieval (de-oracled), the transfer law, the two-pass support-routed architecture
- [docs/methodology.md](docs/methodology.md) — probes, the two-pass composition trick, the matrix, the boost design, registered bands
- [docs/artifact.md](docs/artifact.md) — the T0 artifact: layout, schemas, loading, provenance
- [docs/mcp.md](docs/mcp.md) — demo endpoint reference

## 5. Honest caveats

- **Retrieval realization is a band, not a point**: matrix numbers are the perfect-resolution ceiling; the measured non-generative linker realizes ~63% of the grounding gain end-to-end.
- **Coverage cap**: the corpus grounds 90.5% of probe subjects; the remainder bounds grounded accuracy.
- **Abstention-thresholded metrics**: raw forced-choice gaps are smaller than triad numbers.
- **Contamination**: models have seen Wikipedia/Wikidata in pretraining. The self-generated benchmark is uncontaminated by construction, and the interesting quantity (grounded − ungrounded delta) is contamination-*conservative* — pretraining exposure inflates the raw floor, not the gain.
- **Grounding suppresses correct abstention**: on unanswerable probes, fabrication rises monotonically with corpus coverage. Evidence injection without an answerability signal trades multi-hop hallucination for false-premise fabrication.
- **The TruthfulQA control is not clean**: only 3 of the 12 (model, quant) cells stay inside the 2% margin under grounding; 9 move by up to ±0.07, inconsistent sign ([details](docs/results-hallucination.md)).
- **Quant damage does not have one shape**: total collapse, proportional shrinkage, reading-only loss, recall-only loss that ballast fully repairs, or nothing at all — not predictable from size, bit-count, or format. Measure the specific model, with both a raw and a grounded probe.
- **No pack-time signal knows whether evidence suffices**, and substring support is a delivery detector, not a truth detector — supported-but-wrong answers currently have no verification signal ([details](docs/results-retrieval.md)).
- Facts absent from Wikidata are absent from the ballast; T0 measures the triple-shaped slice of knowledge.

## 6. Current state

The boundary matrix is complete through the Qwen3.5 quant grid: two model families end to end, the full quant sweep on all four pivots, the hallucination families over twelve cells, and three boost arms — all with verdicts recorded above. The Stage F/G retrieval research concluded with the two-pass support-routed architecture validated on a held-out population; every cheaper lever (uniform pack policies, bigger encoders, cross-encoders, pack geometry) was measured to a registered kill. The open lines are: a genuinely new verification signal for supported-but-wrong answers (NLI-class verifier or sample-consistency), match-side retrieval work (query expansion, index-time context enrichment), and corpus coverage expansion beyond the triple-shaped slice.

## 7. Where things live

| surface | role |
|---|---|
| [huggingface.co/OpenBallast](https://huggingface.co/OpenBallast) | **canonical artifacts** — the T0 corpus and eval sets, versioned |
| [github.com/OpenBallast/ballast-cli](https://github.com/OpenBallast/ballast-cli) | **local tooling** — `ballast pull/serve/build/profile/eval`: OpenAI grounding proxy + MCP server over the corpus, offline |
| [mcp.openballast.org](https://mcp.openballast.org) | live **demo endpoint** (MCP + HTTP) — levels L0–L5 served from a $0/month free-tier stack |
| [github.com/OpenBallast](https://github.com/OpenBallast) | docs, spec, this document |
| [openballast.org](https://openballast.org) | landing |

Data: Wikidata contributors, CC0.

## References

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
