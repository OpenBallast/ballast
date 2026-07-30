# The Ballast T0 artifact

Canonical distribution: **[huggingface.co/datasets/OpenBallast/ballast-t0](https://huggingface.co/datasets/OpenBallast/ballast-t0)**

## Layout

Hive-partitioned Parquet (zstd). The partition key **is** the quantization axis:

```
entities/tier=T0/lang=en/rank_bucket={0..7}/*.parquet
triples/tier=T0/lang=en/rank_bucket={0..7}/*.parquet
properties.parquet          # 13,704 property labels — shared by every level
manifest.json               # counts, byte sizes, hashes, build provenance
```

A **level Lk** is buckets 0..k. To hold a smaller corpus, download fewer bucket
directories — nothing else changes. `properties.parquet` and `manifest.json` ride
with every level.

| level | cumulative | entities covered |
|---|---|---|
| L0 | 36 MB | top 0.5% (~127k) |
| L1 | 62 MB | top 1% |
| L2 | 107 MB | top 2% |
| L3 | 179 MB | top 4% |
| L4 | 288 MB | top 8% |
| L5 | 466 MB | top 16% |
| L6 | 756 MB | top 32% |
| L7 | 1,507 MB | all 25.4M |

Ranking: composite of log(sitelinks) and log(claim count) — notability, not
popularity alone. Buckets are nested by construction: L3 is a byte-for-byte prefix
of L7.

## Schemas

`entities`: `qid`, `label`, `description`, `aliases[]`, `sitelinks`, `n_claims`
(+ `rank_bucket` from the path)

`triples`: `qid`, `pid`, `value_type` (`entity|time|quantity|string|coord`), `value`
(+ `rank_bucket` of the *subject*)

Excluded at publish: external identifiers, media/Commons references, URLs — datatypes
that cost bytes and ground nothing a language model can use.

## Loading

DuckDB (recommended — the partition layout is made for it):

```sql
-- everything at level 3
SELECT t.qid, p.label AS prop, t.value
FROM read_parquet('t0/triples/**/*.parquet', hive_partitioning=true) t
JOIN read_parquet('t0/properties.parquet') p USING (pid)
WHERE t.rank_bucket <= 3 AND t.qid = 'Q42';
```

Rendering evidence for a model: resolve entity-valued objects to labels *within the
same level* (drop the triple if the object is outside it — a bare Q-id is not
evidence), then format:

```
Facts about {label}:
- {property}: {value}
…
```

This exact rendering is what the published measurements used, and what the
[demo endpoint](mcp.md) serves.

## Provenance

Built from the Wikidata JSON dump (2026-07 snapshot; dump version and content hashes
in `manifest.json`). Label resolution honors the `mul` language code with English
fallback. Wikimedia-internal pages (categories, templates, disambiguation, modules)
are filtered by `instance of`. Data license: **CC0** (Wikidata contributors).
Reproducible: dump → parse → rank → partition, deterministic given the manifest's
recorded parameters.
