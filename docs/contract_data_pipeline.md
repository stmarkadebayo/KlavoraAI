# Contract Data Pipeline

The contract data path has two output tiers:

- `standard`: full processed chunked dataset under `data/processed/contract_main`
- `demo`: deterministic filtered subset under `data/demo/contract`

Main scripts:

- `scripts/ingest_cuad.py`
- `scripts/ingest_contractnli.py`
- `scripts/ingest_legal_summarization.py`
- `scripts/build_contract_examples.py`

## What The Builder Does

`build_contract_examples.py`:

- turns normalized contract documents into chunked extraction examples
- preserves `source_doc_id`, `chunk_id`, source spans, token estimates, and quality flags
- can emit either the full `standard` tier or the filtered `demo` tier
- writes train/val/test splits, a manifest, and a quality report

## Standard Build

```bash
python3 scripts/build_contract_examples.py --profile standard
```

Writes:

- `data/processed/contract_main/train.jsonl`
- `data/processed/contract_main/val.jsonl`
- `data/processed/contract_main/test.jsonl`
- `data/processed/contract_main/manifest.json`
- `data/processed/contract_main/quality_report.json`

## Demo Build

```bash
python3 scripts/build_contract_examples.py --profile demo
```

Writes:

- `data/demo/contract/train.jsonl`
- `data/demo/contract/val.jsonl`
- `data/demo/contract/test.jsonl`
- `data/demo/contract/manifest.json`
- `data/demo/contract/quality_report.json`

## Demo Filters

The demo tier keeps only:

- extraction rows
- low truncation risk
- medium/high weak-label confidence
- `token_count_estimate <= 1024`
- at least 2 populated target fields

It then sorts deterministically by:

- higher label confidence
- richer target coverage
- lower token estimate
- stable example id seed key

## Caveat

If the CUAD full-text companion data is unavailable, the ingester still falls back to ordered joined clauses. That keeps the pipeline reproducible, but full-text joins remain the preferred upgrade path for stronger standard runs.
