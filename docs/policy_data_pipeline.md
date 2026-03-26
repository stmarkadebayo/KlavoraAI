# Policy Data Pipeline

The policy data path also has two output tiers:

- `standard`: full processed chunked dataset under `data/processed/policy_main`
- `demo`: deterministic filtered subset under `data/demo/policy`

Main scripts:

- `scripts/ingest_opp115.py`
- `scripts/build_policy_examples.py`

## What The Builder Does

`build_policy_examples.py`:

- converts normalized OPP-115 documents into chunked extraction examples
- preserves `source_doc_id`, `chunk_id`, source spans, token estimates, and quality flags
- can emit the full `standard` tier or the filtered `demo` tier
- writes train/val/test splits, a manifest, and a quality report

## Standard Build

```bash
python3 scripts/build_policy_examples.py --profile standard
```

Writes:

- `data/processed/policy_main/train.jsonl`
- `data/processed/policy_main/val.jsonl`
- `data/processed/policy_main/test.jsonl`
- `data/processed/policy_main/manifest.json`
- `data/processed/policy_main/quality_report.json`

## Demo Build

```bash
python3 scripts/build_policy_examples.py --profile demo
```

Writes:

- `data/demo/policy/train.jsonl`
- `data/demo/policy/val.jsonl`
- `data/demo/policy/test.jsonl`
- `data/demo/policy/manifest.json`
- `data/demo/policy/quality_report.json`

## Demo Filters

The demo tier keeps only:

- extraction rows
- low truncation risk
- medium/high weak-label confidence
- `token_count_estimate <= 1024`
- at least 2 populated target fields

Policy remains extraction-only in the tutorial path. Summary tuning stays out of the default flow.
