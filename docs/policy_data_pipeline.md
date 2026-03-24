# Policy Data Pipeline

The policy-side extraction scaffold now uses OPP-115 directly:

- [ingest_opp115.py](/Users/mac/Desktop/KlavoraAI/scripts/ingest_opp115.py)
- [build_policy_examples.py](/Users/mac/Desktop/KlavoraAI/scripts/build_policy_examples.py)

## What it uses

`ingest_opp115.py`

- reads [sanitized_policies](/Users/mac/Desktop/KlavoraAI/data/raw/opp-115/sanitized_policies)
- reads [pretty_print](/Users/mac/Desktop/KlavoraAI/data/raw/opp-115/pretty_print) when available, then falls back to [pretty_print_uniquified](/Users/mac/Desktop/KlavoraAI/data/raw/opp-115/pretty_print_uniquified)
- builds normalized policy documents with:
  - cleaned raw policy text
  - segment-level privacy practice summaries
  - weak extraction targets mapped to the project schema

`build_policy_examples.py`

- converts normalized policy documents into chunked extraction examples
- preserves `source_doc_id`, `chunk_id`, source spans, and quality flags
- writes train/val/test splits under [policy_main](/Users/mac/Desktop/KlavoraAI/data/processed/policy_main)
- writes a dataset quality report at [quality_report.json](/Users/mac/Desktop/KlavoraAI/data/processed/policy_main/quality_report.json)

## Recommended run order

```bash
python3 scripts/ingest_opp115.py
python3 scripts/build_policy_examples.py --target-chunk-tokens 800 --max-tokens 1024
```

## Important caveat

This first OPP-115 builder is extraction-focused and weakly supervised. It uses the policy text plus the corpus' human-readable privacy-practice summaries, but it does not yet add a real summary dataset like ToS-Summaries. That should be the next policy-side data addition after extraction is stable.
