# Policy Extraction Runbook

Policy stays extraction-only until contract passes the promotion gate.

## Local Build

1. Normalize OPP-115:

```bash
python3 scripts/ingest_opp115.py
```

2. Build the chunked extraction dataset:

```bash
python3 scripts/build_policy_examples.py \
  --target-chunk-tokens 800 \
  --max-tokens 1024 \
  --output-dir data/processed/policy_main
```

Artifacts:

- [policy_opp115_documents.jsonl](/Users/mac/Desktop/KlavoraAI/data/normalized/policy_opp115_documents.jsonl)
- [train.jsonl](/Users/mac/Desktop/KlavoraAI/data/processed/policy_main/train.jsonl)
- [val.jsonl](/Users/mac/Desktop/KlavoraAI/data/processed/policy_main/val.jsonl)
- [test.jsonl](/Users/mac/Desktop/KlavoraAI/data/processed/policy_main/test.jsonl)
- [quality_report.json](/Users/mac/Desktop/KlavoraAI/data/processed/policy_main/quality_report.json)

## Training Recipe

Use [policy_gemma3_4b_qlora_sanity.yaml](/Users/mac/Desktop/KlavoraAI/training/configs/policy_gemma3_4b_qlora_sanity.yaml) first, then [policy_gemma3_4b_qlora_main.yaml](/Users/mac/Desktop/KlavoraAI/training/configs/policy_gemma3_4b_qlora_main.yaml).

Dry-run locally:

```bash
python3 scripts/train_unsloth.py --config training/configs/policy_gemma3_4b_qlora_sanity.yaml --dry-run
python3 scripts/train_unsloth.py --config training/configs/policy_gemma3_4b_qlora_main.yaml --dry-run
```

Keep the same T4-safe settings as contract. Adapter-only save remains the default.
