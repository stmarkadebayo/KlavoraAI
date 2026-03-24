# Evaluation Runbook

The evaluation path assumes you already ran inference and saved predictions as JSONL with at least:

- `example_id`
- `raw_output`

If you already parsed the model output yourself, you can also include:

- `parsed_output`

## Build The Fixed Demo Benchmark

Contract:

```bash
python3 scripts/build_eval_benchmark.py \
  --input data/processed/contract_main/test.jsonl \
  --output evaluation/benchmarks/contract_demo_benchmark.jsonl \
  --limit 12
```

Policy:

```bash
python3 scripts/build_eval_benchmark.py \
  --input data/processed/policy_main/test.jsonl \
  --output evaluation/benchmarks/policy_demo_benchmark.jsonl \
  --limit 12
```

## Evaluate Holdout Predictions

Contract example:

```bash
python3 scripts/evaluate_extraction.py \
  --gold data/processed/contract_main/test.jsonl \
  --quality-report data/processed/contract_main/quality_report.json \
  --system base=eval/predictions/contract_base.jsonl \
  --system current_adapter=eval/predictions/contract_current_adapter.jsonl \
  --system improved_adapter=eval/predictions/contract_improved_adapter.jsonl \
  --output-dir evaluation/reports/contract_extract_v2_chunked_fulltext
```

Policy example:

```bash
python3 scripts/evaluate_extraction.py \
  --gold data/processed/policy_main/test.jsonl \
  --quality-report data/processed/policy_main/quality_report.json \
  --system base=eval/predictions/policy_base.jsonl \
  --system adapter=eval/predictions/policy_adapter.jsonl \
  --output-dir evaluation/reports/policy_extract_v1_opp115
```

Outputs:

- `report.json`
- `report.md`
- one per-example JSONL file per system

## Contract Promotion Gate

The default contract gate is:

- JSON validity >= `0.95`
- response truncation risk rate <= `0.05`
- contract type accuracy >= `0.80`
- normalized date accuracy >= `0.75`
- manual benchmark hallucinated unsupported fields <= `1`

Do not advance policy training as the main focus until contract clears that gate.
