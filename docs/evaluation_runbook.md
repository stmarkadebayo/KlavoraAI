# Evaluation Runbook

Every tutorial run should end with evaluation. Training without evaluation is not the default flow in this repo.

Prediction files must contain at least:

- `example_id`
- `raw_output`

If you already parsed the output, you can also include:

- `parsed_output`

## Build The Fixed 12-Example Benchmark

Demo contract benchmark:

```bash
python3 scripts/build_eval_benchmark.py \
  --input data/demo/contract/test.jsonl \
  --output evaluation/benchmarks/contract_demo_benchmark.jsonl \
  --limit 12
```

Demo policy benchmark:

```bash
python3 scripts/build_eval_benchmark.py \
  --input data/demo/policy/test.jsonl \
  --output evaluation/benchmarks/policy_demo_benchmark.jsonl \
  --limit 12
```

Standard paths can use the processed `data/processed/.../test.jsonl` files instead.

## Evaluate Holdout Predictions

Contract demo example:

```bash
python3 scripts/evaluate_extraction.py \
  --gold data/demo/contract/test.jsonl \
  --quality-report data/demo/contract/quality_report.json \
  --system qwen_demo=eval/predictions/contract_qwen_demo.jsonl \
  --benchmark-predictions qwen_demo=eval/predictions/contract_qwen_demo_benchmark.jsonl \
  --output-dir evaluation/reports/contract_qwen3_4b_demo
```

Policy demo example:

```bash
python3 scripts/evaluate_extraction.py \
  --gold data/demo/policy/test.jsonl \
  --quality-report data/demo/policy/quality_report.json \
  --system qwen_demo=eval/predictions/policy_qwen_demo.jsonl \
  --benchmark-predictions qwen_demo=eval/predictions/policy_qwen_demo_benchmark.jsonl \
  --output-dir evaluation/reports/policy_qwen3_4b_demo
```

## Tutorial Acceptance

The demo/tutorial path is acceptable when the newest evaluated system meets all of these:

- JSON validity `>= 0.90`
- schema parse success `>= 0.85`
- unsupported top-level field hallucinations stay bounded
- the 12-example benchmark does not look obviously broken

## Standard Contract Promotion Gate

The stronger contract gate remains:

- JSON validity `>= 0.95`
- response truncation risk rate `<= 0.05`
- contract type accuracy `>= 0.80`
- normalized date accuracy `>= 0.75`
- manual benchmark hallucinated unsupported fields `<= 1`

Outputs:

- `report.json`
- `report.md`
- one per-example JSONL file per evaluated system
