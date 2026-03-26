# Policy Extraction Runbook

Policy follows the same split as contract:

- `demo`: committed free-tier dataset subset
- `standard`: stronger processed dataset run

Start with `Qwen3 4B`. Use `Llama 3.2 3B` only when free Colab cannot reliably complete the Qwen run.

## Demo Path

Build or refresh the committed demo dataset:

```bash
python3 scripts/ingest_opp115.py
python3 scripts/build_policy_examples.py --profile demo
```

Artifacts:

- `data/demo/policy/train.jsonl`
- `data/demo/policy/val.jsonl`
- `data/demo/policy/test.jsonl`
- `data/demo/policy/manifest.json`
- `data/demo/policy/quality_report.json`

Dry-run the primary config locally:

```bash
python3 scripts/train_unsloth.py --config training/configs/policy_qwen3_4b_demo.yaml --dry-run
```

Fallback config:

```bash
python3 scripts/train_unsloth.py --config training/configs/policy_llama32_3b_demo.yaml --dry-run
```

## Standard Path

Build the full processed dataset:

```bash
python3 scripts/ingest_opp115.py
python3 scripts/build_policy_examples.py --profile standard
```

Artifacts:

- `data/processed/policy_main/train.jsonl`
- `data/processed/policy_main/val.jsonl`
- `data/processed/policy_main/test.jsonl`
- `data/processed/policy_main/manifest.json`
- `data/processed/policy_main/quality_report.json`

Dry-run:

```bash
python3 scripts/train_unsloth.py --config training/configs/policy_qwen3_4b_standard.yaml --dry-run
```

## Free-Colab Settings

The demo configs use:

- `model_name = unsloth/Qwen3-4B`
- `load_in_4bit = true`
- `max_seq_length = 1024`
- trainer `max_length = 1024`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 4`
- `num_train_epochs = 1`
- adapter-only save

## Dataset Rules

The demo builder keeps only:

- extraction rows
- `truncation_risk:low`
- `weak_label_confidence:medium` or `weak_label_confidence:high`
- `token_count_estimate <= 1024`
- at least 2 populated fields across:
  - `effective_date`
  - `applies_to`
  - `required_actions`
  - `risk_flags`
  - `key_obligations`

Policy remains extraction-first in the default tutorial path. Summary tuning stays out of scope here.
