# Contract Extraction Runbook

This repo now has two contract paths:

- `demo`: free-Colab-first, committed dataset subset, 1 epoch
- `standard`: stronger run on the full processed dataset, 2 epochs

Start with `Qwen3 4B`. Switch to `Llama 3.2 3B` only if free Colab repeatedly OOMs, disconnects, or becomes unusably slow.

## Demo Path

Build or refresh the committed demo dataset:

```bash
python3 scripts/ingest_cuad.py
python3 scripts/build_contract_examples.py --profile demo
```

Artifacts:

- `data/demo/contract/train.jsonl`
- `data/demo/contract/val.jsonl`
- `data/demo/contract/test.jsonl`
- `data/demo/contract/manifest.json`
- `data/demo/contract/quality_report.json`

Dry-run the primary config locally:

```bash
python3 scripts/train_unsloth.py --config training/configs/contract_qwen3_4b_demo.yaml --dry-run
```

Fallback config:

```bash
python3 scripts/train_unsloth.py --config training/configs/contract_llama32_3b_demo.yaml --dry-run
```

Use the same config file on Colab or Linux for actual training.

## Standard Path

Build the full processed contract dataset:

```bash
python3 scripts/ingest_cuad.py
python3 scripts/build_contract_examples.py --profile standard
```

Artifacts:

- `data/processed/contract_main/train.jsonl`
- `data/processed/contract_main/val.jsonl`
- `data/processed/contract_main/test.jsonl`
- `data/processed/contract_main/manifest.json`
- `data/processed/contract_main/quality_report.json`

Dry-run:

```bash
python3 scripts/train_unsloth.py --config training/configs/contract_qwen3_4b_standard.yaml --dry-run
```

## Free-Colab Settings

The demo configs lock these defaults:

- `model_name = unsloth/Qwen3-4B`
- `load_in_4bit = true`
- `full_finetuning = false`
- `max_seq_length = 1024`
- trainer `max_length = 1024`
- `per_device_train_batch_size = 1`
- `per_device_eval_batch_size = 1`
- `gradient_accumulation_steps = 4`
- `num_train_epochs = 1`
- LoRA only
- adapter-only save

The fallback Llama config keeps the same training recipe and only swaps the base model and chat-template settings.

## Dataset Rules

The demo builder keeps only:

- extraction rows
- `truncation_risk:low`
- `weak_label_confidence:medium` or `weak_label_confidence:high`
- `token_count_estimate <= 1024`
- at least 2 populated fields across:
  - `parties`
  - `effective_date`
  - `renewal_terms`
  - `payment_terms`
  - `termination_terms`
  - `confidentiality_terms`
  - `key_obligations`

## Colab Notes

- Upload only `train.jsonl`, `val.jsonl`, and `test.jsonl`.
- The training script uses the raw `messages` field, applies the model chat template, then runs `train_on_responses_only`.
- Do not run merged save, GGUF export, or merged Hub push in the default path.
- If the runtime is unstable with Qwen, switch to the Llama fallback config instead of increasing complexity.
