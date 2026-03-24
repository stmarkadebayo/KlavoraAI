# Contract Extraction Runbook

This is the main v1 path. It is extraction-first, chunked, and tuned for free Colab / T4 constraints.

## Local Build

1. Normalize CUAD:

```bash
python3 scripts/ingest_cuad.py
```

2. Build the chunked extraction dataset:

```bash
python3 scripts/build_contract_examples.py \
  --target-chunk-tokens 800 \
  --max-tokens 1024 \
  --output-dir data/processed/contract_main
```

Artifacts:

- [contract_cuad_documents.jsonl](/Users/mac/Desktop/KlavoraAI/data/normalized/contract_cuad_documents.jsonl)
- [train.jsonl](/Users/mac/Desktop/KlavoraAI/data/processed/contract_main/train.jsonl)
- [val.jsonl](/Users/mac/Desktop/KlavoraAI/data/processed/contract_main/val.jsonl)
- [test.jsonl](/Users/mac/Desktop/KlavoraAI/data/processed/contract_main/test.jsonl)
- [quality_report.json](/Users/mac/Desktop/KlavoraAI/data/processed/contract_main/quality_report.json)

## Training Recipe

Use [contract_gemma3_4b_qlora_sanity.yaml](/Users/mac/Desktop/KlavoraAI/training/configs/contract_gemma3_4b_qlora_sanity.yaml) first, then [contract_gemma3_4b_qlora_main.yaml](/Users/mac/Desktop/KlavoraAI/training/configs/contract_gemma3_4b_qlora_main.yaml).

Recipe defaults:

- `unsloth/gemma-3-4b-it`
- adapter-only save
- `max_seq_length = 1024`
- trainer `max_length = 1024`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 4`
- `use_gradient_checkpointing = "unsloth"`

Dry-run locally:

```bash
python3 scripts/train_unsloth.py --config training/configs/contract_gemma3_4b_qlora_sanity.yaml --dry-run
python3 scripts/train_unsloth.py --config training/configs/contract_gemma3_4b_qlora_main.yaml --dry-run
```

The training script now saves:

- config snapshot
- preflight report
- training metrics JSON
- adapter weights

## Colab Notes

- Upload only `train.jsonl`, `val.jsonl`, and `test.jsonl`.
- The training path relies on raw `messages`, applies Gemma chat templating, then runs `train_on_responses_only`.
- Do not run merged save, GGUF export, or merged Hub push during iteration.
