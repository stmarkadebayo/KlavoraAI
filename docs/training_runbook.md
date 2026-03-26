# Training Runbook

This repo now has one public default training path and one stronger upgrade path.

## Default Path

- primary model: `Qwen3 4B`
- fallback model: `Llama 3.2 3B`
- dataset tier: `demo`
- target environment: free Colab / T4-class GPU

## Upgrade Path

- model: `Qwen3 4B`
- dataset tier: `standard`
- more stable GPU time preferred

## Important Environment Constraint

Use this Mac for:

- data preparation
- config authoring
- dry-runs
- evaluation scripts

Run actual fine-tuning on:

- Google Colab
- a Linux GPU VM
- WSL with NVIDIA GPU

## Primary Runbooks

- `docs/contract_extraction_runbook.md`
- `docs/policy_extraction_runbook.md`
- `docs/evaluation_runbook.md`

## Local Validation

```bash
python3 scripts/train_unsloth.py --config training/configs/contract_qwen3_4b_demo.yaml --dry-run
python3 scripts/train_unsloth.py --config training/configs/contract_llama32_3b_demo.yaml --dry-run
python3 scripts/train_unsloth.py --config training/configs/policy_qwen3_4b_demo.yaml --dry-run
python3 scripts/train_unsloth.py --config training/configs/policy_llama32_3b_demo.yaml --dry-run
python3 scripts/train_unsloth.py --config training/configs/contract_qwen3_4b_standard.yaml --dry-run
python3 scripts/train_unsloth.py --config training/configs/policy_qwen3_4b_standard.yaml --dry-run
```

## Caveat

The demo configs are tuned for reproducibility and accessibility, not maximum quality. If you have more compute, move to the `standard` configs, keep adapter-only saves during iteration, and run the full evaluation flow after every training run.
