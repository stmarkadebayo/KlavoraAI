# Training Runbook

The repo now has a repeatable extraction-first path for:

- chunked contract extraction
- chunked policy extraction
- T4-safe Unsloth configs
- config snapshot + preflight + eval artifacts

## Important environment constraint

This machine is a macOS workstation. As of March 23, 2026, Unsloth's official quickstart shows:

- `Unsloth Studio` on macOS supports chat and data recipes, with MLX training still pending
- `Unsloth Core` install instructions are documented for Linux/WSL and Windows
- the current quickstart examples use Python 3.13 environments

For that reason, use this repo on the Mac for data preparation and config authoring, then run actual fine-tuning on:

- Google Colab
- a Linux GPU VM
- WSL with an NVIDIA GPU

## Primary runbooks

- [Contract extraction](/Users/mac/Desktop/KlavoraAI/docs/contract_extraction_runbook.md)
- [Policy extraction](/Users/mac/Desktop/KlavoraAI/docs/policy_extraction_runbook.md)
- [Evaluation](/Users/mac/Desktop/KlavoraAI/docs/evaluation_runbook.md)

## Local validation

```bash
python3 scripts/train_unsloth.py --config training/configs/contract_gemma3_4b_qlora_sanity.yaml --dry-run
python3 scripts/train_unsloth.py --config training/configs/contract_gemma3_4b_qlora_main.yaml --dry-run
python3 scripts/train_unsloth.py --config training/configs/policy_gemma3_4b_qlora_sanity.yaml --dry-run
python3 scripts/train_unsloth.py --config training/configs/policy_gemma3_4b_qlora_main.yaml --dry-run
```

Use Colab or Linux for actual training. Keep adapter-only saves during iteration.
