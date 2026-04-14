# KlavoraAI

KlavoraAI is a reproducible, extraction-first document intelligence stack for two narrow domains:

- contract extraction
- policy extraction

The repo is tuned for two use cases:

- `demo`: a free-Colab-first path that anybody can run end to end
- `standard`: a stronger path that uses the same pipeline with larger processed datasets and longer runs

The default public flow is:

1. build or use the committed `data/demo` datasets
2. fine-tune `Qwen3 4B` on free Colab
3. fall back to `Llama 3.2 3B` if Colab becomes unstable
4. evaluate on the held-out test split and fixed 12-example benchmark

## Model Strategy

Primary tutorial model:

- `Qwen3 4B`

Fallback tutorial model:

- `Llama 3.2 3B`

Why this split:

- `Qwen3 4B` is the main quality/free-tier balance
- `Llama 3.2 3B` is the reliability fallback when free Colab has OOMs, disconnects, or very slow step times

Gemma configs are kept only as older archived examples. They are not part of the default docs flow.

## Project Layout

- `src/klavora_ai/`: schemas, chunking, dataset generation, config models
- `scripts/`: ingestion, dataset building, training, evaluation, benchmark generation
- `training/configs/`: demo and standard Unsloth configs
- `docs/`: runbooks for training and evaluation
- `data/demo/`: committed free-tier demo datasets
- `evaluation/benchmarks/`: committed manual-review benchmark prompts

Generated large artifacts are intentionally not committed:

- `data/raw/`
- `data/normalized/`
- `data/processed/`
- `training/outputs/`
- `adapters/`
- `evaluation/reports/`

## Default Free-Colab Path

Contract demo:

```bash
python3 scripts/ingest_cuad.py
python3 scripts/build_contract_examples.py --profile demo
python3 scripts/train_unsloth.py --config training/configs/contract_qwen3_4b_demo.yaml --dry-run
```

Policy demo:

```bash
python3 scripts/ingest_opp115.py
python3 scripts/build_policy_examples.py --profile demo
python3 scripts/train_unsloth.py --config training/configs/policy_qwen3_4b_demo.yaml --dry-run
```

Then run the same config on Colab or Linux with a GPU.

Fallback if Qwen is unstable on free Colab:

```bash
python3 scripts/train_unsloth.py --config training/configs/contract_llama32_3b_demo.yaml --dry-run
python3 scripts/train_unsloth.py --config training/configs/policy_llama32_3b_demo.yaml --dry-run
```

## Training Settings

The default demo configs are intentionally conservative:

- `load_in_4bit = true`
- `max_seq_length = 1024`
- trainer `max_length = 1024`
- `per_device_train_batch_size = 1`
- `per_device_eval_batch_size = 1`
- `gradient_accumulation_steps = 4`
- `num_train_epochs = 1`
- LoRA only
- adapter-only save

This is a reproducibility baseline, not a claim of maximum model quality.

If you have better compute, switch to the `standard` configs and larger processed datasets:

- `training/configs/contract_qwen3_4b_standard.yaml`
- `training/configs/policy_qwen3_4b_standard.yaml`

## Demo Dataset Filtering

The committed demo datasets are deterministic filtered subsets of the larger chunked datasets.

Contract demo rules:

- extraction only
- low truncation risk only
- medium/high weak-label confidence only
- token estimate `<= 1024`
- at least 2 populated target fields

Policy demo rules:

- extraction only
- low truncation risk only
- medium/high weak-label confidence only
- token estimate `<= 1024`
- at least 2 populated target fields

## Evaluation

Every training run should end with evaluation.

The default checks are:

- JSON validity
- schema parse success
- type/area accuracy
- normalized date accuracy
- field presence precision/recall

The tutorial acceptance bar is lighter than the standard promotion gate:

- JSON validity `>= 0.90`
- schema parse success `>= 0.85`
- no unsupported top-level field explosions
- no obviously broken outputs on the 12-example benchmark

## Runbooks

- `docs/contract_extraction_runbook.md`
- `docs/policy_extraction_runbook.md`
- `docs/evaluation_runbook.md`
- `docs/training_runbook.md`

## Caveats

- The default path is optimized for free Colab reliability, not SOTA quality.
- The strongest quality lever in this repo is filtered extraction data plus disciplined evaluation.
- Summary tuning is intentionally out of scope for the default tutorial path.


 one of the side goals of this project is to have the following projects: So here are 4 skills + projects that will actually prepare you for that AI/ML role you’re eyeing 👀

1️⃣ 𝗙𝗶𝗻𝗲-𝗧𝘂𝗻𝗶𝗻𝗴 𝗢𝗽𝗲𝗻 𝗟𝗟𝗠𝘀 𝘄𝗶𝘁𝗵 𝗟𝗼𝗥𝗔/𝗤𝗟𝗼𝗥𝗔 🗂️
Project: Take an open model like Qwen, Gemma, or Llama and adapt it for a narrow task like legal clause extraction or domain-specific summarization. 

🔥 This shows you understand model adaptation, dataset preparation, task-specific tuning, and how to make open models useful for actual business workflows.

—————

2️⃣ 𝗠𝗼𝗱𝗲𝗹 𝗗𝗲𝗽𝗹𝗼𝘆𝗺𝗲𝗻𝘁 ☁️
Project: Deploy a self-hosted LLM API for inference.

Do not stop at “𝑰 𝒓𝒂𝒏 𝒕𝒉𝒆 𝒎𝒐𝒅𝒆𝒍 𝒍𝒐𝒄𝒂𝒍𝒍𝒚.”
Serve it properly, expose it through an API, add streaming, test concurrency, and make it usable inside a real application.

🔥 This shows you understand inference serving, APIs, deployment environment

A lot of people can build AI demos. Far fewer can version datasets, track experiments, test changes, evaluate outputs, monitor failures, and improve systems after deployment.

🔥 This project shows you understand 𝐫𝐞𝐥𝐢𝐚𝐛𝐢𝐥𝐢𝐭𝐲, 𝐨𝐛𝐬𝐞𝐫𝐯𝐚𝐛𝐢𝐥𝐢𝐭𝐲, 𝐚𝐧𝐝 𝐜𝐨𝐧𝐭𝐢𝐧𝐮𝐨𝐮𝐬 𝐢𝐦𝐩𝐫𝐨𝐯𝐞𝐦𝐞𝐧𝐭.

—————

4️⃣ 𝗖𝗼𝘀𝘁 𝗮𝗻𝗱 𝗟𝗮𝘁𝗲𝗻𝗰𝘆 𝗢𝗽𝘁𝗶𝗺𝗶𝘇𝗮𝘁𝗶𝗼𝗻 𝗳𝗼𝗿 𝗟𝗟𝗠 𝗦𝘆𝘀𝘁𝗲𝗺𝘀 ⚖️
Project: Benchmark and optimize an LLM stack for speed, quality, and cost.

𝘗𝘳𝘰𝘥𝘶𝘤𝘵𝘪𝘰𝘯 𝘈𝘐 𝘪𝘴 𝘯𝘰𝘵 𝘫𝘶𝘴𝘵 𝘢𝘣𝘰𝘶𝘵 𝘨𝘦𝘵𝘵𝘪𝘯𝘨 𝘵𝘩𝘦 𝘣𝘦𝘴𝘵 𝘰𝘶𝘵𝘱𝘶𝘵❌. Tradeoffs based on model speeds, performance, and how much each workflow costs at scale are equally important. 🔥 . and the plan was to lora or qlora finetune a model on contract and policy. where the weights fine tuned for contract are selectively different from the ones to be updated for policy. then the next step would be to somehow deploy the model for inference, whatevr and however that means. then do 𝗖𝗼𝘀𝘁 𝗮𝗻𝗱 𝗟𝗮𝘁𝗲𝗻𝗰𝘆 𝗢𝗽𝘁𝗶𝗺𝗶𝘇𝗮𝘁𝗶𝗼𝗻 𝗳𝗼𝗿 the llm system. basically knocking out 3 out of the 4 projects. make sense?