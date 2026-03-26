# Legacy Unsloth Colab: Gemma 3 (4B)

This document is archived for older runs only. The default tutorial path for this repo is now:

- `Qwen3 4B` primary
- `Llama 3.2 3B` fallback

Use the current runbooks instead of this file unless you are reproducing an older Gemma experiment.

This is the exact path to use for the first fine-tuning run.

## Official notebook

Use Unsloth's Gemma 3 (4B) text notebook:

- https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_%284B%29.ipynb

Current official references:

- Gemma 3 guide: https://unsloth.ai/docs/models/gemma-3-how-to-run-and-fine-tune
- Notebook index: https://unsloth.ai/docs/get-started/unsloth-notebooks

The docs currently state that:

- free Colab Tesla T4 runs are supported for Gemma 3 training
- the Gemma 3 (4B) notebook exists for text fine-tuning
- the recommended model load path in the notebook is `unsloth/gemma-3-4b-it` with `load_in_4bit = True`

## What to prepare on your Mac first

Run:

```bash
python3 scripts/build_seed_datasets.py
```

This creates:

- [policy train split](/Users/mac/Desktop/KlavoraAI/data/processed/policy/train.jsonl)
- [policy val split](/Users/mac/Desktop/KlavoraAI/data/processed/policy/val.jsonl)
- [contract train split](/Users/mac/Desktop/KlavoraAI/data/processed/contract/train.jsonl)
- [contract val split](/Users/mac/Desktop/KlavoraAI/data/processed/contract/val.jsonl)

For the first run, pick one domain only. Start with `contract`.

## Files to upload to Colab

Upload these files from this repo:

- `data/processed/contract/train.jsonl`
- `data/processed/contract/val.jsonl`

Optional:

- `data/processed/contract/test.jsonl`

## Notebook edits

Change the official notebook in these places.

### 1. Keep the install cell

Use the notebook's own install cell as-is.

### 2. Keep the model load cell mostly as-is

The official notebook currently loads:

```python
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)
```

### 3. Keep the Gemma chat template

```python
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
```

### 4. Replace the sample dataset cell with local JSONL loading

Use:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "/content/train.jsonl",
        "validation": "/content/val.jsonl",
    },
)

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
```

### 5. Skip the `standardize_data_formats` cell

Do not use the FineTome conversion cells. Your exported JSONL already contains the final `text` column.

### 6. Replace the formatting cell

Use:

```python
train_dataset = train_dataset.remove_columns(
    [c for c in train_dataset.column_names if c != "text"]
)
eval_dataset = eval_dataset.remove_columns(
    [c for c in eval_dataset.column_names if c != "text"]
)
```

### 7. Replace the LoRA cell with the text-only setup

Use:

```python
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers = False,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)
```

### 8. Replace the trainer cell

Use:

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        output_dir = "outputs",
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        logging_steps = 1,
        eval_strategy = "steps",
        eval_steps = 10,
        save_steps = 25,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        report_to = "none",
    ),
)
```

### 9. Save the adapter with a domain-specific name

Use:

```python
model.save_pretrained("contract_gemma3_4b_lora")
tokenizer.save_pretrained("contract_gemma3_4b_lora")
```

If training `policy`, rename the folder accordingly.

## Recommended first run

Start with:

- domain: `contract`
- model: `unsloth/gemma-3-4b-it`
- quantization: 4-bit
- LoRA rank: `8`
- epochs: `3`
- sequence length: `2048`

This is the lowest-friction path for a first successful run.
