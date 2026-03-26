#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from klavora_ai.io_utils import summarize_numeric_series  # noqa: E402
from klavora_ai.training_config import UnslothTrainingConfig  # noqa: E402


def _load_config(path: Path) -> UnslothTrainingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return UnslothTrainingConfig.model_validate(raw)


def _count_jsonl_rows(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _supports_bf16() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())


def _load_quality_report(train_path: Path) -> dict[str, Any] | None:
    quality_path = train_path.parent / "quality_report.json"
    if not quality_path.exists():
        return None
    return json.loads(quality_path.read_text(encoding="utf-8"))


def _validate_dataset_shape(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    first_line = next((line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()), None)
    if first_line is None:
        raise ValueError(f"Dataset file is empty: {path}")
    record = json.loads(first_line)
    required_keys = {"example_id", "doc_id", "split", "domain", "task", "messages", "text"}
    missing = required_keys - set(record)
    if missing:
        raise ValueError(f"{path} is missing required keys: {sorted(missing)}")


def _normalize_messages_for_chat(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        normalized.append({"role": message["role"], "content": content})
    return normalized


def _save_config_snapshot(config: UnslothTrainingConfig, config_path: Path, output_dir: Path) -> None:
    snapshot = {"config_path": str(config_path), **config.model_dump()}
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False),
        encoding="utf-8",
    )


def _apply_lora(model: Any, config: UnslothTrainingConfig, fast_model_cls: Any) -> Any:
    if "gemma-3" in config.model_name.lower():
        return fast_model_cls.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            use_gradient_checkpointing=config.lora.gradient_checkpointing,
            random_state=config.seed,
        )

    return fast_model_cls.get_peft_model(
        model,
        r=config.lora.r,
        target_modules=config.lora.target_modules,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        use_gradient_checkpointing=config.lora.gradient_checkpointing,
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
    )


def _compute_token_stats(tokenizer: Any, dataset: Any) -> dict[str, float]:
    lengths = [len(tokenizer(example["text"], add_special_tokens=False)["input_ids"]) for example in dataset]
    return summarize_numeric_series(lengths)


def _limit_dataset(dataset: Any, max_rows: Optional[int]) -> Any:
    if max_rows is None or max_rows <= 0 or len(dataset) <= max_rows:
        return dataset
    return dataset.select(range(max_rows))


def _run_training(config: UnslothTrainingConfig, config_path: Path) -> None:
    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
        try:
            from unsloth import FastModel
        except ImportError:
            from unsloth import FastLanguageModel as FastModel
        from unsloth.chat_templates import get_chat_template, train_on_responses_only
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install Unsloth, datasets, TRL, transformers, and accelerate "
            "inside a supported Linux/WSL/Windows training environment."
        ) from exc

    train_path = Path(config.dataset.train_path)
    val_path = Path(config.dataset.val_path) if config.dataset.val_path else None

    data_files = {"train": str(train_path)}
    if val_path and val_path.exists():
        data_files["validation"] = str(val_path)
    dataset = load_dataset("json", data_files=data_files)

    model, tokenizer = FastModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
        full_finetuning=config.full_finetuning,
    )
    if config.chat_template:
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=config.chat_template,
            **config.chat_template_kwargs,
        )

    def _format_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts: list[str] = []
        for messages in batch["messages"]:
            rendered = tokenizer.apply_chat_template(
                _normalize_messages_for_chat(messages),
                tokenize=False,
                add_generation_prompt=False,
                **config.apply_chat_template_kwargs,
            )
            texts.append(rendered.removeprefix("<bos>"))
        return {"text": texts}

    formatted = dataset.map(_format_batch, batched=True)
    train_dataset = _limit_dataset(formatted["train"], config.max_train_samples)
    eval_dataset = _limit_dataset(formatted["validation"], config.max_eval_samples) if "validation" in formatted else None

    train_dataset = train_dataset.remove_columns([name for name in train_dataset.column_names if name != "text"])
    if eval_dataset is not None:
        eval_dataset = eval_dataset.remove_columns([name for name in eval_dataset.column_names if name != "text"])

    model = _apply_lora(model, config, FastModel)

    output_dir = Path(config.output_dir)
    adapter_output_dir = Path(config.adapter_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output_dir.mkdir(parents=True, exist_ok=True)
    _save_config_snapshot(config, config_path, output_dir)

    preflight_report: dict[str, Any] = {
        "run_name": config.run_name,
        "domain": config.domain,
        "tier": config.tier,
        "chat_template": config.chat_template,
        "response_only": config.response_only,
        "configured_max_seq_length": config.max_seq_length,
        "configured_trainer_max_length": config.trainer.max_length,
        "initial_train_rows": len(train_dataset),
        "initial_eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "train_token_stats": _compute_token_stats(tokenizer, train_dataset),
        "eval_token_stats": _compute_token_stats(tokenizer, eval_dataset) if eval_dataset is not None else None,
        "quality_report": _load_quality_report(train_path),
    }
    if preflight_report["initial_train_rows"] == 0:
        raise ValueError("Preflight failed: templated train dataset is empty.")
    if preflight_report["train_token_stats"]["p95"] > config.preflight_max_p95_tokens:
        raise ValueError(
            "Preflight failed: train token p95 exceeds threshold "
            f"({preflight_report['train_token_stats']['p95']} > {config.preflight_max_p95_tokens})."
        )

    bf16 = _supports_bf16()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            max_length=config.trainer.max_length,
            per_device_train_batch_size=config.trainer.per_device_train_batch_size,
            per_device_eval_batch_size=config.trainer.per_device_eval_batch_size,
            gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
            warmup_ratio=config.trainer.warmup_ratio,
            num_train_epochs=config.trainer.num_train_epochs,
            learning_rate=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay,
            logging_steps=config.trainer.logging_steps,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=config.trainer.eval_steps if eval_dataset is not None else None,
            save_steps=config.trainer.save_steps,
            max_steps=config.trainer.max_steps,
            lr_scheduler_type=config.trainer.lr_scheduler_type,
            optim=config.trainer.optim,
            packing=config.trainer.packing,
            fp16=not bf16,
            bf16=bf16,
            seed=config.seed,
            report_to="none",
        ),
    )

    if config.response_only:
        train_before = len(trainer.train_dataset)
        eval_before = len(trainer.eval_dataset) if trainer.eval_dataset is not None else 0
        trainer = train_on_responses_only(
            trainer,
            instruction_part=config.response_only_instruction_part,
            response_part=config.response_only_response_part,
        )
        train_after = len(trainer.train_dataset)
        eval_after = len(trainer.eval_dataset) if trainer.eval_dataset is not None else 0
        removed_fraction = (train_before - train_after) / train_before if train_before else 0.0
        preflight_report["response_only_filtering"] = {
            "train_before": train_before,
            "train_after": train_after,
            "train_removed": train_before - train_after,
            "train_removed_fraction": round(removed_fraction, 4),
            "eval_before": eval_before,
            "eval_after": eval_after,
            "eval_removed": eval_before - eval_after,
        }
        if train_after == 0:
            raise ValueError("Preflight failed: train_on_responses_only removed every training row.")
        if removed_fraction > config.preflight_max_removed_fraction:
            raise ValueError(
                "Preflight failed: response-only filtering removed too many rows "
                f"({removed_fraction:.4f} > {config.preflight_max_removed_fraction:.4f})."
            )

    (output_dir / "preflight_report.json").write_text(
        json.dumps(preflight_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trainer_stats = trainer.train()
    model.save_pretrained(str(adapter_output_dir))
    tokenizer.save_pretrained(str(adapter_output_dir))
    (output_dir / "train_result.json").write_text(
        json.dumps(trainer_stats.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a KlavoraAI adapter with Unsloth.")
    parser.add_argument("--config", required=True, help="Path to a training YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset paths without importing training dependencies.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    train_path = Path(config.dataset.train_path)
    val_path = Path(config.dataset.val_path) if config.dataset.val_path else None
    test_path = Path(config.dataset.test_path) if config.dataset.test_path else None

    _validate_dataset_shape(train_path)
    if val_path and val_path.exists():
        _validate_dataset_shape(val_path)
    if test_path and test_path.exists():
        _validate_dataset_shape(test_path)

    print(
        json.dumps(
            {
                "run_name": config.run_name,
                "domain": config.domain,
                "tier": config.tier,
                "model_name": config.model_name,
                "train_rows": _count_jsonl_rows(train_path),
                "val_rows": _count_jsonl_rows(val_path),
                "test_rows": _count_jsonl_rows(test_path),
                "output_dir": config.output_dir,
                "adapter_output_dir": config.adapter_output_dir,
                "quality_report_path": str(train_path.parent / "quality_report.json")
                if (train_path.parent / "quality_report.json").exists()
                else None,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )

    if args.dry_run:
        return 0

    _run_training(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
