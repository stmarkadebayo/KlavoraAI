from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import Field

from klavora_ai.schemas import StrictModel


class DatasetPaths(StrictModel):
    train_path: str
    val_path: Optional[str] = None
    test_path: Optional[str] = None


class LoraConfig(StrictModel):
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    gradient_checkpointing: Union[str, bool] = "unsloth"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainerConfig(StrictModel):
    max_length: int = 1024
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    logging_steps: int = 1
    eval_steps: int = 10
    save_steps: int = 25
    max_steps: int = -1
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    packing: bool = False


class UnslothTrainingConfig(StrictModel):
    run_name: str
    domain: Literal["policy", "contract"]
    tier: Literal["sanity", "main"] = "main"
    model_name: str
    dataset: DatasetPaths
    output_dir: str
    adapter_output_dir: str
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    seed: int = 3407
    chat_template: str = "gemma-3"
    response_only: bool = True
    preflight_max_removed_fraction: float = 0.10
    preflight_max_p95_tokens: int = 1125
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    lora: LoraConfig = Field(default_factory=LoraConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
