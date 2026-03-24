from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from klavora_ai.prompts import (
    SYSTEM_PROMPTS,
    render_extract_prompt,
    render_summary_prompt,
    render_text_from_messages,
)
from klavora_ai.schemas import (
    ContractSeedExample,
    DatasetExample,
    PolicySeedExample,
)


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_policy_seed_examples(path: Path) -> list[PolicySeedExample]:
    return [PolicySeedExample.model_validate(record) for record in _read_jsonl(path)]


def load_contract_seed_examples(path: Path) -> list[ContractSeedExample]:
    return [ContractSeedExample.model_validate(record) for record in _read_jsonl(path)]


def _render_json(target: object) -> str:
    return json.dumps(target, indent=2, ensure_ascii=True, sort_keys=True)


def _split_doc_ids(doc_ids: list[str]) -> dict[str, str]:
    ordered = sorted(doc_ids)
    total = len(ordered)
    if total == 0:
        return {}
    if total == 1:
        return {ordered[0]: "train"}
    if total == 2:
        return {ordered[0]: "train", ordered[1]: "val"}

    train_count = max(1, round(total * 0.7))
    val_count = max(1, round(total * 0.15))
    if train_count + val_count >= total:
        val_count = 1
        train_count = total - 2
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        train_count -= 1

    split_map: dict[str, str] = {}
    for doc_id in ordered[:train_count]:
        split_map[doc_id] = "train"
    for doc_id in ordered[train_count : train_count + val_count]:
        split_map[doc_id] = "val"
    for doc_id in ordered[train_count + val_count :]:
        split_map[doc_id] = "test"
    return split_map


def make_dataset_example(
    *,
    example_id: str,
    doc_id: str,
    split: str,
    domain: str,
    task: str,
    user_content: str,
    assistant_content: str,
    summary_type: Optional[str] = None,
    source_doc_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    token_count_estimate: Optional[int] = None,
    source_spans: Optional[list] = None,
    quality_flags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
) -> DatasetExample:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[(domain, task)]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return DatasetExample(
        example_id=example_id,
        doc_id=doc_id,
        split=split,
        domain=domain,
        task=task,
        summary_type=summary_type,
        messages=messages,
        text=render_text_from_messages(messages),
        target_format="json" if task == "extract" else "text",
        source_doc_id=source_doc_id,
        chunk_id=chunk_id,
        token_count_estimate=token_count_estimate,
        source_spans=source_spans or [],
        quality_flags=quality_flags or [],
        metadata=metadata or {},
    )


def build_policy_examples(seed_examples: Iterable[PolicySeedExample]) -> list[DatasetExample]:
    seed_examples = list(seed_examples)
    split_map = _split_doc_ids([example.doc_id for example in seed_examples])
    rows: list[DatasetExample] = []

    for example in seed_examples:
        split = split_map[example.doc_id]
        rows.append(
            make_dataset_example(
                example_id=f"{example.doc_id}__extract",
                doc_id=example.doc_id,
                split=split,
                domain="policy",
                task="extract",
                user_content=render_extract_prompt("policy", example.document_text),
                assistant_content=_render_json(example.extraction_target.model_dump()),
                metadata={"title": example.title, "source_dataset": example.source_dataset},
            )
        )
        rows.append(
            make_dataset_example(
                example_id=f"{example.doc_id}__employee_summary",
                doc_id=example.doc_id,
                split=split,
                domain="policy",
                task="summarize",
                summary_type="employee_summary",
                user_content=render_summary_prompt(
                    "policy",
                    "employee_summary",
                    example.document_text,
                ),
                assistant_content=example.summaries.employee_summary,
                metadata={"title": example.title, "source_dataset": example.source_dataset},
            )
        )
        rows.append(
            make_dataset_example(
                example_id=f"{example.doc_id}__ops_summary",
                doc_id=example.doc_id,
                split=split,
                domain="policy",
                task="summarize",
                summary_type="ops_summary",
                user_content=render_summary_prompt("policy", "ops_summary", example.document_text),
                assistant_content=example.summaries.ops_summary,
                metadata={"title": example.title, "source_dataset": example.source_dataset},
            )
        )
    return rows


def build_contract_examples(seed_examples: Iterable[ContractSeedExample]) -> list[DatasetExample]:
    seed_examples = list(seed_examples)
    split_map = _split_doc_ids([example.doc_id for example in seed_examples])
    rows: list[DatasetExample] = []

    for example in seed_examples:
        split = split_map[example.doc_id]
        rows.append(
            make_dataset_example(
                example_id=f"{example.doc_id}__extract",
                doc_id=example.doc_id,
                split=split,
                domain="contract",
                task="extract",
                user_content=render_extract_prompt("contract", example.document_text),
                assistant_content=_render_json(example.extraction_target.model_dump()),
                metadata={"title": example.title, "source_dataset": example.source_dataset},
            )
        )
        rows.append(
            make_dataset_example(
                example_id=f"{example.doc_id}__executive_summary",
                doc_id=example.doc_id,
                split=split,
                domain="contract",
                task="summarize",
                summary_type="executive_summary",
                user_content=render_summary_prompt(
                    "contract",
                    "executive_summary",
                    example.document_text,
                ),
                assistant_content=example.summaries.executive_summary,
                metadata={"title": example.title, "source_dataset": example.source_dataset},
            )
        )
        rows.append(
            make_dataset_example(
                example_id=f"{example.doc_id}__action_summary",
                doc_id=example.doc_id,
                split=split,
                domain="contract",
                task="summarize",
                summary_type="action_summary",
                user_content=render_summary_prompt("contract", "action_summary", example.document_text),
                assistant_content=example.summaries.action_summary,
                metadata={"title": example.title, "source_dataset": example.source_dataset},
            )
        )
    return rows


def write_split_jsonl(examples: Iterable[DatasetExample], output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = list(examples)
    counts = {"train": 0, "val": 0, "test": 0}

    for split in counts:
        path = output_dir / f"{split}.jsonl"
        split_examples = [example for example in examples if example.split == split]
        path.write_text(
            "\n".join(json.dumps(example.model_dump(), ensure_ascii=True) for example in split_examples)
            + ("\n" if split_examples else ""),
            encoding="utf-8",
        )
        counts[split] = len(split_examples)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8")
    return counts
