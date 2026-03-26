from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Literal

from klavora_ai.io_utils import summarize_numeric_series
from klavora_ai.schemas import DatasetExample

ConfidenceLevel = Literal["low", "medium", "high"]

CONFIDENCE_ORDER: dict[ConfidenceLevel, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

CONTRACT_DEMO_DEFAULTS = {
    "max_train_examples": 600,
    "max_val_examples": 80,
    "max_test_examples": 80,
    "min_populated_fields": 2,
    "max_token_estimate": 1024,
    "min_label_confidence": "medium",
}

POLICY_DEMO_DEFAULTS = {
    "max_train_examples": 500,
    "max_val_examples": 64,
    "max_test_examples": 64,
    "min_populated_fields": 2,
    "max_token_estimate": 1024,
    "min_label_confidence": "medium",
}

CONTRACT_POPULATED_FIELDS = [
    "parties",
    "effective_date",
    "renewal_terms",
    "payment_terms",
    "termination_terms",
    "confidentiality_terms",
    "key_obligations",
]

POLICY_POPULATED_FIELDS = [
    "effective_date",
    "applies_to",
    "required_actions",
    "risk_flags",
    "key_obligations",
]


def _presence(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _parse_confidence(example: DatasetExample) -> ConfidenceLevel:
    for flag in example.quality_flags:
        if flag.startswith("weak_label_confidence:"):
            value = flag.split(":", 1)[1]
            if value in CONFIDENCE_ORDER:
                return value  # type: ignore[return-value]
    return "low"


def _has_low_truncation_risk(example: DatasetExample) -> bool:
    return "truncation_risk:low" in example.quality_flags


def _extract_target(example: DatasetExample) -> dict[str, Any]:
    return json.loads(example.messages[-1]["content"])


def _count_populated_fields(target: dict[str, Any], fields: list[str]) -> int:
    return sum(1 for field in fields if _presence(target.get(field)))


def _field_coverage(rows: list[DatasetExample], fields: list[str]) -> dict[str, int]:
    coverage = {field: 0 for field in fields}
    for example in rows:
        target = _extract_target(example)
        for field in fields:
            coverage[field] += int(_presence(target.get(field)))
    return coverage


def _seed_key(seed: int, example_id: str) -> str:
    return hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).hexdigest()


def _rank_example(example: DatasetExample, populated_fields: list[str], seed: int) -> tuple[Any, ...]:
    target = _extract_target(example)
    confidence = _parse_confidence(example)
    populated = _count_populated_fields(target, populated_fields)
    token_estimate = example.token_count_estimate or 0
    return (
        -CONFIDENCE_ORDER[confidence],
        -populated,
        token_estimate,
        _seed_key(seed, example.example_id),
        example.example_id,
    )


def _apply_filter(
    rows: list[DatasetExample],
    report: dict[str, Any],
    key: str,
    predicate,
) -> list[DatasetExample]:
    kept = [row for row in rows if predicate(row)]
    report["filter_steps"][key] = {
        "before": len(rows),
        "after": len(kept),
        "dropped": len(rows) - len(kept),
    }
    return kept


def build_demo_profile(
    *,
    examples: list[DatasetExample],
    domain: Literal["contract", "policy"],
    seed: int,
    max_token_estimate: int,
    min_label_confidence: ConfidenceLevel,
    min_populated_fields: int,
    max_train_examples: int,
    max_val_examples: int,
    max_test_examples: int,
    base_quality_report: dict[str, Any] | None = None,
) -> tuple[list[DatasetExample], dict[str, Any]]:
    populated_fields = CONTRACT_POPULATED_FIELDS if domain == "contract" else POLICY_POPULATED_FIELDS
    report: dict[str, Any] = {
        "profile": "demo",
        "domain": domain,
        "selection_seed": seed,
        "base_quality_report": base_quality_report,
        "filter_settings": {
            "max_token_estimate": max_token_estimate,
            "min_label_confidence": min_label_confidence,
            "min_populated_fields": min_populated_fields,
            "max_train_examples": max_train_examples,
            "max_val_examples": max_val_examples,
            "max_test_examples": max_test_examples,
        },
        "total_candidates_before_filter": len(examples),
        "filter_steps": {},
    }

    rows = _apply_filter(examples, report, "task_extract_only", lambda row: row.task == "extract")
    rows = _apply_filter(rows, report, "truncation_risk_low", _has_low_truncation_risk)
    rows = _apply_filter(
        rows,
        report,
        "label_confidence_threshold",
        lambda row: CONFIDENCE_ORDER[_parse_confidence(row)] >= CONFIDENCE_ORDER[min_label_confidence],
    )
    rows = _apply_filter(
        rows,
        report,
        "max_token_estimate",
        lambda row: (row.token_count_estimate or 0) <= max_token_estimate,
    )
    rows = _apply_filter(
        rows,
        report,
        "min_populated_fields",
        lambda row: _count_populated_fields(_extract_target(row), populated_fields) >= min_populated_fields,
    )

    capped_rows: list[DatasetExample] = []
    split_caps = {
        "train": max_train_examples,
        "val": max_val_examples,
        "test": max_test_examples,
    }
    split_counts_before = Counter(row.split for row in rows)
    for split_name in ["train", "val", "test"]:
        split_rows = [row for row in rows if row.split == split_name]
        split_rows = sorted(split_rows, key=lambda row: _rank_example(row, populated_fields, seed))
        capped_rows.extend(split_rows[: split_caps[split_name]])

    label_distribution = Counter(_parse_confidence(row) for row in capped_rows)
    token_values = [row.token_count_estimate or 0 for row in capped_rows]
    report.update(
        {
            "kept_after_filtering": len(rows),
            "split_counts_before_caps": dict(split_counts_before),
            "split_counts_after_caps": dict(Counter(row.split for row in capped_rows)),
            "label_confidence_distribution": {
                "low": label_distribution.get("low", 0),
                "medium": label_distribution.get("medium", 0),
                "high": label_distribution.get("high", 0),
            },
            "token_length_estimate": summarize_numeric_series(token_values),
            "field_coverage": _field_coverage(capped_rows, populated_fields),
        }
    )
    return capped_rows, report
