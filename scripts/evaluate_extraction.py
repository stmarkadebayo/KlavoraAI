#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from klavora_ai.schemas import ContractExtraction, PolicyExtraction  # noqa: E402

JSON_RE = re.compile(r"\{.*\}", flags=re.S)
CONTRACT_PRESENCE_FIELDS = [
    "renewal_terms",
    "payment_terms",
    "termination_terms",
    "liability_or_penalty_terms",
    "confidentiality_terms",
]
POLICY_PRESENCE_FIELDS = [
    "applies_to",
    "key_obligations",
    "required_actions",
    "risk_flags",
]
PROMOTION_THRESHOLDS = {
    "json_validity_rate": 0.95,
    "max_response_truncation_risk_rate": 0.05,
    "contract_type_accuracy": 0.80,
    "normalized_date_accuracy": 0.75,
    "manual_benchmark_max_hallucinated_fields": 1,
}
TUTORIAL_THRESHOLDS = {
    "json_validity_rate": 0.90,
    "schema_parse_success_rate": 0.85,
    "max_unsupported_field_hallucinations": 5,
}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _extract_json(raw_output: str) -> dict[str, Any] | None:
    match = JSON_RE.search(raw_output)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"(19|20)\d{2}-\d{2}-\d{2}", text):
        return text
    cleaned = re.sub(r"\s+", " ", text.replace(",", ", ")).strip()
    cleaned = re.sub(r",\s+", ", ", cleaned)
    for pattern in ("%B %d, %Y", "%B %d %Y"):
        try:
            return datetime.strptime(cleaned, pattern).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text.lower()


def _presence(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _precision_recall(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4)}


def _load_gold(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    rows = _read_jsonl(path)
    extract_rows = [row for row in rows if row.get("task") == "extract"]
    if not extract_rows:
        raise ValueError(f"No extract rows found in {path}")
    domain = extract_rows[0]["domain"]
    gold_by_id: dict[str, dict[str, Any]] = {}
    for row in extract_rows:
        gold_by_id[row["example_id"]] = {
            "record": row,
            "target": json.loads(row["messages"][-1]["content"]),
        }
    return domain, gold_by_id


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        raw_output = row.get("raw_output") or row.get("output") or ""
        parsed = row.get("parsed_output")
        if parsed is None and raw_output:
            parsed = _extract_json(raw_output)
        predictions[row["example_id"]] = {
            "raw_output": raw_output,
            "parsed_output": parsed,
        }
    return predictions


def _validate_prediction(domain: str, parsed: Any) -> tuple[bool, dict[str, Any] | None]:
    if not isinstance(parsed, dict):
        return False, None
    try:
        if domain == "contract":
            return True, ContractExtraction.model_validate(parsed).model_dump()
        return True, PolicyExtraction.model_validate(parsed).model_dump()
    except ValidationError:
        return False, None


def _evaluate_system(
    domain: str,
    gold_by_id: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    quality_report: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = len(gold_by_id)
    top_level_fields = set(ContractExtraction.model_fields) if domain == "contract" else set(PolicyExtraction.model_fields)
    presence_fields = CONTRACT_PRESENCE_FIELDS if domain == "contract" else POLICY_PRESENCE_FIELDS
    date_fields = ["effective_date", "termination_date"] if domain == "contract" else ["effective_date", "review_date"]
    type_field = "contract_type" if domain == "contract" else "policy_area"

    json_valid = 0
    schema_valid = 0
    type_correct = 0
    unknown_field_count = 0
    date_matches = 0
    date_total = 0
    presence_counts = {field: {"tp": 0, "fp": 0, "fn": 0} for field in presence_fields}
    per_example: list[dict[str, Any]] = []

    for example_id, gold in sorted(gold_by_id.items()):
        prediction = predictions.get(example_id, {})
        raw_output = prediction.get("raw_output", "")
        parsed = prediction.get("parsed_output")
        if isinstance(parsed, dict):
            json_valid += 1
            unknown_field_count += len(set(parsed.keys()) - top_level_fields)
        schema_ok, validated = _validate_prediction(domain, parsed)
        if schema_ok:
            schema_valid += 1
            if validated.get(type_field) == gold["target"].get(type_field):
                type_correct += 1
        else:
            validated = {}

        for field in date_fields:
            gold_value = gold["target"].get(field)
            if gold_value:
                date_total += 1
                if _normalize_date(validated.get(field)) == _normalize_date(gold_value):
                    date_matches += 1

        for field in presence_fields:
            gold_present = _presence(gold["target"].get(field))
            pred_present = _presence(validated.get(field))
            if gold_present and pred_present:
                presence_counts[field]["tp"] += 1
            elif pred_present and not gold_present:
                presence_counts[field]["fp"] += 1
            elif gold_present and not pred_present:
                presence_counts[field]["fn"] += 1

        per_example.append(
            {
                "example_id": example_id,
                "source_doc_id": gold["record"].get("source_doc_id"),
                "json_valid": isinstance(parsed, dict),
                "schema_valid": schema_ok,
                "gold_type_or_area": gold["target"].get(type_field),
                "predicted_type_or_area": validated.get(type_field),
                "unknown_top_level_fields": sorted(set(parsed.keys()) - top_level_fields) if isinstance(parsed, dict) else [],
                "raw_output": raw_output,
            }
        )

    metrics: dict[str, Any] = {
        "examples": total,
        "json_validity_rate": round(json_valid / total, 4),
        "schema_parse_success_rate": round(schema_valid / total, 4),
        f"{type_field}_accuracy": round(type_correct / total, 4),
        "normalized_date_accuracy": round(date_matches / date_total, 4) if date_total else 0.0,
        "unsupported_field_hallucination_count": unknown_field_count,
        "presence_metrics": {
            field: _precision_recall(values["tp"], values["fp"], values["fn"])
            for field, values in presence_counts.items()
        },
    }
    if quality_report:
        metrics["quality_report"] = quality_report
        metrics["response_truncation_risk_rate"] = quality_report.get("response_truncation_risk", {}).get("rate", 0.0)
    return metrics, per_example


def _promotion_gate(metrics: dict[str, Any], benchmark_predictions: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    checks = {
        "json_validity_rate": metrics["json_validity_rate"] >= PROMOTION_THRESHOLDS["json_validity_rate"],
        "response_truncation_risk_rate": metrics.get("response_truncation_risk_rate", 1.0)
        <= PROMOTION_THRESHOLDS["max_response_truncation_risk_rate"],
        "contract_type_accuracy": metrics.get("contract_type_accuracy", 0.0) >= PROMOTION_THRESHOLDS["contract_type_accuracy"],
        "normalized_date_accuracy": metrics["normalized_date_accuracy"] >= PROMOTION_THRESHOLDS["normalized_date_accuracy"],
    }
    hallucinated_fields = 0
    if benchmark_predictions:
        for prediction in benchmark_predictions.values():
            parsed = prediction.get("parsed_output")
            if isinstance(parsed, dict):
                hallucinated_fields += len(set(parsed.keys()) - set(ContractExtraction.model_fields))
    checks["manual_benchmark_hallucinations"] = hallucinated_fields <= PROMOTION_THRESHOLDS["manual_benchmark_max_hallucinated_fields"]
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "manual_benchmark_hallucinated_fields": hallucinated_fields,
    }


def _tutorial_acceptance(metrics: dict[str, Any], benchmark_predictions: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    top_level_fields = (
        set(ContractExtraction.model_fields)
        if "contract_type_accuracy" in metrics
        else set(PolicyExtraction.model_fields)
    )
    benchmark_hallucinated_fields = 0
    benchmark_examples_with_unknown_fields = 0
    if benchmark_predictions:
        for prediction in benchmark_predictions.values():
            parsed = prediction.get("parsed_output")
            if isinstance(parsed, dict):
                extra = len(set(parsed.keys()) - top_level_fields)
                benchmark_hallucinated_fields += extra
                if extra:
                    benchmark_examples_with_unknown_fields += 1

    checks = {
        "json_validity_rate": metrics["json_validity_rate"] >= TUTORIAL_THRESHOLDS["json_validity_rate"],
        "schema_parse_success_rate": metrics["schema_parse_success_rate"] >= TUTORIAL_THRESHOLDS["schema_parse_success_rate"],
        "unsupported_field_hallucination_count": metrics["unsupported_field_hallucination_count"]
        <= TUTORIAL_THRESHOLDS["max_unsupported_field_hallucinations"],
        "benchmark_not_obviously_broken": benchmark_examples_with_unknown_fields == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "benchmark_hallucinated_fields": benchmark_hallucinated_fields,
        "benchmark_examples_with_unknown_fields": benchmark_examples_with_unknown_fields,
    }


def _write_markdown_report(output_path: Path, systems: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Extraction Evaluation",
        "",
        "| System | JSON Validity | Schema Parse | Type/Area Accuracy | Date Accuracy | Unsupported Fields |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system_name, payload in systems.items():
        metrics = payload["metrics"]
        type_key = "contract_type_accuracy" if "contract_type_accuracy" in metrics else "policy_area_accuracy"
        lines.append(
            f"| {system_name} | {metrics['json_validity_rate']:.4f} | {metrics['schema_parse_success_rate']:.4f} | "
            f"{metrics.get(type_key, 0.0):.4f} | {metrics['normalized_date_accuracy']:.4f} | "
            f"{metrics['unsupported_field_hallucination_count']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate extraction predictions against a processed holdout split.")
    parser.add_argument("--gold", required=True, help="Processed test split JSONL with gold assistant outputs.")
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        help="System prediction spec in the form name=path/to/predictions.jsonl. Repeat for multiple systems.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for report.json, report.md, and per-example outputs.")
    parser.add_argument(
        "--benchmark-predictions",
        default="",
        help="Optional benchmark prediction spec in the form name=path for promotion-gate manual benchmark checks.",
    )
    parser.add_argument(
        "--quality-report",
        default="",
        help="Optional dataset quality_report.json to attach to system metrics.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    domain, gold_by_id = _load_gold(Path(args.gold))
    quality_report = json.loads(Path(args.quality_report).read_text(encoding="utf-8")) if args.quality_report else None

    systems_payload: dict[str, dict[str, Any]] = {}
    for item in args.system:
        name, raw_path = item.split("=", 1)
        predictions = _load_predictions(Path(raw_path))
        metrics, per_example = _evaluate_system(domain, gold_by_id, predictions, quality_report)
        systems_payload[name] = {"metrics": metrics, "per_example": per_example}

    benchmark_predictions = None
    if args.benchmark_predictions:
        _, benchmark_path = args.benchmark_predictions.split("=", 1)
        benchmark_predictions = _load_predictions(Path(benchmark_path))

    report = {
        "domain": domain,
        "gold_path": args.gold,
        "systems": {name: payload["metrics"] for name, payload in systems_payload.items()},
    }
    newest_system = next(reversed(systems_payload))
    report["tutorial_acceptance"] = _tutorial_acceptance(systems_payload[newest_system]["metrics"], benchmark_predictions)
    if domain == "contract":
        report["promotion_gate"] = _promotion_gate(systems_payload[newest_system]["metrics"], benchmark_predictions)

    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_report(output_dir / "report.md", systems_payload)
    for name, payload in systems_payload.items():
        (output_dir / f"{name}_per_example.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=True) for row in payload["per_example"]) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"output_dir": str(output_dir), "systems": list(systems_payload)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
