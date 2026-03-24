#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _coverage_score(record: dict) -> int:
    target = json.loads(record["messages"][-1]["content"])
    score = 0
    for value in target.values():
        if isinstance(value, list) and value:
            score += 1
        elif isinstance(value, str) and value.strip():
            score += 1
        elif isinstance(value, dict) and value:
            score += 1
    return score


def build_benchmark(rows: list[dict], limit: int) -> list[dict]:
    extract_rows = [row for row in rows if row.get("task") == "extract"]
    ordered = sorted(
        extract_rows,
        key=lambda row: (
            -_coverage_score(row),
            row.get("source_doc_id") or row["doc_id"],
            row["example_id"],
        ),
    )
    selected: list[dict] = []
    seen_docs: set[str] = set()
    for row in ordered:
        source_doc_id = row.get("source_doc_id") or row["doc_id"]
        if source_doc_id in seen_docs:
            continue
        seen_docs.add(source_doc_id)
        selected.append(
            {
                "benchmark_id": f"{row['domain']}_benchmark_{len(selected) + 1:02d}",
                "domain": row["domain"],
                "example_id": row["example_id"],
                "source_doc_id": source_doc_id,
                "doc_id": row["doc_id"],
                "user_prompt": row["messages"][1]["content"],
                "gold_output": row["messages"][2]["content"],
                "checklist": [
                    "valid_json",
                    "no_invented_facts",
                    "correct_type_or_area",
                    "no_cross_contaminated_fields",
                    "captures_obvious_grounded_clauses",
                ],
            }
        )
        if len(selected) >= limit:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fixed manual-review benchmark from a processed extract split.")
    parser.add_argument("--input", required=True, help="Processed extract split JSONL, usually a test split.")
    parser.add_argument("--output", required=True, help="Output benchmark JSONL path.")
    parser.add_argument("--limit", type=int, default=12, help="Maximum number of benchmark prompts to export.")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.input))
    benchmark = build_benchmark(rows, args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in benchmark) + ("\n" if benchmark else ""),
        encoding="utf-8",
    )
    print(json.dumps({"output_path": str(output_path), "examples": len(benchmark)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
