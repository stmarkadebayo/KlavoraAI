#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from klavora_ai.contract_data import build_contract_documents_from_cuad_rows  # noqa: E402
from klavora_ai.io_utils import read_parquet_records, write_jsonl  # noqa: E402


def _load_full_text_lookup(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    rows = read_parquet_records(path)
    if not rows:
        return {}
    if not {"file_name", "text"}.issubset(rows[0].keys()):
        raise ValueError(f"Full text parquet must include file_name and text columns: {path}")
    return {row["file_name"]: row["text"] for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize CUAD clauses into contract documents.")
    parser.add_argument(
        "--clauses-path",
        default=str(ROOT / "data" / "raw" / "cuad" / "data" / "train-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--fulltext-path",
        default=str(ROOT / "data" / "raw" / "cuad_fulltext" / "data" / "train-00000-of-00001.parquet"),
        help="Optional parquet with file_name and full contract text.",
    )
    parser.add_argument(
        "--output-path",
        default=str(ROOT / "data" / "normalized" / "contract_cuad_documents.jsonl"),
    )
    args = parser.parse_args()

    clauses_path = Path(args.clauses_path)
    fulltext_path = Path(args.fulltext_path)
    output_path = Path(args.output_path)

    clause_rows = read_parquet_records(clauses_path)
    full_text_lookup = _load_full_text_lookup(fulltext_path if fulltext_path.exists() else None)
    documents = build_contract_documents_from_cuad_rows(
        clause_rows,
        full_text_by_file=full_text_lookup,
        source_dataset="cuad_clause_classification",
    )
    count = write_jsonl(output_path, [doc.model_dump() for doc in documents])

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "documents": count,
                "has_full_text": bool(full_text_lookup),
                "clauses_path": str(clauses_path),
                "fulltext_path": str(fulltext_path) if full_text_lookup else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "documents": count,
                "has_full_text": bool(full_text_lookup),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
