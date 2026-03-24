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

from klavora_ai.io_utils import read_jsonl, write_jsonl  # noqa: E402
from klavora_ai.schemas import LegalSummaryPair  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize legal_summarization pairs.")
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "data" / "raw" / "legal_summarization"),
    )
    parser.add_argument(
        "--output-path",
        default=str(ROOT / "data" / "normalized" / "legal_summarization_pairs.jsonl"),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    corpus = {row["_id"]: row for row in read_jsonl(input_dir / "corpus.jsonl")}
    queries = {row["_id"]: row for row in read_jsonl(input_dir / "queries.jsonl")}
    qrels = read_jsonl(input_dir / "qrels" / "test.jsonl")

    pairs: list[dict] = []
    for index, row in enumerate(qrels):
        corpus_row = corpus[row["corpus-id"]]
        query_row = queries[row["query-id"]]
        pair = LegalSummaryPair(
            example_id=f"legal_summarization_{index:06d}",
            doc_id=corpus_row["_id"],
            summary_id=query_row["_id"],
            document_text=corpus_row["text"],
            summary_text=query_row["text"],
            source_metadata={
                "score": row["score"],
                "title": corpus_row.get("title", ""),
            },
        )
        pairs.append(pair.model_dump())

    output_path = Path(args.output_path)
    count = write_jsonl(output_path, pairs)
    print(json.dumps({"output_path": str(output_path), "pairs": count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

