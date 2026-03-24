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

from klavora_ai.contract_data import (  # noqa: E402
    build_contract_dataset_examples,
    load_contract_normalized_documents,
    write_contract_processed_splits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build contract training examples from normalized sources.")
    parser.add_argument(
        "--cuad-docs",
        default=str(ROOT / "data" / "normalized" / "contract_cuad_documents.jsonl"),
    )
    parser.add_argument(
        "--target-chunk-tokens",
        type=int,
        default=800,
        help="Approximate token target for each contract chunk before prompt templating.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Hard token cap used by the chunk builder and risk reporting.",
    )
    parser.add_argument(
        "--legal-summaries",
        default=str(ROOT / "data" / "normalized" / "legal_summarization_pairs.jsonl"),
        help="Reserved for phase 2 summary tuning. Ignored by the extraction-first builder.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed" / "contract_main"),
    )
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Deprecated compatibility flag. Summary examples are not built in this phase.",
    )
    args = parser.parse_args()

    normalized_docs = load_contract_normalized_documents(Path(args.cuad_docs))

    examples, quality_report = build_contract_dataset_examples(
        normalized_docs,
        target_chunk_tokens=args.target_chunk_tokens,
        max_tokens=args.max_tokens,
    )
    output_dir = Path(args.output_dir)
    counts = write_contract_processed_splits(examples, output_dir, quality_report=quality_report)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "counts": counts,
                "quality_report_path": str(output_dir / "quality_report.json"),
                "phase": "extraction_only",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
