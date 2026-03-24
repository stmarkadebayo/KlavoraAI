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

from klavora_ai.policy_data import (  # noqa: E402
    build_policy_dataset_examples,
    load_policy_normalized_documents,
    write_policy_processed_splits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build policy training examples from normalized OPP-115 docs.")
    parser.add_argument(
        "--opp-docs",
        default=str(ROOT / "data" / "normalized" / "policy_opp115_documents.jsonl"),
    )
    parser.add_argument(
        "--target-chunk-tokens",
        type=int,
        default=800,
        help="Approximate token target for each policy chunk before prompt templating.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Hard token cap used by the chunk builder and risk reporting.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed" / "policy_main"),
    )
    args = parser.parse_args()

    normalized_docs = load_policy_normalized_documents(Path(args.opp_docs))
    examples, quality_report = build_policy_dataset_examples(
        normalized_docs,
        target_chunk_tokens=args.target_chunk_tokens,
        max_tokens=args.max_tokens,
    )
    output_dir = Path(args.output_dir)
    counts = write_policy_processed_splits(examples, output_dir, quality_report=quality_report)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "counts": counts,
                "quality_report_path": str(output_dir / "quality_report.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
