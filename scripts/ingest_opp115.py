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

from klavora_ai.io_utils import write_jsonl  # noqa: E402
from klavora_ai.policy_data import build_policy_documents  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize OPP-115 into policy documents.")
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "data" / "raw" / "opp-115"),
    )
    parser.add_argument(
        "--pretty-print-dir",
        default="",
        help="Optional override for the OPP pretty-print directory. Defaults to pretty_print, then pretty_print_uniquified.",
    )
    parser.add_argument(
        "--output-path",
        default=str(ROOT / "data" / "normalized" / "policy_opp115_documents.jsonl"),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.pretty_print_dir:
        pretty_print_dir = Path(args.pretty_print_dir)
    else:
        pretty_print_dir = input_dir / "pretty_print"
        if not pretty_print_dir.exists():
            pretty_print_dir = input_dir / "pretty_print_uniquified"
    documents = build_policy_documents(
        sanitized_dir=input_dir / "sanitized_policies",
        pretty_print_dir=pretty_print_dir,
        source_dataset="opp_115",
    )

    output_path = Path(args.output_path)
    count = write_jsonl(output_path, [doc.model_dump() for doc in documents])
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "documents": count,
                "input_dir": str(input_dir),
                "pretty_print_dir": str(pretty_print_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_path": str(output_path), "documents": count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
