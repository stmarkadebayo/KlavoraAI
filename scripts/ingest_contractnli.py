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

from klavora_ai.io_utils import read_parquet_records, write_jsonl  # noqa: E402
from klavora_ai.schemas import ContractNLIPair  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize ContractNLI splits.")
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "data" / "raw" / "contractnli" / "data"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "normalized"),
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {}
    for split in ["train", "validation", "test", "dev"]:
        rows = read_parquet_records(input_dir / f"{split}-00000-of-00001.parquet")
        normalized = [
            ContractNLIPair(
                example_id=f"contractnli_{split}_{index:06d}",
                premise=row["sentence1"],
                hypothesis=row["sentence2"],
                label=row["gold_label"],
                split=split,
                source_metadata={"numeric_label": row["label"]},
            ).model_dump()
            for index, row in enumerate(rows)
        ]
        output_path = output_dir / f"contractnli_{split}.jsonl"
        split_counts[split] = write_jsonl(output_path, normalized)

    print(json.dumps(split_counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

