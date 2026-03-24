#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from klavora_ai.dataset_builder import (  # noqa: E402
    build_contract_examples,
    build_policy_examples,
    load_contract_seed_examples,
    load_policy_seed_examples,
    write_split_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build synthetic training splits for KlavoraAI.")
    parser.add_argument(
        "--domain",
        choices=["policy", "contract", "all"],
        default="all",
        help="Which domain to build.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed"),
        help="Destination root for split JSONL files.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.domain in {"policy", "all"}:
        policy_seed_path = ROOT / "data" / "synthetic" / "policy_seed.jsonl"
        policy_rows = build_policy_examples(load_policy_seed_examples(policy_seed_path))
        counts = write_split_jsonl(policy_rows, output_root / "policy")
        print(f"policy: {counts}")

    if args.domain in {"contract", "all"}:
        contract_seed_path = ROOT / "data" / "synthetic" / "contract_seed.jsonl"
        contract_rows = build_contract_examples(load_contract_seed_examples(contract_seed_path))
        counts = write_split_jsonl(contract_rows, output_root / "contract")
        print(f"contract: {counts}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

