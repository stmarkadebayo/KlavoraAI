#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from klavora_ai.policy_data import (  # noqa: E402
    build_policy_dataset_examples,
    load_policy_normalized_documents,
    write_policy_processed_splits,
)
from klavora_ai.demo_profiles import POLICY_DEMO_DEFAULTS, build_demo_profile  # noqa: E402


def _positive_int_or_none(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    parsed = int(lowered)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer or 'none'")
    return parsed


def _resolve_policy_output_dir(profile: str, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir)
    if profile == "demo":
        return ROOT / "data" / "demo" / "policy"
    return ROOT / "data" / "processed" / "policy_main"


def _merge_profile_metadata(quality_report: dict[str, Any], profile: str, profile_report: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(quality_report)
    merged["profile"] = profile
    if profile_report:
        merged["demo_selection"] = profile_report
    return merged


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
        default=None,
    )
    parser.add_argument("--profile", choices=["standard", "demo"], default="standard")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-token-estimate", type=_positive_int_or_none, default=None)
    parser.add_argument(
        "--min-label-confidence",
        choices=["low", "medium", "high"],
        default="medium",
    )
    parser.add_argument("--min-populated-fields", type=int, default=2)
    parser.add_argument("--max-train-examples", type=_positive_int_or_none, default=None)
    parser.add_argument("--max-val-examples", type=_positive_int_or_none, default=None)
    parser.add_argument("--max-test-examples", type=_positive_int_or_none, default=None)
    args = parser.parse_args()

    normalized_docs = load_policy_normalized_documents(Path(args.opp_docs))
    examples, quality_report = build_policy_dataset_examples(
        normalized_docs,
        target_chunk_tokens=args.target_chunk_tokens,
        max_tokens=args.max_tokens,
    )
    output_dir = _resolve_policy_output_dir(args.profile, args.output_dir)

    profile_report = None
    if args.profile == "demo":
        profile_defaults = POLICY_DEMO_DEFAULTS
        examples, profile_report = build_demo_profile(
            examples=examples,
            domain="policy",
            seed=args.seed,
            base_quality_report=quality_report,
            max_token_estimate=args.max_token_estimate or profile_defaults["max_token_estimate"],
            min_label_confidence=args.min_label_confidence,
            min_populated_fields=args.min_populated_fields,
            max_train_examples=args.max_train_examples or profile_defaults["max_train_examples"],
            max_val_examples=args.max_val_examples or profile_defaults["max_val_examples"],
            max_test_examples=args.max_test_examples or profile_defaults["max_test_examples"],
        )

    final_quality_report = _merge_profile_metadata(quality_report, args.profile, profile_report)
    counts = write_policy_processed_splits(examples, output_dir, quality_report=final_quality_report)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "counts": counts,
                "quality_report_path": str(output_dir / "quality_report.json"),
                "profile": args.profile,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
