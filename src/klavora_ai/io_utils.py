from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    rows = list(records)
    ensure_parent(path)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return len(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_parquet_records(path: Path) -> list[dict]:
    dataframe = pd.read_parquet(path)
    return dataframe.to_dict(orient="records")


def slugify_filename(value: str) -> str:
    stem = Path(value).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    return stem.strip("_")


def estimate_token_count(text: str) -> int:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return 0
    return max(1, round(len(normalized) / 4))


def summarize_numeric_series(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p95": 0, "max": 0, "avg": 0}

    ordered = sorted(values)

    def _percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(0.50),
        "p95": _percentile(0.95),
        "max": ordered[-1],
        "avg": round(sum(ordered) / len(ordered), 2),
    }
