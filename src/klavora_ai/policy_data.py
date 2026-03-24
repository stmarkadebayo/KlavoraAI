from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from klavora_ai.dataset_builder import make_dataset_example, write_split_jsonl
from klavora_ai.io_utils import estimate_token_count, read_jsonl, slugify_filename, summarize_numeric_series
from klavora_ai.prompts import render_extract_prompt
from klavora_ai.schemas import (
    DatasetExample,
    PolicyExtraction,
    PolicyNormalizedDocument,
    PolicyObligation,
    PolicySegment,
    SourceSpanReference,
)

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
    flags=re.I,
)
SEGMENT_SPLIT_RE = re.compile(r"\|\|\|")
CATEGORY_KEYWORDS = {
    "security": ["security", "fraud", "protect", "encrypt", "authentication"],
    "it": ["cookie", "device", "tracking", "ip address", "browser", "online activities"],
    "operations": ["retained", "policy change", "service operation", "legal requirement"],
}


def sanitize_policy_html(raw_html: str) -> str:
    text = raw_html.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _normalize_sentence(value: str) -> str:
    sentence = WHITESPACE_RE.sub(" ", value).strip().strip('"')
    return sentence


def _dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for item in items:
        value = _normalize_sentence(item)
        if value and value not in seen:
            seen.add(value)
            rows.append(value)
    return rows


def _extract_first_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    return match.group(0) if match else None


def _infer_policy_area(practice_summaries: list[str]) -> str:
    joined = "\n".join(practice_summaries).lower()
    for area, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in joined for keyword in keywords):
            return area
    return "other"


def build_weak_policy_target(raw_text: str, practice_summaries: list[str]) -> PolicyExtraction:
    lower_summaries = [summary.lower() for summary in practice_summaries]
    applies_to: list[str] = []
    if any("child" in summary for summary in lower_summaries):
        applies_to.append("children")
    if any("european" in summary or "europe" in summary for summary in lower_summaries):
        applies_to.append("regional users")
    if any("mobile app" in summary for summary in lower_summaries):
        applies_to.append("mobile app users")
    applies_to.append("website users")

    required_actions = [
        summary
        for summary in practice_summaries
        if summary.lower().startswith("you can")
        or summary.lower().startswith("users can")
        or summary.lower().startswith("a user can")
    ][:3]

    risk_flags = [
        summary
        for summary in practice_summaries
        if any(
            keyword in summary.lower()
            for keyword in [
                "third party",
                "tracking",
                "cookie",
                "retained",
                "policy change",
                "security",
                "advertising",
                "children",
                "location",
            ]
        )
    ][:5]

    key_obligations = [
        PolicyObligation(
            obligation=summary,
            owner=None,
            deadline=_extract_first_date(summary),
        )
        for summary in practice_summaries[:5]
    ]

    return PolicyExtraction(
        document_type="policy",
        policy_area=_infer_policy_area(practice_summaries),
        effective_date=_extract_first_date(raw_text),
        review_date=None,
        responsible_roles=[],
        applies_to=_dedupe_preserve(applies_to),
        key_obligations=key_obligations,
        exceptions=[],
        violations_or_consequences=[],
        required_actions=_dedupe_preserve(required_actions),
        risk_flags=_dedupe_preserve(risk_flags),
    )


def load_policy_normalized_documents(path: Path) -> list[PolicyNormalizedDocument]:
    return [PolicyNormalizedDocument.model_validate(row) for row in read_jsonl(path)]


def _split_ids(doc_ids: list[str]) -> dict[str, str]:
    ordered = sorted(doc_ids)
    total = len(ordered)
    if total <= 1:
        return {doc_id: "train" for doc_id in ordered}
    train_end = max(1, int(total * 0.8))
    val_end = min(total, max(train_end + 1, int(total * 0.9)))
    mapping: dict[str, str] = {}
    for doc_id in ordered[:train_end]:
        mapping[doc_id] = "train"
    for doc_id in ordered[train_end:val_end]:
        mapping[doc_id] = "val"
    for doc_id in ordered[val_end:]:
        mapping[doc_id] = "test"
    if "val" not in mapping.values() and ordered:
        mapping[ordered[-1]] = "val"
    return mapping


def _chunk_segments(segments: list[PolicySegment], target_tokens: int, max_tokens: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current: list[PolicySegment] = []
    current_tokens = 0

    def _flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n\n".join(segment.text for segment in current)
        summaries = _dedupe_preserve(summary for segment in current for summary in segment.practice_summaries)
        chunks.append(
            {
                "text": text,
                "segments": list(current),
                "practice_summaries": summaries,
                "token_estimate": estimate_token_count(text),
            }
        )
        current = []
        current_tokens = 0

    for segment in segments:
        segment_tokens = estimate_token_count(segment.text)
        if current and current_tokens + segment_tokens > target_tokens:
            _flush()
        if segment_tokens > max_tokens:
            _flush()
            chunks.append(
                {
                    "text": segment.text,
                    "segments": [segment],
                    "practice_summaries": segment.practice_summaries,
                    "token_estimate": segment_tokens,
                }
            )
            continue
        current.append(segment)
        current_tokens += segment_tokens
    _flush()
    return chunks


def _split_policy_chunk(chunk: dict[str, Any], max_tokens: int) -> list[dict[str, Any]]:
    segments = chunk.get("segments", [])
    if len(segments) > 1:
        midpoint = max(1, len(segments) // 2)
        grouped = [segments[:midpoint], segments[midpoint:]]
        results: list[dict[str, Any]] = []
        for group in grouped:
            if not group:
                continue
            text = "\n\n".join(segment.text for segment in group).strip()
            results.append(
                {
                    "text": text,
                    "segments": group,
                    "practice_summaries": _dedupe_preserve(summary for segment in group for summary in segment.practice_summaries),
                    "token_estimate": estimate_token_count(text),
                }
            )
        if len(results) > 1:
            return results

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"]) if sentence.strip()]
    if len(sentences) <= 1:
        midpoint = max(1, len(chunk["text"]) // 2)
        sentences = [chunk["text"][:midpoint].strip(), chunk["text"][midpoint:].strip()]
    results = []
    for piece in sentences:
        if not piece:
            continue
        results.append(
            {
                "text": piece,
                "segments": segments[:1],
                "practice_summaries": chunk.get("practice_summaries", [])[:3],
                "token_estimate": estimate_token_count(piece),
            }
        )
    return results or [chunk]


def _policy_field_coverage(target: PolicyExtraction) -> dict[str, int]:
    return {
        "effective_date": int(bool(target.effective_date)),
        "applies_to": int(bool(target.applies_to)),
        "key_obligations": int(bool(target.key_obligations)),
        "required_actions": int(bool(target.required_actions)),
        "risk_flags": int(bool(target.risk_flags)),
    }


def build_policy_dataset_examples(
    normalized_docs: list[PolicyNormalizedDocument],
    target_chunk_tokens: int = 800,
    max_tokens: int = 1024,
) -> tuple[list[DatasetExample], dict[str, Any]]:
    split_map = _split_ids([doc.doc_id for doc in normalized_docs])
    rows: list[DatasetExample] = []
    token_estimates_pre: list[int] = []
    token_estimates_post: list[int] = []
    field_coverage: dict[str, int] = defaultdict(int)
    truncation_risk_count = 0
    total_chunks = 0

    for doc in sorted(normalized_docs, key=lambda item: item.doc_id):
        segments = doc.segments or [PolicySegment(segment_id=0, text=doc.raw_text, practice_summaries=doc.practice_summaries)]
        pending_chunks = _chunk_segments(segments, target_tokens=target_chunk_tokens, max_tokens=max_tokens)
        finalized_chunks: list[dict[str, Any]] = []
        while pending_chunks:
            chunk = pending_chunks.pop(0)
            target = build_weak_policy_target(chunk["text"], chunk["practice_summaries"])
            token_estimate = estimate_token_count(chunk["text"]) + estimate_token_count(target.model_dump_json())
            if token_estimate > max_tokens and estimate_token_count(chunk["text"]) > max(120, max_tokens // 3):
                split_chunks = _split_policy_chunk(chunk, max_tokens=max_tokens)
                if len(split_chunks) > 1:
                    pending_chunks = split_chunks + pending_chunks
                    continue
            chunk["target"] = target
            chunk["token_estimate_total"] = token_estimate
            finalized_chunks.append(chunk)

        total_chunks += len(finalized_chunks)
        for index, chunk in enumerate(finalized_chunks):
            chunk_id = f"chunk_{index:03d}"
            target = chunk["target"]
            token_estimate = chunk["token_estimate_total"]
            token_estimates_pre.append(token_estimate)
            token_estimates_post.append(min(token_estimate, max_tokens))
            quality_flags = ["truncation_risk:high" if token_estimate > max_tokens else "truncation_risk:low"]
            quality_flags.append(
                "weak_label_confidence:high" if len(chunk["practice_summaries"]) >= 3 else "weak_label_confidence:medium"
            )
            if token_estimate > max_tokens:
                truncation_risk_count += 1

            coverage = _policy_field_coverage(target)
            for field_name, count in coverage.items():
                field_coverage[field_name] += count

            rows.append(
                make_dataset_example(
                    example_id=f"{doc.doc_id}__{chunk_id}__extract",
                    doc_id=f"{doc.doc_id}__{chunk_id}",
                    source_doc_id=doc.doc_id,
                    chunk_id=chunk_id,
                    split=split_map[doc.doc_id],
                    domain="policy",
                    task="extract",
                    user_content=render_extract_prompt("policy", chunk["text"]),
                    assistant_content=target.model_dump_json(indent=2),
                    token_count_estimate=token_estimate,
                    source_spans=[SourceSpanReference(segment_id=segment.segment_id) for segment in chunk["segments"]],
                    quality_flags=quality_flags,
                    metadata={
                        "title": doc.title,
                        "source_dataset": doc.source_dataset,
                        "practice_summary_count": len(chunk["practice_summaries"]),
                    },
                )
            )

    quality_report = {
        "document_count": len(normalized_docs),
        "chunk_count": total_chunks,
        "example_count": len(rows),
        "target_chunk_tokens": target_chunk_tokens,
        "max_tokens": max_tokens,
        "token_length_estimate_pre_truncation": summarize_numeric_series(token_estimates_pre),
        "token_length_estimate_post_truncation": summarize_numeric_series(token_estimates_post),
        "response_truncation_risk": {
            "count": truncation_risk_count,
            "rate": round(truncation_risk_count / len(rows), 4) if rows else 0.0,
        },
        "field_coverage": dict(sorted(field_coverage.items())),
    }
    return rows, quality_report


def write_policy_processed_splits(
    examples: list[DatasetExample],
    output_dir: Path,
    quality_report: Optional[dict[str, Any]] = None,
) -> dict[str, int]:
    counts = write_split_jsonl(examples, output_dir)
    if quality_report is not None:
        (output_dir / "quality_report.json").write_text(json.dumps(quality_report, indent=2, sort_keys=True), encoding="utf-8")
    return counts


def _load_pretty_print_rows(path: Path) -> dict[int, list[str]]:
    by_segment: dict[int, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            segment_id = int(row[1])
            by_segment[segment_id].append(row[3])
    return {segment_id: _dedupe_preserve(values) for segment_id, values in by_segment.items()}


def build_policy_documents(
    sanitized_dir: Path,
    pretty_print_dir: Path,
    source_dataset: str = "opp_115",
) -> list[PolicyNormalizedDocument]:
    documents: list[PolicyNormalizedDocument] = []

    for html_path in sorted(sanitized_dir.glob("*.html")):
        prefix, domain = html_path.stem.split("_", 1)
        pretty_name = domain.replace("www.", "")
        pretty_path = pretty_print_dir / f"{pretty_name}.csv"
        segment_summaries = _load_pretty_print_rows(pretty_path) if pretty_path.exists() else {}

        raw_source = html_path.read_text(encoding="utf-8", errors="ignore")
        raw_segments = [sanitize_policy_html(segment) for segment in SEGMENT_SPLIT_RE.split(raw_source)]
        segments: list[PolicySegment] = []
        for index, segment_text in enumerate(raw_segments):
            if not segment_text:
                continue
            segments.append(
                PolicySegment(
                    segment_id=index,
                    text=segment_text,
                    practice_summaries=segment_summaries.get(index, []),
                )
            )

        raw_text = "\n\n".join(segment.text for segment in segments)
        practice_summaries = _dedupe_preserve(summary for segment in segments for summary in segment.practice_summaries)
        weak_target = build_weak_policy_target(raw_text, practice_summaries)

        documents.append(
            PolicyNormalizedDocument(
                doc_id=slugify_filename(html_path.stem),
                source_dataset=source_dataset,
                title=html_path.stem,
                raw_text=raw_text,
                practice_summaries=practice_summaries,
                segments=segments,
                weak_extraction_target=weak_target.model_dump(),
                source_metadata={
                    "policy_prefix": prefix,
                    "pretty_print_path": str(pretty_path) if pretty_path.exists() else None,
                    "segment_count": len(segments),
                },
            )
        )
    return documents
