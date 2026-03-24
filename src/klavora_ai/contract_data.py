from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from klavora_ai.dataset_builder import make_dataset_example, write_split_jsonl
from klavora_ai.io_utils import estimate_token_count, read_jsonl, slugify_filename, summarize_numeric_series
from klavora_ai.prompts import render_extract_prompt
from klavora_ai.schemas import (
    ClauseSpan,
    ContractExtraction,
    ContractNLIPair,
    ContractNormalizedDocument,
    ContractObligation,
    DatasetExample,
    LegalSummaryPair,
    SourceSpanReference,
)

DATE_PATTERNS = [
    re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),
    re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
        flags=re.I,
    ),
]
QUOTE_RE = re.compile(r"^[\"']+|[\"']+$")
PARTY_KEY_RE = re.compile(r"[^a-z0-9]+")
SECTION_BREAK_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

LABEL_GROUPS = {
    "effective_date": {"Agreement Date", "Effective Date"},
    "termination_date": {"Expiration Date"},
    "renewal_terms": {"Renewal Term", "Notice Period To Terminate Renewal"},
    "payment_terms": {"Revenue/Profit Sharing", "Price Restrictions"},
    "termination_terms": {"Termination For Convenience", "Post-Termination Services"},
    "liability_or_penalty_terms": {"Cap On Liability", "Uncapped Liability", "Liquidated Damages", "Covenant Not To Sue"},
    "key_obligations": {"Audit Rights", "Insurance", "Minimum Commitment", "Volume Restriction", "Warranty Duration"},
    "risk_flags": {"Most Favored Nation", "Anti-Assignment", "Non-Compete", "Exclusivity", "Change Of Control", "Rofr/Rofo/Rofn"},
}
MAX_FIELD_TEXT_CHARS = 240


def _dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for item in items:
        value = " ".join(str(item).split()).strip()
        if value and value not in seen:
            seen.add(value)
            rows.append(value)
    return rows


def _clip_text(value: str, max_chars: int = MAX_FIELD_TEXT_CHARS) -> str:
    normalized = " ".join(str(value).split()).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _normalize_party_name(value: str) -> str:
    cleaned = QUOTE_RE.sub("", value.strip())
    cleaned = cleaned.strip("()[]{} ,;:.")
    cleaned = " ".join(cleaned.split())
    return cleaned


def _party_key(value: str) -> str:
    return PARTY_KEY_RE.sub("", value.lower())


def _collapse_party_names(items: Iterable[str]) -> list[str]:
    chosen: dict[str, str] = {}
    for item in items:
        cleaned = _normalize_party_name(item)
        if not cleaned:
            continue
        key = _party_key(cleaned)
        if not key:
            continue
        current = chosen.get(key)
        if current is None or len(cleaned) > len(current):
            chosen[key] = cleaned
    return list(chosen.values())


def infer_contract_type(title: str, raw_text: str) -> str:
    combined = f"{title}\n{raw_text[:2000]}".lower()
    if "master services agreement" in combined or re.search(r"\bmsa\b", combined):
        return "msa"
    if "statement of work" in combined or re.search(r"\bsow\b", combined):
        return "sow"
    if "non-disclosure agreement" in combined or re.search(r"\bnda\b", combined):
        return "nda"
    if "employment agreement" in combined or "offer letter" in combined:
        return "employment"
    if "vendor" in combined or "supplier" in combined or "procurement" in combined:
        return "vendor"
    if "lease" in combined:
        return "lease"
    if "partnership" in combined or "joint venture" in combined:
        return "partnership"
    return "other"


def _extract_first_date(text: str) -> str | None:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _extract_dates(text: str) -> list[str]:
    results: list[str] = []
    for pattern in DATE_PATTERNS:
        results.extend(match.group(0) for match in pattern.finditer(text))
    return _dedupe_preserve(results)


def _extract_parties_from_text(text: str) -> list[str]:
    quoted = re.findall(r'"([^"]+)"', text)
    uppercase_lines = []
    for line in text.splitlines():
        normalized = line.strip()
        if normalized and normalized.upper() == normalized and any(char.isalpha() for char in normalized):
            uppercase_lines.append(normalized)
    candidates = quoted + uppercase_lines
    if " and " in text.lower():
        candidates.extend(part.strip() for part in re.split(r"\band\b", text, flags=re.I))
    return _collapse_party_names(candidates)


def _find_confidentiality_terms(raw_text: str, clause_spans: list[ClauseSpan]) -> list[str]:
    clauses = [_clip_text(span.text) for span in clause_spans if "confidential" in span.text.lower()]
    if not clauses:
        sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(raw_text) if part.strip()]
        clauses = [_clip_text(sentence) for sentence in sentences if "confidential" in sentence.lower()]
    return _dedupe_preserve(clauses)[:3]


def _find_payment_terms(raw_text: str, spans: list[ClauseSpan], existing: list[str]) -> list[str]:
    if existing:
        return existing
    candidates: list[str] = []
    for span in spans:
        lowered = span.text.lower()
        if any(keyword in lowered for keyword in ["pay ", "fee", "invoice", "royalty", "compensation", "payment"]):
            candidates.append(_clip_text(span.text))
    if not candidates:
        sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(raw_text) if part.strip()]
        candidates = [
            _clip_text(sentence)
            for sentence in sentences
            if any(keyword in sentence.lower() for keyword in ["pay ", "fee", "invoice", "royalty", "payment"])
        ]
    return _dedupe_preserve(candidates)[:3]


def _find_termination_terms(raw_text: str, spans: list[ClauseSpan], existing: list[str]) -> list[str]:
    if existing:
        return existing
    candidates: list[str] = []
    for span in spans:
        lowered = span.text.lower()
        if "terminat" in lowered or "expire" in lowered:
            candidates.append(_clip_text(span.text))
    if not candidates:
        sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(raw_text) if part.strip()]
        candidates = [_clip_text(sentence) for sentence in sentences if "terminat" in sentence.lower() or "expire" in sentence.lower()]
    return _dedupe_preserve(candidates)[:3]


def _find_renewal_terms(raw_text: str, spans: list[ClauseSpan], existing: list[str]) -> list[str]:
    if existing:
        return existing
    candidates: list[str] = []
    for span in spans:
        lowered = span.text.lower()
        if "renew" in lowered:
            candidates.append(_clip_text(span.text))
    if not candidates:
        sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(raw_text) if part.strip()]
        candidates = [_clip_text(sentence) for sentence in sentences if "renew" in sentence.lower()]
    return _dedupe_preserve(candidates)[:3]


def _build_obligations(spans: list[ClauseSpan], parties: list[str], raw_text: str) -> list[ContractObligation]:
    obligations: list[ContractObligation] = []
    lowered_parties = [(party, party.lower()) for party in parties]
    for span in spans:
        party = "unspecified"
        lowered_text = span.text.lower()
        for candidate, lowered_candidate in lowered_parties:
            if lowered_candidate and lowered_candidate in lowered_text:
                party = candidate
                break
        obligations.append(
            ContractObligation(
                party=party,
                obligation=_clip_text(span.text),
                deadline=_extract_first_date(span.text),
            )
        )
    if obligations:
        return obligations[:5]

    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(raw_text) if part.strip()]
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in ["shall", "must", "required to"]):
            obligations.append(
                ContractObligation(
                    party="unspecified",
                    obligation=_clip_text(sentence),
                    deadline=_extract_first_date(sentence),
                )
            )
    return obligations[:5]


def build_weak_contract_target(title: str, raw_text: str, clause_spans: list[ClauseSpan]) -> ContractExtraction:
    field_values: dict[str, list[str]] = defaultdict(list)
    key_obligation_spans: list[ClauseSpan] = []

    for span in clause_spans:
        for field_name, labels in LABEL_GROUPS.items():
            if span.label in labels:
                if field_name == "key_obligations":
                    key_obligation_spans.append(span)
                else:
                    field_values[field_name].append(_clip_text(span.text))
                break
        if span.label == "Parties":
            field_values["parties"].extend(_extract_parties_from_text(span.text))
        elif span.label == "Document Name":
            field_values["document_name"].append(_clip_text(span.text))

    parties = _collapse_party_names(field_values["parties"])
    if not parties:
        parties = _collapse_party_names(_extract_parties_from_text(raw_text[:2000]))

    effective_date_candidates = _dedupe_preserve(
        [date for text in field_values["effective_date"] for date in _extract_dates(text)]
    )
    effective_date = next(iter(effective_date_candidates), None) or _extract_first_date(raw_text)
    termination_date_candidates = _dedupe_preserve(
        [date for text in field_values["termination_date"] for date in _extract_dates(text)]
    )
    termination_date = next(iter(termination_date_candidates), None)

    renewal_terms = _find_renewal_terms(raw_text, clause_spans, _dedupe_preserve(field_values["renewal_terms"]))
    payment_terms = _find_payment_terms(raw_text, clause_spans, _dedupe_preserve(field_values["payment_terms"]))
    termination_terms = _find_termination_terms(raw_text, clause_spans, _dedupe_preserve(field_values["termination_terms"]))
    liability_terms = _dedupe_preserve(field_values["liability_or_penalty_terms"])[:3]
    confidentiality_terms = _find_confidentiality_terms(raw_text, clause_spans)
    risk_flags = _dedupe_preserve(field_values["risk_flags"])[:5]
    action_items = _dedupe_preserve(field_values["action_items"])[:5]
    key_obligations = _build_obligations(key_obligation_spans, parties, raw_text)

    return ContractExtraction(
        contract_type=infer_contract_type(title, raw_text),
        parties=parties,
        effective_date=effective_date,
        termination_date=termination_date,
        renewal_terms=renewal_terms,
        payment_terms=payment_terms,
        key_obligations=key_obligations,
        liability_or_penalty_terms=liability_terms,
        termination_terms=termination_terms,
        confidentiality_terms=confidentiality_terms,
        risk_flags=risk_flags,
        action_items=action_items,
    )


def _split_long_block(text: str, max_tokens: int) -> list[str]:
    if estimate_token_count(text) <= max_tokens:
        return [text]
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_token_count(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _split_text_blocks(raw_text: str, max_tokens: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    cursor = 0
    for match in SECTION_BREAK_RE.finditer(raw_text):
        text = raw_text[cursor : match.start()].strip()
        if text:
            for piece in _split_long_block(text, max_tokens):
                start_at = raw_text.find(piece, cursor)
                end_at = start_at + len(piece)
                blocks.append({"text": piece, "start_at": start_at, "end_at": end_at})
                cursor = end_at
        cursor = match.end()
    trailing = raw_text[cursor:].strip()
    if trailing:
        for piece in _split_long_block(trailing, max_tokens):
            start_at = raw_text.find(piece, cursor)
            end_at = start_at + len(piece)
            blocks.append({"text": piece, "start_at": start_at, "end_at": end_at})
            cursor = end_at
    return blocks


def _chunk_contract_blocks(blocks: list[dict[str, Any]], target_tokens: int, max_tokens: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current_blocks: list[dict[str, Any]] = []
    current_tokens = 0

    def _flush() -> None:
        nonlocal current_blocks, current_tokens
        if not current_blocks:
            return
        chunk_text = "\n\n".join(block["text"] for block in current_blocks).strip()
        chunks.append(
            {
                "text": chunk_text,
                "start_at": current_blocks[0].get("start_at"),
                "end_at": current_blocks[-1].get("end_at"),
                "token_estimate": estimate_token_count(chunk_text),
            }
        )
        current_blocks = []
        current_tokens = 0

    for block in blocks:
        block_tokens = estimate_token_count(block["text"])
        if current_blocks and current_tokens + block_tokens > target_tokens:
            _flush()
        if block_tokens > max_tokens:
            for piece in _split_long_block(block["text"], max_tokens):
                piece_tokens = estimate_token_count(piece)
                if current_blocks and current_tokens + piece_tokens > target_tokens:
                    _flush()
                current_blocks.append(
                    {
                        "text": piece,
                        "start_at": block.get("start_at"),
                        "end_at": block.get("end_at"),
                    }
                )
                current_tokens += piece_tokens
                if current_tokens >= target_tokens:
                    _flush()
        else:
            current_blocks.append(block)
            current_tokens += block_tokens
    _flush()
    return chunks


def _build_clause_overlap(chunk: dict[str, Any], clause_spans: list[ClauseSpan], raw_text_source: str) -> list[ClauseSpan]:
    if raw_text_source == "full_text":
        start_at = chunk.get("start_at")
        end_at = chunk.get("end_at")
        overlaps = [
            span
            for span in clause_spans
            if span.start_at is not None
            and span.end_at is not None
            and start_at is not None
            and end_at is not None
            and span.start_at < end_at
            and span.end_at > start_at
        ]
        return overlaps

    chunk_text = chunk["text"]
    overlaps = [span for span in clause_spans if span.text in chunk_text]
    if overlaps:
        return overlaps
    return clause_spans[:3]


def _split_contract_chunk(chunk: dict[str, Any], raw_text_source: str, max_tokens: int) -> list[dict[str, Any]]:
    if raw_text_source != "full_text" and chunk.get("clause_spans"):
        spans = chunk["clause_spans"]
        if len(spans) > 1:
            midpoint = max(1, len(spans) // 2)
            split_spans = [spans[:midpoint], spans[midpoint:]]
            results: list[dict[str, Any]] = []
            for group in split_spans:
                if not group:
                    continue
                text = "\n\n".join(span.text for span in group).strip()
                results.append(
                    {
                        "text": text,
                        "start_at": group[0].start_at,
                        "end_at": group[-1].end_at,
                        "clause_spans": group,
                    }
                )
            if len(results) > 1:
                return results

    pieces = _split_long_block(chunk["text"], max(200, max_tokens // 2))
    if len(pieces) <= 1:
        midpoint = max(1, len(chunk["text"]) // 2)
        pieces = [chunk["text"][:midpoint].strip(), chunk["text"][midpoint:].strip()]
    results = []
    for piece in pieces:
        if not piece:
            continue
        results.append(
            {
                "text": piece,
                "start_at": None,
                "end_at": None,
                "clause_spans": _build_clause_overlap(
                    {"text": piece, "start_at": None, "end_at": None},
                    chunk.get("clause_spans", []),
                    raw_text_source,
                ),
            }
        )
    return results or [chunk]


def build_contract_documents_from_cuad_rows(
    clause_rows: list[dict],
    full_text_by_file: dict[str, str] | None = None,
    source_dataset: str = "cuad",
) -> list[ContractNormalizedDocument]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in clause_rows:
        grouped[row["file_name"]].append(row)

    documents: list[ContractNormalizedDocument] = []
    for file_name, rows in sorted(grouped.items()):
        ordered_rows = sorted(rows, key=lambda item: (item.get("start_at") or 0, item.get("end_at") or 0))
        clause_spans = [
            ClauseSpan(
                label=row["label"],
                text=row["clause"],
                start_at=row.get("start_at"),
                end_at=row.get("end_at"),
                pages=row.get("pages"),
            )
            for row in ordered_rows
        ]
        fallback_text = "\n\n".join(_dedupe_preserve(span.text for span in clause_spans))
        raw_text = (full_text_by_file or {}).get(file_name) or fallback_text
        raw_text_source = "full_text" if (full_text_by_file or {}).get(file_name) else "joined_clauses"
        title = Path(file_name).stem
        weak_target = build_weak_contract_target(title, raw_text, clause_spans)
        documents.append(
            ContractNormalizedDocument(
                doc_id=slugify_filename(file_name),
                source_dataset=source_dataset,
                title=title,
                raw_text=raw_text,
                clause_spans=clause_spans,
                weak_extraction_target=weak_target.model_dump(),
                source_metadata={
                    "file_name": file_name,
                    "num_clause_spans": len(clause_spans),
                    "raw_text_source": raw_text_source,
                },
            )
        )
    return documents


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


def load_contract_normalized_documents(path: Path) -> list[ContractNormalizedDocument]:
    return [ContractNormalizedDocument.model_validate(row) for row in read_jsonl(path)]


def load_contract_nli_pairs(path: Path) -> list[ContractNLIPair]:
    return [ContractNLIPair.model_validate(row) for row in read_jsonl(path)]


def load_legal_summary_pairs(path: Path) -> list[LegalSummaryPair]:
    return [LegalSummaryPair.model_validate(row) for row in read_jsonl(path)]


def _build_contract_chunks(
    doc: ContractNormalizedDocument,
    target_tokens: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    raw_text_source = doc.source_metadata.get("raw_text_source", "joined_clauses")
    if raw_text_source == "full_text":
        blocks = _split_text_blocks(doc.raw_text, max_tokens=max_tokens)
    else:
        blocks = [{"text": span.text, "start_at": span.start_at, "end_at": span.end_at} for span in doc.clause_spans]
    chunks = _chunk_contract_blocks(blocks, target_tokens=target_tokens, max_tokens=max_tokens)

    for index, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"chunk_{index:03d}"
        overlaps = _build_clause_overlap(chunk, doc.clause_spans, raw_text_source)
        chunk["clause_spans"] = overlaps
    return chunks


def _json_field_coverage(target: ContractExtraction) -> dict[str, int]:
    return {
        "contract_type": int(bool(target.contract_type)),
        "parties": int(bool(target.parties)),
        "effective_date": int(bool(target.effective_date)),
        "termination_date": int(bool(target.termination_date)),
        "renewal_terms": int(bool(target.renewal_terms)),
        "payment_terms": int(bool(target.payment_terms)),
        "key_obligations": int(bool(target.key_obligations)),
        "liability_or_penalty_terms": int(bool(target.liability_or_penalty_terms)),
        "termination_terms": int(bool(target.termination_terms)),
        "confidentiality_terms": int(bool(target.confidentiality_terms)),
        "risk_flags": int(bool(target.risk_flags)),
        "action_items": int(bool(target.action_items)),
    }


def build_contract_dataset_examples(
    normalized_docs: list[ContractNormalizedDocument],
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

    for doc in normalized_docs:
        raw_text_source = doc.source_metadata.get("raw_text_source", "joined_clauses")
        pending_chunks = _build_contract_chunks(doc, target_tokens=target_chunk_tokens, max_tokens=max_tokens)
        finalized_chunks: list[dict[str, Any]] = []
        while pending_chunks:
            chunk = pending_chunks.pop(0)
            chunk_target = build_weak_contract_target(doc.title, chunk["text"], chunk["clause_spans"])
            token_estimate = estimate_token_count(chunk["text"]) + estimate_token_count(chunk_target.model_dump_json())
            if token_estimate > max_tokens and estimate_token_count(chunk["text"]) > max(120, max_tokens // 3):
                split_chunks = _split_contract_chunk(chunk, raw_text_source, max_tokens=max_tokens)
                if len(split_chunks) > 1:
                    pending_chunks = split_chunks + pending_chunks
                    continue
            chunk["chunk_target"] = chunk_target
            chunk["token_estimate"] = token_estimate
            finalized_chunks.append(chunk)

        total_chunks += len(finalized_chunks)
        for index, chunk in enumerate(finalized_chunks):
            chunk["chunk_id"] = f"chunk_{index:03d}"
            chunk_target = chunk["chunk_target"]
            source_spans = [
                SourceSpanReference(
                    label=span.label,
                    start_at=span.start_at,
                    end_at=span.end_at,
                    pages=span.pages,
                )
                for span in chunk["clause_spans"]
            ]
            token_estimate = chunk["token_estimate"]
            token_estimates_pre.append(token_estimate)
            token_estimates_post.append(min(token_estimate, max_tokens))
            quality_flags: list[str] = []
            overlap_count = len(chunk["clause_spans"])
            if token_estimate > max_tokens:
                truncation_risk_count += 1
                quality_flags.append("truncation_risk:high")
            else:
                quality_flags.append("truncation_risk:low")
            if overlap_count >= 4:
                quality_flags.append("weak_label_confidence:high")
            elif overlap_count >= 2:
                quality_flags.append("weak_label_confidence:medium")
            else:
                quality_flags.append("weak_label_confidence:low")

            coverage = _json_field_coverage(chunk_target)
            for field_name, count in coverage.items():
                field_coverage[field_name] += count

            rows.append(
                make_dataset_example(
                    example_id=f"{doc.doc_id}__{chunk['chunk_id']}__extract",
                    doc_id=f"{doc.doc_id}__{chunk['chunk_id']}",
                    source_doc_id=doc.doc_id,
                    chunk_id=chunk["chunk_id"],
                    split=split_map[doc.doc_id],
                    domain="contract",
                    task="extract",
                    user_content=render_extract_prompt("contract", chunk["text"]),
                    assistant_content=chunk_target.model_dump_json(indent=2),
                    token_count_estimate=token_estimate,
                    source_spans=source_spans,
                    quality_flags=quality_flags,
                    metadata={
                        "title": doc.title,
                        "source_dataset": doc.source_dataset,
                        "raw_text_source": doc.source_metadata.get("raw_text_source"),
                        "chunk_char_start": chunk.get("start_at"),
                        "chunk_char_end": chunk.get("end_at"),
                        "overlap_clause_count": overlap_count,
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
        "raw_text_sources": {
            "full_text": sum(1 for doc in normalized_docs if doc.source_metadata.get("raw_text_source") == "full_text"),
            "joined_clauses": sum(
                1 for doc in normalized_docs if doc.source_metadata.get("raw_text_source") == "joined_clauses"
            ),
        },
    }
    return rows, quality_report


def write_contract_processed_splits(
    examples: list[DatasetExample],
    output_dir: Path,
    quality_report: Optional[dict[str, Any]] = None,
) -> dict[str, int]:
    counts = write_split_jsonl(examples, output_dir)
    if quality_report is not None:
        manifest_path = output_dir / "quality_report.json"
        manifest_path.write_text(json.dumps(quality_report, indent=2, sort_keys=True), encoding="utf-8")
    return counts
