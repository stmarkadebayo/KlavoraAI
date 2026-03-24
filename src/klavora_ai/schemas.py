from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NormalizedDocument(StrictModel):
    doc_id: str
    source_dataset: str
    domain: Literal["policy", "contract"]
    raw_text: str
    source_labels: dict[str, Any] = Field(default_factory=dict)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyObligation(StrictModel):
    obligation: str
    owner: Optional[str] = None
    deadline: Optional[str] = None


class PolicyExtraction(StrictModel):
    document_type: Literal["policy", "procedure", "guideline", "standard", "notice", "other"]
    policy_area: Literal["security", "hr", "finance", "it", "procurement", "operations", "other"]
    effective_date: Optional[str] = None
    review_date: Optional[str] = None
    responsible_roles: list[str] = Field(default_factory=list)
    applies_to: list[str] = Field(default_factory=list)
    key_obligations: list[PolicyObligation] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    violations_or_consequences: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class ContractObligation(StrictModel):
    party: str
    obligation: str
    deadline: Optional[str] = None


class ContractExtraction(StrictModel):
    contract_type: Literal["nda", "msa", "sow", "employment", "vendor", "lease", "partnership", "other"]
    parties: list[str] = Field(default_factory=list)
    effective_date: Optional[str] = None
    termination_date: Optional[str] = None
    renewal_terms: list[str] = Field(default_factory=list)
    payment_terms: list[str] = Field(default_factory=list)
    key_obligations: list[ContractObligation] = Field(default_factory=list)
    liability_or_penalty_terms: list[str] = Field(default_factory=list)
    termination_terms: list[str] = Field(default_factory=list)
    confidentiality_terms: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)


class PolicySummaries(StrictModel):
    employee_summary: str
    ops_summary: str


class ContractSummaries(StrictModel):
    executive_summary: str
    action_summary: str


class ClauseSpan(StrictModel):
    label: str
    text: str
    start_at: Optional[int] = None
    end_at: Optional[int] = None
    pages: Optional[str] = None


class SourceSpanReference(StrictModel):
    label: Optional[str] = None
    start_at: Optional[int] = None
    end_at: Optional[int] = None
    pages: Optional[str] = None
    segment_id: Optional[int] = None


class PolicySegment(StrictModel):
    segment_id: int
    text: str
    practice_summaries: list[str] = Field(default_factory=list)


class ContractNormalizedDocument(StrictModel):
    doc_id: str
    domain: Literal["contract"] = "contract"
    source_dataset: str
    title: str
    raw_text: str
    clause_spans: list[ClauseSpan] = Field(default_factory=list)
    weak_extraction_target: dict[str, Any] = Field(default_factory=dict)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyNormalizedDocument(StrictModel):
    doc_id: str
    domain: Literal["policy"] = "policy"
    source_dataset: str
    title: str
    raw_text: str
    practice_summaries: list[str] = Field(default_factory=list)
    segments: list[PolicySegment] = Field(default_factory=list)
    weak_extraction_target: dict[str, Any] = Field(default_factory=dict)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class ContractNLIPair(StrictModel):
    example_id: str
    source_dataset: str = "contractnli"
    premise: str
    hypothesis: str
    label: Literal["entailment", "contradiction", "neutral"]
    split: Literal["train", "validation", "test", "dev"]
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class LegalSummaryPair(StrictModel):
    example_id: str
    source_dataset: str = "legal_summarization"
    doc_id: str
    summary_id: str
    document_text: str
    summary_text: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class PolicySeedExample(StrictModel):
    doc_id: str
    domain: Literal["policy"] = "policy"
    title: str
    source_dataset: str = "synthetic_seed_policy"
    document_text: str
    extraction_target: PolicyExtraction
    summaries: PolicySummaries
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class ContractSeedExample(StrictModel):
    doc_id: str
    domain: Literal["contract"] = "contract"
    title: str
    source_dataset: str = "synthetic_seed_contract"
    document_text: str
    extraction_target: ContractExtraction
    summaries: ContractSummaries
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetExample(StrictModel):
    example_id: str
    doc_id: str
    split: Literal["train", "val", "test"]
    domain: Literal["policy", "contract"]
    task: Literal["extract", "summarize"]
    summary_type: Optional[str] = None
    messages: list[dict[str, str]]
    text: str
    target_format: Literal["json", "text"]
    source_doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    token_count_estimate: Optional[int] = None
    source_spans: list[SourceSpanReference] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
