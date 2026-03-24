# Private Document Intelligence Assistant

A production-style, self-hosted document intelligence system built on a shared open-weight base model with two domain-specific adapters:

- **Policy / Compliance Adapter**
- **Contract / Legal-lite Adapter**

The system is designed to demonstrate all four tracks from the original project post:

1. **Fine-tuning open LLMs with LoRA/QLoRA**
2. **Model deployment**
3. **LLMOps for production AI**
4. **Cost and latency optimization**

This README is written as a handoff document so the project can be continued inside **Codex**, while actual fine-tuning is done with **Unsloth**.

---

## 1. Project summary

### Goal
Build a **self-hosted private document assistant** that can:

- ingest private documents
- extract structured fields into JSON
- generate template-based summaries
- answer basic domain questions later if needed
- expose functionality through an API
- track experiments, traces, and evaluations
- benchmark latency, throughput, and cost tradeoffs

### Why this project exists
The project is meant to show practical competence across the stack, not just model prompting:

- adapting a model for narrow tasks
- serving it like a real application backend
- tracking and evaluating quality over time
- optimizing real-world performance under constraints

### Core design decision
Use **one shared base model** and train **two separate adapters** instead of one mixed fine-tune:

- **Adapter A: Policy / Compliance Intelligence**
- **Adapter B: Contract / Legal-lite Intelligence**

This keeps the architecture modular while preserving one common serving and application layer.

---

## 2. Final project scope

### Product name
**Private Document Intelligence Assistant**

### Modes
The assistant will support two modes:

1. **Policy mode**
2. **Contract mode**

The user selects the mode, and the system loads or routes to the appropriate adapter on top of the shared base model.

### Core user flows

#### Policy mode
Input: internal policy, compliance guideline, procedure, standard, notice, privacy-like policy text

Output:
- structured JSON extraction
- employee-friendly summary
- ops/compliance summary
- highlighted obligations, deadlines, and risks

#### Contract mode
Input: NDA, MSA, SOW, vendor agreement, employment agreement, partnership agreement, lease, etc.

Output:
- structured JSON extraction
- executive summary
- action summary
- highlighted risks, obligations, dates, and penalties

---

## 3. Model strategy

### Shared base model
Use:

- **Primary recommendation:** `Qwen3-8B`
- **Fallback for tighter hardware:** `Gemma 3 4B Instruct`

### Why the shared-base-plus-adapters approach
This project should prove:

- domain adaptation with parameter-efficient fine-tuning
- multi-adapter architecture on one base model
- ability to deploy modular model variants
- cleaner evaluation by domain

### Adapter strategy

#### Adapter A: Policy / Compliance Intelligence
Train on:
- structured extraction into JSON
- template-based summarization

#### Adapter B: Contract / Legal-lite Intelligence
Train on:
- structured extraction into JSON
- template-based summarization

### Important constraint
Do **not** try to fine-tune for all of these at once:

- generic open-ended chat
- retrieval QA
- classification zoo
- support triage
- answer style mimicry
- agent behavior

Keep the tuning targets narrow:

- **structured extraction**
- **template-based summarization**

Everything else can be layered later in the application and agent stack.

---

## 4. Exact target tasks

## Task family 1: Structured extraction into JSON
This is the main fine-tuning target for both adapters.

### Why this is the main task
- easy to evaluate
- easy to compare before/after tuning
- strong portfolio signal
- useful for downstream systems

## Task family 2: Template-based summarization
This is the second fine-tuning target.

### Why this is second
- directly complements extraction
- useful in product workflows
- narrower and cleaner than open-ended chat
- easier to evaluate than generic QA

---

## 5. Output schemas

## 5.1 Policy / Compliance schema

```json
{
  "document_type": "policy|procedure|guideline|standard|notice|other",
  "policy_area": "security|hr|finance|it|procurement|operations|other",
  "effective_date": "string|null",
  "review_date": "string|null",
  "responsible_roles": ["string"],
  "applies_to": ["string"],
  "key_obligations": [
    {
      "obligation": "string",
      "owner": "string|null",
      "deadline": "string|null"
    }
  ],
  "exceptions": ["string"],
  "violations_or_consequences": ["string"],
  "required_actions": ["string"],
  "risk_flags": ["string"]
}
```

### Policy summaries
Generate two summary types:

- `employee_summary`
- `ops_summary`

---

## 5.2 Contract / Legal-lite schema

```json
{
  "contract_type": "nda|msa|sow|employment|vendor|lease|partnership|other",
  "parties": ["string"],
  "effective_date": "string|null",
  "termination_date": "string|null",
  "renewal_terms": ["string"],
  "payment_terms": ["string"],
  "key_obligations": [
    {
      "party": "string",
      "obligation": "string",
      "deadline": "string|null"
    }
  ],
  "liability_or_penalty_terms": ["string"],
  "termination_terms": ["string"],
  "confidentiality_terms": ["string"],
  "risk_flags": ["string"],
  "action_items": ["string"]
}
```

### Contract summaries
Generate two summary types:

- `executive_summary`
- `action_summary`

---

## 6. Dataset strategy

Use a **hybrid dataset strategy**.

Do not rely on raw public datasets exactly as they come. Public datasets will be used as supervision sources, then converted into the project’s own schema and instruction format.

### Contract adapter datasets
Use these datasets:

1. **CUAD**
   - main extraction anchor
   - clause/risk/obligation extraction

2. **ContractNLI**
   - grounding and support verification
   - useful for later eval and evidence checks

3. **LEDGAR**
   - optional clause/provision breadth
   - can expand label coverage

4. **legal_summarization**
   - summarization support
   - summary tuning and eval

### Policy / compliance adapter datasets
Use these datasets:

1. **OPP-115**
   - main extraction anchor
   - strong source for privacy/compliance-style clause annotations

2. **PrivacyQA**
   - grounded QA/evidence support
   - useful for later eval and auxiliary supervision

3. **PolicyQA**
   - optional QA support/eval

4. **ToS-Summaries**
   - policy-like summarization bootstrapping

### Reality check
The policy/compliance side is weaker in public data quality than the contract side.

Plan for:
- synthetic examples
- schema-aligned manual curation
- weak supervision for missing fields

---

## 7. How datasets map to the schemas

## 7.1 Contract adapter mapping

### Directly supervised or near-directly supervised fields
Likely from CUAD / LEDGAR:

- `confidentiality_terms`
- `termination_terms`
- `payment_terms`
- `renewal_terms`
- `liability_or_penalty_terms`
- portions of `key_obligations`
- portions of `risk_flags`

### Weakly derived or synthetic fields
Likely not directly available from public data:

- `action_items`
- `executive_summary`
- normalized `risk_flags`
- structured next-step outputs

### Contract dataset build order
1. Start with CUAD
2. Map CUAD labels to target schema
3. Add legal_summarization for summary behavior
4. Add ContractNLI for grounding and eval
5. Add LEDGAR if more clause breadth is needed

---

## 7.2 Policy / compliance adapter mapping

### Directly supervised or near-directly supervised fields
Likely from OPP-115 and related policy datasets:

- `policy_area`
- `applies_to`
- portions of `key_obligations`
- portions of `risk_flags`
- portions of `violations_or_consequences`

### Weakly derived or synthetic fields
Need manual/synthetic support:

- `responsible_roles`
- `review_date`
- `required_actions`
- `employee_summary`
- `ops_summary`

### Policy dataset build order
1. Start with OPP-115
2. Map annotations to target schema
3. Add ToS-Summaries for summary behavior
4. Add PrivacyQA / PolicyQA for grounded eval and auxiliary supervision
5. Add synthetic/manual schema-aligned examples where public coverage is weak

---

## 8. Data engineering pipeline

Build the dataset pipeline in four stages.

## Stage A: Normalize raw sources
Every source should be converted into a common internal document format.

Suggested normalized structure:

```json
{
  "doc_id": "string",
  "source_dataset": "string",
  "domain": "policy|contract",
  "raw_text": "string",
  "source_labels": {},
  "source_metadata": {}
}
```

## Stage B: Convert into project task formats
Create two trainable task formats for each adapter:

### Extraction format
Input:
- document text
- task instruction

Output:
- target JSON schema

### Summarization format
Input:
- document text
- requested summary type

Output:
- target summary template

## Stage C: Add synthetic and weakly supervised labels
Use:
- deterministic rules where possible
- LLM-assisted transformations where needed
- manual review for a smaller high-quality subset

Use synthetic augmentation mostly for:
- `action_items`
- role-based summaries
- normalized obligations
- missing responsibility/deadline fields

## Stage D: Create gold evaluation sets
Do not use only converted training data for evaluation.

Make separate manually checked validation/test sets for:
- extraction quality
- summary quality
- grounding/faithfulness
- field-level correctness

---

## 9. Recommended first versions

## Contract adapter v1
Start with:
- CUAD
- legal_summarization
- small manually reviewed synthetic set aligned to the target schema

Then add:
- ContractNLI
- LEDGAR

## Policy adapter v1
Start with:
- OPP-115
- ToS-Summaries
- small manually reviewed synthetic set aligned to the target schema

Then add:
- PrivacyQA
- PolicyQA

This keeps the first version manageable.

---

## 10. Training plan

Fine-tuning will be done with **Unsloth**.

### General training design
Train separate adapters on top of the same base model:

- policy adapter
- contract adapter

### Fine-tuning phases

#### Phase 1: extraction-only tuning
Train the model to reliably produce structured JSON.

Reason:
- stricter target
- easier metrics
- clearer error diagnosis

#### Phase 2: summary tuning
Train template-based summaries after extraction behavior stabilizes.

Reason:
- summary quality is easier to improve after the model has learned domain structure

### Suggested experiment matrix
For each adapter:

1. Base model zero-shot baseline
2. LoRA fine-tune
3. QLoRA fine-tune
4. Optional comparison between prompt-only vs adapter-tuned

### What to log during training
- dataset version
- schema version
- train/val split IDs
- prompt template version
- LoRA/QLoRA configuration
- base model version
- epoch/checkpoint metrics
- qualitative failure examples

---

## 11. Suggested Unsloth workstreams

The actual notebook/scripts in Unsloth should likely cover:

1. loading the shared base model
2. preparing the dataset in instruction format
3. training adapter A
4. training adapter B
5. saving adapters separately
6. exporting adapter artifacts for inference use
7. running quick post-train evals

### Important output artifacts from Unsloth
Store these clearly:

- `adapter_policy/`
- `adapter_contract/`
- tokenizer/config references
- training configs
- metrics snapshots
- example outputs before/after tuning

---

## 12. Inference and serving plan

Deployment should be done separately from training.

### Serving target
Expose the model behind an API with:
- streaming support
- structured output mode
- adapter selection
- logging
- authentication if needed

### Primary serving backend
Use **vLLM**.

### Why
- clean serving story
- production-style API setup
- suitable for later benchmarking and comparison

### Optional secondary backend
Add **llama.cpp / llama-cpp-python** later as a comparison backend for optimization experiments.

### Required API capabilities
- choose domain mode: `policy` or `contract`
- select summary type
- return extracted JSON
- return summary output
- return metadata such as model, adapter, latency, token counts

### Possible API endpoints

```text
POST /extract
POST /summarize
POST /analyze
GET /health
GET /metrics
```

### Suggested request shape

```json
{
  "mode": "policy",
  "task": "extract",
  "document_text": "...",
  "summary_type": null
}
```

### Suggested response shape

```json
{
  "mode": "policy",
  "task": "extract",
  "result": {},
  "metadata": {
    "base_model": "Qwen3-8B",
    "adapter": "policy_adapter_v1",
    "latency_ms": 0,
    "input_tokens": 0,
    "output_tokens": 0
  }
}
```

---

## 13. LLMOps plan

This project should not stop at training and serving.

Add an LLMOps layer for:
- experiment tracking
- data versioning
- tracing
- evaluation runs
- regression checks
- failure logging

### Suggested tools
- **MLflow** for experiment tracking and evaluation runs
- **Langfuse** for tracing and prompt/output observability
- **DVC** for data and experiment versioning

### What to track
- dataset version
- model/base version
- adapter version
- schema version
- prompt version
- endpoint latency
- JSON validity rate
- field-level extraction scores
- summary scores
- groundedness/faithfulness checks
- common error patterns

### Failure taxonomy to create
Track failure types like:
- invalid JSON
- missing key fields
- hallucinated obligations
- wrong party/owner mapping
- unsupported deadlines
- summary omissions
- summary contradictions

---

## 14. Evaluation plan

Evaluation should be domain-specific and task-specific.

## 14.1 Extraction metrics
Use:
- JSON validity rate
- exact-match on selected fields where possible
- field-level precision / recall / F1
- list overlap for obligations, actions, and risks
- normalized date extraction accuracy

### Example contract extraction fields to score
- parties
- effective_date
- termination_terms
- payment_terms
- confidentiality_terms
- risk_flags

### Example policy extraction fields to score
- policy_area
- applies_to
- key_obligations
- responsible_roles
- risk_flags
- required_actions

## 14.2 Summary metrics
Use a mix of:
- template adherence
- coverage of required facts
- groundedness / faithfulness
- factual omission rate
- human review on a small gold subset

Avoid relying only on generic summary metrics. They will not tell you enough.

## 14.3 Grounding / support checks
Use evidence-aware eval where possible, especially on the contract side.

Evaluate whether extracted claims and summary claims are actually supported by the source document.

## 14.4 Baselines to compare
Compare against:
- base model prompt-only
- LoRA adapter
- QLoRA adapter
- optional secondary backend or quantized variant

---

## 15. Cost and latency optimization plan

This project must include performance engineering, not just task quality.

### Benchmarks to run
Compare:
- base model vs adapter-tuned model
- LoRA vs QLoRA
- different quantization setups
- vLLM vs secondary backend later
- different context lengths
- different concurrency levels

### Metrics to collect
- TTFT
- tokens/sec
- end-to-end latency
- inter-token latency if possible
- memory usage
- throughput under concurrent load
- cost estimate per 1,000 requests

### Workload types
Run benchmarks for:
- short documents
- medium documents
- long documents
- extraction-only requests
- summary-only requests
- combined analysis requests

### Why this matters
This is what makes the project align with the fourth track from the original post.

---

## 16. Recommended repository structure

```text
private-document-intelligence-assistant/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── schema_policy.md
│   ├── schema_contract.md
│   ├── evaluation_plan.md
│   └── benchmark_plan.md
├── data/
│   ├── raw/
│   ├── normalized/
│   ├── processed/
│   ├── synthetic/
│   ├── gold_eval/
│   └── schema_maps/
├── notebooks/
│   ├── unsloth_policy_training.ipynb
│   ├── unsloth_contract_training.ipynb
│   └── dataset_exploration.ipynb
├── scripts/
│   ├── ingest_cuad.py
│   ├── ingest_contractnli.py
│   ├── ingest_ledgar.py
│   ├── ingest_opp115.py
│   ├── ingest_privacyqa.py
│   ├── ingest_policyqa.py
│   ├── ingest_tos_summaries.py
│   ├── normalize_documents.py
│   ├── build_contract_examples.py
│   ├── build_policy_examples.py
│   ├── build_synthetic_examples.py
│   ├── run_eval.py
│   ├── run_benchmarks.py
│   └── export_adapters.py
├── training/
│   ├── configs/
│   ├── outputs/
│   └── logs/
├── adapters/
│   ├── policy/
│   └── contract/
├── serving/
│   ├── api/
│   ├── schemas/
│   ├── middleware/
│   └── config/
├── eval/
│   ├── extraction/
│   ├── summarization/
│   ├── grounding/
│   └── regression/
├── benchmarks/
│   ├── load/
│   ├── latency/
│   └── reports/
├── mlops/
│   ├── mlflow/
│   ├── dvc/
│   └── langfuse/
└── app/
    ├── ui/
    └── client/
```

---

## 17. Immediate build order

Follow this order.

### Phase 0: planning
1. lock the base model
2. lock schema v1 for both adapters
3. create repo skeleton
4. define evaluation rules

### Phase 1: data
1. ingest CUAD and OPP-115 first
2. normalize raw documents
3. build schema-mapped extraction examples
4. create small synthetic sets for missing fields
5. create small gold eval sets

### Phase 2: training
1. run base-model prompt-only baseline
2. train contract extraction adapter
3. train policy extraction adapter
4. evaluate extraction
5. train summary variants

### Phase 3: serving
1. stand up API server
2. expose extraction and summarization endpoints
3. add adapter selection
4. add response metadata and logging

### Phase 4: ops and eval
1. add MLflow, DVC, Langfuse
2. add regression eval suite
3. create failure dashboard and error taxonomy

### Phase 5: optimization
1. benchmark latency and throughput
2. compare LoRA vs QLoRA
3. compare quantization settings
4. optionally compare serving backends

---

## 18. Questions this project should answer clearly

By the end of the build, this project should be able to answer:

1. Can a small open model be adapted to produce reliable structured outputs in these domains?
2. Is QLoRA enough, or is standard LoRA materially better here?
3. How much quality improvement do the adapters deliver over prompt-only baselines?
4. Which fields are most error-prone?
5. What is the latency/cost tradeoff of each serving setup?
6. Can one base model support multiple specialized adapters cleanly?

---

## 19. What this project is not

To avoid scope creep, this project is **not** initially:

- a generic chatbot
- a broad legal advice system
- a broad compliance advisory system
- a multi-agent platform
- a full RAG platform
- a workflow router

Those can come later.

For now, the project is:

- document intelligence
- structured extraction
- template summarization
- production-style operation and benchmarking

---

## 20. How this project connects to later work

This project is intentionally designed to become the foundation for a later **agentic document review system**.

The same tuned specialist model can later be used inside:
- multi-step document review workflows
- tool-using assistants
- human-in-the-loop approval systems
- report generation pipelines

But that is **Project 2**. Do not merge it into v1.

---

## 21. Codex handoff notes

When continuing in Codex, treat this README as the authoritative spec for v1.

### First implementation priorities in Codex
1. create repo skeleton
2. write ingestion scripts for CUAD and OPP-115
3. define canonical normalized document format
4. define JSON schemas in code
5. build dataset conversion pipeline
6. create baseline evaluation scripts
7. create placeholders for Unsloth training notebooks/scripts

### Keep these boundaries clear
- **Codex:** repository work, data pipelines, serving code, eval code, benchmarking, orchestration
- **Unsloth:** actual fine-tuning runs and adapter export

### Required artifacts to preserve
Every major run should preserve:
- config used
- dataset snapshot/version
- checkpoint or adapter output
- evaluation report
- qualitative sample outputs
- latency benchmark report

---

## 22. Suggested task list for the next session in Codex

Use this as the initial implementation checklist.

### Repo setup
- [ ] create repository structure
- [ ] create `docs/` and schema docs
- [ ] add base config files
- [ ] add dependency manifests

### Data ingestion
- [ ] ingest CUAD
- [ ] ingest OPP-115
- [ ] define raw document schema
- [ ] normalize both datasets

### Data conversion
- [ ] map CUAD to contract schema
- [ ] map OPP-115 to policy schema
- [ ] generate instruction-format examples
- [ ] build synthetic support for missing fields

### Evaluation
- [ ] implement JSON validity checks
- [ ] implement field-level scoring
- [ ] create small gold eval sets
- [ ] create baseline comparison harness

### Training support
- [ ] prepare exportable datasets for Unsloth
- [ ] create training configs for both adapters
- [ ] define artifact storage paths

### Serving
- [ ] scaffold API server
- [ ] define `/extract` and `/summarize`
- [ ] implement adapter selection
- [ ] return metadata in responses

### Ops
- [ ] wire MLflow
- [ ] wire DVC
- [ ] wire Langfuse placeholders

### Optimization
- [ ] design benchmark scenarios
- [ ] define latency/throughput metrics
- [ ] create benchmark runner skeleton

---

## 23. Final v1 recommendation

Build this exact version first:

**A self-hosted private document intelligence assistant with one shared base model and two domain-specific adapters, focused on structured extraction and template-based summarization for policy/compliance documents and contracts.**

That scope is large enough to be impressive and small enough to finish well.


## DATASETS NEEDED

For v1, you need 2 training datasets, not one giant mixed dataset:

1. `contract` adapter dataset
2. `policy` adapter dataset

Inside each one, you should have 2 task subsets:

- extraction examples
- summary examples

So conceptually the full data plan is 4 dataset buckets:

1. contract extraction
2. contract summarization
3. policy extraction
4. policy summarization

For actual project execution, I’d stage it like this:

Phase 1:
- contract extraction dataset
- policy extraction dataset

Phase 2:
- contract summary dataset
- policy summary dataset

That is enough for the main fine-tuning plan.

If you mean raw source datasets from outside:
- Contract side: start with `CUAD`, then optionally `legal_summarization`, `ContractNLI`, `LEDGAR`
- Policy side: start with `OPP-115`, then optionally `ToS-Summaries`, `PrivacyQA`, `PolicyQA`

So minimum external sources for a serious v1 start:
- 2 core source datasets: `CUAD` and `OPP-115`

Better practical v1:
- 4 core-ish sources:
  - `CUAD`
  - `legal_summarization`
  - `OPP-115`
  - `ToS-Summaries`

My recommendation:
- Don’t try to collect everything first.
- Build 1 solid dataset first: `contract extraction`.
- Then add `policy extraction`.
- Only after that, add the 2 summary datasets.

So the shortest answer is:
- Total training datasets for the product design: 4
- Minimum raw source datasets to start real work: 2
- Best realistic v1 source coverage: about 4 to 8 sources over time