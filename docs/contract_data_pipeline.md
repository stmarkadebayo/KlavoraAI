# Contract Data Pipeline

The contract fine-tuning path now has four scripts:

- [ingest_cuad.py](/Users/mac/Desktop/KlavoraAI/scripts/ingest_cuad.py)
- [ingest_contractnli.py](/Users/mac/Desktop/KlavoraAI/scripts/ingest_contractnli.py)
- [ingest_legal_summarization.py](/Users/mac/Desktop/KlavoraAI/scripts/ingest_legal_summarization.py)
- [build_contract_examples.py](/Users/mac/Desktop/KlavoraAI/scripts/build_contract_examples.py)

## What each script does

`ingest_cuad.py`

- reads the downloaded CUAD clause parquet
- groups clause rows by source contract
- optionally joins full contract text if you later add the companion full-text dataset
- writes normalized contract documents to [contract_cuad_documents.jsonl](/Users/mac/Desktop/KlavoraAI/data/normalized/contract_cuad_documents.jsonl)
- derives weak extraction targets mapped into the project schema

`ingest_contractnli.py`

- converts ContractNLI parquet splits into normalized JSONL
- this is mainly for grounding/eval later, not the first extraction fine-tune

`ingest_legal_summarization.py`

- joins the corpus, queries, and qrels files into document-summary pairs
- writes [legal_summarization_pairs.jsonl](/Users/mac/Desktop/KlavoraAI/data/normalized/legal_summarization_pairs.jsonl)

`build_contract_examples.py`

- turns normalized contract documents into chunked instruction-style extraction examples
- targets roughly `700-900` tokens per chunk and caps the training recipe at `1024`
- preserves `source_doc_id`, `chunk_id`, source spans, and quality flags
- writes train/val/test JSONL splits under [contract_main](/Users/mac/Desktop/KlavoraAI/data/processed/contract_main)
- writes a dataset quality report at [quality_report.json](/Users/mac/Desktop/KlavoraAI/data/processed/contract_main/quality_report.json)

## Recommended run order

```bash
python3 scripts/ingest_cuad.py
python3 scripts/ingest_contractnli.py
python3 scripts/ingest_legal_summarization.py
python3 scripts/build_contract_examples.py --target-chunk-tokens 800 --max-tokens 1024
```

Then inspect:

- [contract_cuad_documents.jsonl](/Users/mac/Desktop/KlavoraAI/data/normalized/contract_cuad_documents.jsonl)
- [contract_main train split](/Users/mac/Desktop/KlavoraAI/data/processed/contract_main/train.jsonl)
- [contract_main quality report](/Users/mac/Desktop/KlavoraAI/data/processed/contract_main/quality_report.json)

## Important caveat

The downloaded CUAD clause-classification dataset does not contain full contract text by itself. Without the companion full-text dataset, the ingestion script falls back to joining labeled clauses in order. That is enough to keep the pipeline moving, but the preferred path is to add the companion full-text dataset so chunking and field grounding operate on the real document.
