# production-rag-with-python

Companion code for **Retrieval-Augmented Generation (RAG) in Production** —
*Build, Evaluate, Optimize, and Deploy Enterprise RAG Systems with Python*
by Masud Mondal.

One system runs through the whole book: **Lighthouse**, the internal assistant
of a fictional insurer called **Harbourline**. This repository is that system
at three points in its life, plus the harness that measures it.

---

## What is here

| Path | Chapters | Contents |
|---|---|---|
| `data/generate_corpus.py` | 3 | Deterministic synthetic Harbourline corpus |
| `src/lighthouse/ingestion/` | 4–5 | Document schema, chunk metadata |
| `src/lighthouse/chunking/` | 6 | Fixed, recursive, structural, parent-document |
| `src/lighthouse/embeddings/` | 7 | Embedder interface and a local implementation |
| `src/lighthouse/retrieval/` | 8, 9, 12–15 | Vector store, BM25, fusion, filters, types |
| `src/lighthouse/evaluation/` | 16–19 | Hit rate, recall@k, precision@k, MRR, nDCG |
| `evaluation/run_experiments.py` | 6, 11, 12, 13, 17 | The harness behind the measured tables |
| `projects/01-knowledge-assistant/` | 10 | Project 1 — the baseline |
| `projects/02-hybrid-rag/` | 21 | Project 2 — improved and measured |
| `projects/03-production-api/` | 27 | Project 3 — the deployed service |
| `tests/` | — | 19 tests over the executable paths |

---

## Prerequisites

- **Python 3.11 or later** (3.13 is the reference)
- No GPU required for anything in this README

## Installation

```bash
git clone <your-clone-url> production-rag-with-python
cd production-rag-with-python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Configuration

```bash
cp .env.example .env
```

Fill in `OPENAI_API_KEY`, `GENERATION_MODEL`, and `EMBEDDING_MODEL`. `.env` is
git-ignored. No key is read from anywhere else, and none is ever logged —
Chapter 22 explains the allowlist that enforces that.

## Generate the corpus

```bash
python3 data/generate_corpus.py
```

Prints the document counts and a corpus hash. The hash is deterministic: the
same seed always produces the same corpus, which is what makes a measurement
comparable to the one before it.

## Run the experiments

```bash
python3 evaluation/run_experiments.py
```

Executes the chunking comparison, the retrieval-mode comparison, the filtering
and leakage checks, and the metric verification. Writes JSON to
`evaluation/results/`.

## Run the tests

```bash
python3 -m pytest
mypy src tests
```

---

## What runs without credentials, and what does not

**Runs offline:** corpus generation, all four chunking strategies, the local
embedder, the flat vector store, BM25, reciprocal rank fusion, metadata and
access filtering, all five retrieval metrics, and the full test suite.

**Needs an API key:** generation, API embeddings, groundedness scoring, and
anything in Project 3 that calls a model.

**Needs more than this repository:** a served Qdrant instance (Chapter 8), a
cross-encoder reranker (Chapter 15), and a container runtime (Chapter 26).

The book is explicit about which figures were measured on this corpus, which
were calculated, and which are illustrative. Appendix A restates it.

## A note on the corpus size

The book narrates a 40,000-document corpus. The corpus shipped here is 1,200
documents, so that you can regenerate and re-index it in minutes. Absolute
numbers will differ; the procedure and the ratios are what transfer.

## Licence

MIT. See `LICENSE`.
