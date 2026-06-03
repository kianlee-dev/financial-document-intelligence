# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Financial document intelligence system for analyzing annual reports and 10-K filings. The `data/` directory contains source PDFs (AAPL 10-K, AIA/HSBC/JPMC annual reports). No source code exists yet — this project is at the scaffolding stage.

## Behavioral Guidelines

1. **Think Before Coding** — State assumptions explicitly. If multiple interpretations exist, present them. If a simpler approach exists, say so. If something is unclear, stop and ask.

2. **Simplicity First** — Minimum code that solves the problem. No features beyond what was asked. No abstractions for single-use code. No speculative flexibility. If 200 lines could be 50, rewrite.

3. **Surgical Changes** — Touch only what you must. Don't improve adjacent code or formatting. Match existing style. Remove only imports/variables that YOUR changes made unused.

4. **Goal-Driven Execution** — Transform tasks into verifiable goals. For multi-step tasks, state a brief plan with verification checks. Write tests that reproduce bugs before fixing them.

## Environment

Python 3.11 venv at `./venv`. Activate before running anything:

```bash
source venv/bin/activate
```

Key installed libraries (no requirements.txt yet — derive from venv/lib/python3.11/site-packages):
- **anthropic** — Claude API for LLM inference
- **chromadb** — vector store for document embeddings
- **pypdf** — PDF parsing
- **sentence_transformers** + **torch** — local embeddings
- **fastapi** + **uvicorn** — API layer
- **langgraph** — agent state graph orchestration
- **langfuse** — LLM observability and tracing
- **pydantic** / **pydantic_settings** — data models and config
- **pytest** — testing

## Architecture

```
data/                        # source PDFs
src/
  ingestion/                 # PDF loading + chunking (pypdf, RecursiveCharacterTextSplitter)
  retrieval/                 # vector DB + similarity search (ChromaDB, sentence-transformers)
  agent/                     # LangGraph state graph with tool calling
  context/                   # system prompt assembly, sliding window memory, token budget tracking
  llm/                       # abstraction layer (Claude API; vLLM as future addition)
  api/                       # FastAPI endpoints
  evals/                     # evaluation suite (run as pytest)
tests/
Dockerfile                   # future
```

### Component Details

- **Ingestion** (`src/ingestion/`) — load PDFs with `pypdf`, split into chunks via `RecursiveCharacterTextSplitter`
- **Retrieval** (`src/retrieval/`) — embed chunks with `sentence-transformers`, store and query via ChromaDB
- **Agent** (`src/agent/`) — LangGraph state graph; tools: `search_documents`, `calculate`, `compare_companies`
- **Context engineering** (`src/context/`) — system prompt assembly, sliding window memory, token budget tracking
- **LLM backend** (`src/llm/`) — abstraction layer supporting Claude API; vLLM deferred as future addition
- **API** (`src/api/`) — FastAPI endpoints exposing agent queries
- **Observability** — Langfuse tracing, scores, and datasets wired throughout
- **Evaluation** (`src/evals/`) — retrieval precision@k, LLM-as-judge, faithfulness detection; run as pytest
- **Google ADK** — deferred; future framework comparison

## Development Commands

```bash
# Run tests
pytest

# Run a single test file
pytest tests/path/to/test_file.py

# Run a single test
pytest tests/path/to/test_file.py::test_name

# Start FastAPI dev server (once app exists)
uvicorn src.api.main:app --reload
```
