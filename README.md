# Financial Document Intelligence

An AI agent that ingests financial reports, stores them in a vector database, and answers questions with citations. Built with a LangGraph tool-calling agent over a RAG pipeline, traced through Langfuse, and served via FastAPI. Supports annual reports and SEC filings across multiple markets (US, UK, HK, JP).

## Project Structure
```
financial-document-intelligence/
├── src/
│   ├── ingestion/
│   │   ├── loader.py          # PDF loading — single and batch, metadata attachment
│   │   └── chunker.py         # Text chunking with RecursiveCharacterTextSplitter
│   ├── retrieval/
│   │   ├── embeddings.py      # Embedding service — sentence-transformers wrapper
│   │   └── vectorstore.py     # ChromaDB — storage, similarity search, deletion
│   ├── agent/
│   │   ├── tools.py           # Tool implementations, Anthropic schemas, dispatch registry
│   │   └── graph.py           # LangGraph StateGraph — nodes, routing, run()
│   ├── context/
│   │   └── prompt_builder.py  # System prompt, chunk formatting, prompt assembly
│   ├── llm/
│   │   └── client.py          # Claude API wrapper — generate() and generate_with_tools()
│   ├── api/
│   │   └── main.py            # FastAPI endpoints — query, upload, list, delete
│   └── evals/                 # Evaluation suite (planned)
├── scripts/
│   ├── ingest.py              # Batch ingestion pipeline
│   ├── query.py               # Direct Q&A chain (no agent)
│   └── agent_query.py         # Agent-based queries with tool calling
├── tests/
│   └── agent/
│       ├── test_tools.py      # 11 tests — filters, search, metadata, schemas
│       ├── test_graph.py      # 7 tests — routing, dispatch, multi-turn, edge cases
│       └── test_smoke.py      # 1 test — real API end-to-end, skippable
├── data/                      # Financial PDFs (not tracked) + metadata.json
├── design_decisions.md        # Architecture decisions with reasoning
├── Dockerfile
├── requirements.txt
└── pytest.ini
```

## Architecture
```
                         ┌──── tool call ────→ [Tools Node] ───┐
                         │                     search_documents │
                         │                     get_metadata      │
User Query → [FastAPI] → [Agent Node] ◄────────────────────────┘
                         │
                         └──── final answer ──→ [Response + Citations]
                                                       │
                                               ┌───────┴───────┐
                                               │               │
                                        [Langfuse Trace]  [Return to User]
```

The agent sits on top of a standard RAG pipeline. Instead of a fixed retrieve → generate chain, the LangGraph agent decides what to search, how many times, and when it has enough information to answer.

**Why the agent matters:** A fixed chain can only do one retrieval per query. The agent calls `search_documents` multiple times with different filters — when asked to compare HSBC and JPMorgan, it searches each company separately, then synthesises the results. That cross-company comparison is impossible with a fixed chain.

## Component Responsibilities

| Module | Responsibility | Justification |
|--------|---------------|---------------|
| `ingestion/` | PDF loading and chunking | Separated from retrieval — ingestion runs once, retrieval runs per query |
| `retrieval/` | Embedding and vector search | Storage and search logic testable independently of agent |
| `agent/` | Tool-calling decision loop | Agent orchestrates retrieval without knowing storage internals |
| `context/` | Prompt assembly | What goes into the prompt is separate from how it gets sent |
| `llm/` | API wrapper | Model-agnostic interface — swap Claude for any provider in one line |
| `api/` | HTTP interface | FastAPI layer is separate from agent logic |

## Design Decisions

**Tool-calling agent, not ReAct or multi-agent**
ReAct agents reason in free-form loops which is unpredictable iteration counts, harder to debug. Multi-agent architectures add orchestration complexity without benefit at this scale. Tool-calling gives structured, predictable behavior: the model either calls a defined tool or gives a final answer. Each decision is visible in Langfuse traces.

**Raw Anthropic SDK, not LangChain ChatAnthropic**
`LLMClient` wraps the raw `anthropic` SDK directly. This avoids locking the agent into LangChain's model abstraction which makes changing the LLM backend a one-line config change in the client constructor.

**Minimal tools**
`search_documents` and `get_metadata` are sufficient. A `calculate` tool is unnecessary as Claude handles financial arithmetic natively. A `compare_companies` tool is redundant; it's just two `search_documents` calls with different filters. Fewer tools means fewer wrong decisions by the agent.

**ChromaDB as metadata source, not a separate config file**
Document metadata (company, year, type) is stored in ChromaDB alongside each chunk. `get_documents_list()` queries ChromaDB directly for unique document metadata as there is no separate config file to maintain or sync. A `metadata.json` file exists only for batch ingestion via `scripts/ingest.py`.

**RecursiveCharacterTextSplitter at 2000 characters (~500 tokens)**
Financial text mixes narrative paragraphs with dense tables. 2000 characters is roughly 1–2 paragraphs which is small enough for topically focused embeddings, but large enough to be self-contained. 400-character overlap (20%) prevents information loss at chunk boundaries. Fixed-size splitting would break mid-sentence; semantic splitting adds embedding overhead during ingestion without proportional benefit.

**Labelled chunk formatting for citations**
Each chunk in the prompt is wrapped with `[Source: company, type, year, Page N]`. The LLM reads these labels and cites them in answers. Raw text would make citation impossible; structured XML adds parsing complexity without benefit at this scale.

**Additive reducer on messages state**
LangGraph's `Annotated[list, operator.add]` ensures each node appends to the messages list rather than replacing it. Without this, each node would overwrite previous messages — the agent would lose its tool call history mid-loop.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Send a question, get an answer with citations and Langfuse trace ID |
| `POST` | `/documents` | Upload a PDF with metadata for ingestion |
| `GET` | `/documents` | List all available documents in the collection |
| `DELETE` | `/documents` | Remove a document's chunks by metadata filter |

## Test Coverage

18 tests + 1 smoke test, all passing.

**Tool tests**

| Test | What it verifies |
|------|-----------------|
| `test_build_filters_*` (4 tests) | Filter construction for None, company-only, year-only, and both |
| `test_search_documents_passes_filters` | Correct filters passed to VectorStore, output has `[Source:]` labels |
| `test_search_documents_no_results` | Empty results return clean message, no crash |
| `test_get_metadata` | All 5 companies returned from ChromaDB |
| `test_get_metadata_empty_collection` | Empty collection returns "No documents available" |
| `test_get_documents_list_empty_collection` | Empty collection returns empty list |
| `test_registry_and_schemas_match` | Tool registry and Anthropic schemas have matching names and required fields |

**Graph tests**

| Test | What it verifies |
|------|-----------------|
| `test_route_to_tools` | Tool-use response routes to tools node |
| `test_route_to_end` | Text-only response routes to END |
| `test_tools_node_dispatches` | Tool node executes correct function, wraps result with matching tool_use_id |
| `test_run_multi_turn` | Full loop: search → answer, exactly 2 LLM calls |
| `test_run_direct_answer` | Direct answer without tool call, exactly 1 LLM call |
| `test_run_cross_company_comparison` | Agent calls search twice with different filters, 3 LLM calls |
| `test_run_missing_company` | Agent handles empty search results gracefully |

**Smoke test**

| Test | What it verifies |
|------|-----------------|
| `test_agent_answers_single_company` | Real API end-to-end — skipped without `ANTHROPIC_API_KEY` |

## Observability

Every query is traced through Langfuse at three levels:
- **Retrieval** — query, results, latency
- **Context assembly** — formatted prompt
- **LLM calls** — prompt, completion, tokens

The API returns a `trace_id` with each response for direct lookup in the Langfuse dashboard.

## How to Run

**Prerequisites**
- Python 3.11+
- Anthropic API key (for LLM calls)
- Langfuse account (free tier, for tracing)

**Local setup**
```bash
git clone https://github.com/kianlee-dev/financial-document-intelligence.git
cd financial-document-intelligence
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Environment variables**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="https://us.cloud.langfuse.com"
```

**Add documents**

Option 1 — via API (recommended):
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@AAPL_2025_10K.pdf" \
  -F "company_name=Apple" \
  -F "report_year=2025" \
  -F "document_type=10-K"
```

Option 2 — batch ingestion (development):
Place PDFs in `data/` with a `data/metadata.json` mapping filenames to metadata, then run:
```bash
PYTHONPATH=. python scripts/ingest.py
```

**Run the API**
```bash
PYTHONPATH=. uvicorn src.api.main:app --reload
```

Open `http://localhost:8000/docs` for interactive Swagger UI.

**Run tests**
```bash
# All tests except smoke (no API key needed)
PYTHONPATH=. pytest tests/agent -m "not smoke"

# Smoke test (requires API key + ingested data)
PYTHONPATH=. pytest tests/agent/test_smoke.py -m smoke
```

**Docker**
```bash
docker build -t financial-document-intelligence .
docker run -p 8000:8000 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e LANGFUSE_PUBLIC_KEY=$LANGFUSE_PUBLIC_KEY \
  -e LANGFUSE_SECRET_KEY=$LANGFUSE_SECRET_KEY \
  -e LANGFUSE_HOST=$LANGFUSE_HOST \
  financial-document-intelligence
```

## Limitations and Future Work

**Current limitations**
- No cross-query memory — each query is independent, follow-up questions require full context
- No token budget management — works within Claude's context window at current scale but would need truncation logic for longer conversations
- Single embedding model — all-MiniLM-L6-v2 is a prototyping choice, not production-grade
- No hybrid search — pure vector search can miss exact keyword matches (acronyms, ticker symbols)

**Planned extensions**
- Evaluation suite — retrieval precision@k, LLM-as-judge scoring, faithfulness detection, run as pytest with Langfuse integration
- Sliding window memory for conversational follow-up queries
- Reflexion/self-critique node — second LLM pass to verify answer is grounded in sources, critical for financial accuracy
- vLLM integration for local model serving in air-gapped deployments
- Google ADK comparison — rebuild one agent flow in ADK to demonstrate framework flexibility
- Hybrid search (vector + BM25 keyword) via reciprocal rank fusion
