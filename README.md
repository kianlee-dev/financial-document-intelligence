# Financial Document Intelligence

An AI agent that ingests financial reports, stores them in a vector database, and answers questions with citations. Built with a LangGraph tool-calling agent over a RAG pipeline, evaluated with an LLM-as-judge suite, traced through Langfuse, and served via FastAPI. Supports swappable LLM backends (Claude API or local models via Ollama/vLLM) and includes a Google ADK comparison agent. Covers annual reports and SEC filings across multiple markets (US, UK, HK, JP).

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
│   ├── agent_adk/
│   │   └── agent.py           # Google ADK agent — async tool wrappers, Gemini-specific prompt
│   ├── context/
│   │   └── prompt_builder.py  # System prompt, chunk formatting, prompt assembly
│   ├── llm/
│   │   ├── client.py          # Factory — routes to Anthropic or OpenAI backend via LLM_BACKEND
│   │   ├── anthropic_client.py # Claude API wrapper — generate() and generate_with_tools()
│   │   └── openai_client.py   # OpenAI-compatible wrapper — Ollama/vLLM with message translation
│   ├── api/
│   │   └── main.py            # FastAPI endpoints — query, upload, list, delete
│   └── evals/
│       ├── test_dataset.py    # 35 ground-truth Q&A pairs across 5 categories
│       ├── judge.py           # LLM-as-judge — scores relevance, accuracy, faithfulness
│       └── runner.py          # Eval pipeline — agent + judge + Langfuse scoring
├── scripts/
│   ├── ingest.py              # Batch ingestion pipeline
│   ├── query.py               # Direct Q&A chain (no agent)
│   ├── agent_query.py         # Agent-based queries with tool calling
│   └── adk_query.py           # ADK agent queries with Gemini
├── tests/
│   ├── agent/
│   │   ├── test_tools.py      # 11 tests — filters, search, metadata, schemas
│   │   ├── test_graph.py      # 7 tests — routing, dispatch, multi-turn, edge cases
│   │   └── test_smoke.py      # 1 test — real API end-to-end, skippable
│   └── evals/
│       └── test_eval_suite.py # 5 threshold tests — relevance, accuracy, faithfulness, precision, hallucinations
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
| `agent_adk/` | Google ADK comparison agent | Same tools and RAG pipeline, different framework — demonstrates portability |
| `context/` | Prompt assembly | What goes into the prompt is separate from how it gets sent |
| `llm/` | LLM backend abstraction | Factory pattern — swap Claude for Ollama/vLLM via environment variable |
| `api/` | HTTP interface | FastAPI layer is separate from agent logic |
| `evals/` | LLM-as-judge evaluation suite | Measures retrieval and answer quality independently of agent logic |

## Design Decisions

**Tool-calling agent, not ReAct or multi-agent**
ReAct agents reason in free-form loops which is unpredictable iteration counts, harder to debug. Multi-agent architectures add orchestration complexity without benefit at this scale. Tool-calling gives structured, predictable behavior: the model either calls a defined tool or gives a final answer. Each decision is visible in Langfuse traces.

**Raw Anthropic SDK, not LangChain ChatAnthropic**
`AnthropicClient` wraps the raw `anthropic` SDK directly. This avoids locking the agent into LangChain's model abstraction. A parallel `OpenAIClient` wraps the OpenAI SDK for Ollama/vLLM compatibility. Both expose identical `generate()` and `generate_with_tools()` interfaces. The factory class (`LLMClient`) routes to the correct backend via the `LLM_BACKEND` environment variable — the agent graph doesn't know which backend is active.

**Message translation over neutral format**
The agent graph stores messages in Anthropic format (the primary backend). `OpenAIClient` translates tool schemas, message history, and responses internally — converting Anthropic's `tool_use` blocks to OpenAI's `tool_calls`, `tool_result` messages to `role: tool`, and wrapping responses in SimpleNamespace to match Anthropic's attribute interface. This keeps the translation cost in the secondary backend only.

**Minimal tools**
`search_documents` and `get_metadata` are sufficient for Claude. A `calculate` tool is unnecessary as Claude handles financial arithmetic natively. A `compare_companies` tool is redundant with Claude — it's just two `search_documents` calls with different filters. Fewer tools means fewer wrong decisions by the agent. (The ADK comparison revealed that Gemini *does* need `compare_companies` because it can't reliably self-regulate multi-call sequences — see the ADK section.)

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

## Evaluation Suite

35 ground-truth Q&A pairs across 5 categories, scored by a second LLM call (LLM-as-judge) on three dimensions. Scores are attached to Langfuse traces for dashboard analysis. Run as pytest with threshold assertions.

**Results (35 test cases, 5 companies)**

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Relevance | 4.97/5 | ≥ 3.5 | ✅ Pass |
| Accuracy | 4.68/5 | ≥ 3.5 | ✅ Pass |
| Faithfulness | 4.27/5 | ≥ 3.5 | ✅ Pass |
| Retrieval Precision@3 | 0.97 | ≥ 0.7 | ✅ Pass |
| Hallucinations | 0 | 0 | ✅ Pass |

**Test categories**

| Category | Count | What it tests |
|----------|-------|---------------|
| Factual extraction | 16 | Single correct answer — revenue, net income, employee count |
| Cross-company comparison | 4 | Multiple retrievals with different filters, synthesised answer |
| Qualitative | 7 | Narrative retrieval — risk factors, strategy, business segments |
| Unanswerable | 4 | Company not in collection or wrong year — should refuse |
| Metadata lookup | 3 | Available companies, document types, years covered |

**How it works**

Each test case runs through the full agent pipeline, producing a Langfuse trace. A second Claude call (the judge) receives the query, expected answer, actual response, and retrieved chunks, then scores relevance, accuracy, and faithfulness on a 1–5 scale. Scores are attached to the trace via `langfuse.create_score()`. Retrieval precision checks whether chunks came from the expected company using `[Source:]` label matching. Metadata queries are excluded from faithfulness scoring because they use `get_metadata` (not `search_documents`), so empty chunks are by design.

**Known weakness:** Sony's annual report contains chart-heavy pages with encoded figures that pypdf flattens poorly. The eval suite catches this — Sony factual extraction queries score lower on accuracy than other companies. A future upgrade to pdfplumber or unstructured.io would address this.

## LLM Backend Abstraction

The LLM layer supports two backends, swappable via environment variable:

| Backend | Client | Model | Use case |
|---------|--------|-------|----------|
| `anthropic` (default) | `AnthropicClient` | Claude Sonnet | Production — highest quality |
| `openai` | `OpenAIClient` | Ollama (Qwen 2.5 14B, Mistral, etc.) | Air-gapped — data stays local |

```bash
# Default — Claude API
PYTHONPATH=. python scripts/agent_query.py

# Local model via Ollama
LLM_BACKEND=openai LOCAL_MODEL=qwen2.5:14b PYTHONPATH=. python scripts/agent_query.py
```

**Cloud vs local quality:** Claude Sonnet scores 4.68/5 accuracy on the eval suite with correct citations. Qwen 2.5 14B successfully calls tools and completes the agent loop, but produces lower quality answers — wrong figures, missed cross-company retrievals, occasional wrong-language responses. The quality gap is expected; local models serve air-gapped deployments where data isolation is the priority, not answer quality parity. In production on GPU servers, vLLM replaces Ollama with the same `OpenAIClient` code — only `base_url` changes.

## Google ADK Comparison

The same RAG pipeline rebuilt with Google's Agent Development Kit to compare framework patterns. Uses the same ChromaDB, same retrieval logic, same tools — only the agent orchestration and LLM differ.

```bash
# Run the ADK agent
PYTHONPATH=. python scripts/adk_query.py
```

**What's different from LangGraph:**

| | LangGraph | Google ADK |
|---|---|---|
| Agent definition | ~80 lines — StateGraph, nodes, edges, routing | ~30 lines — `Agent()` with tools and instruction |
| Tool-calling loop | Explicit — you wire the decide → call → return cycle | Implicit — ADK handles it internally |
| Tool format | Anthropic schemas (`input_schema`) | Auto-generated from function docstrings and type hints |
| Execution model | Synchronous — `graph.invoke()` blocks until done | Async — `runner.run_async()` streams events |
| Observability | Langfuse tracing at every layer | ADK's built-in tracing |

**Results (Gemini 3.5 Flash):**

| Query type | Result |
|---|---|
| Factual extraction (Apple revenue) | ✅ Correct — $416,161M with citations |
| Metadata lookup (available companies) | ✅ Correct — all 5 companies listed |
| Unanswerable (Tesla) | ✅ Correctly refused |
| Cross-company comparison | ⚠️ Intermittent — see findings below |

**What I found during integration:**

The ADK agent works for single-tool-call queries. Cross-company comparisons — where Claude makes two `search_documents` calls and synthesises — exposed three issues with Gemini's tool-calling behaviour:

1. **Verbose system prompts cause over-calling.** Reusing Claude's system prompt made Gemini call `search_documents` 4-6 times per company instead of once. A shorter, Gemini-specific instruction fixed this. The same prompt doesn't transfer across models — context engineering is model-specific.

2. **Gemini doesn't self-regulate tool usage.** Even with explicit instructions ("search once per company"), Gemini kept making redundant calls. Claude follows these constraints precisely. This led to adding a `compare_companies` tool that searches both companies internally in a single call — unnecessary for Claude, required for Gemini.

3. **Accumulated tool results cause API disconnects.** Multiple tool calls produce ~12,000+ chars of conversation history. Gemini's API disconnects during the final synthesis turn when context grows this large. Truncating tool output to 3,000 chars per call and reducing tool call count mitigates this. Testing over VPN (due to regional API restrictions) likely contributed to connection instability, as longer requests are more sensitive to VPN routing latency and timeouts.

**Conclusion:** ADK's simplicity (30 lines vs 80) comes with less control over the tool-calling loop. LangGraph's explicit graph gives you visibility into every decision and the ability to constrain agent behaviour precisely. For production financial analysis, LangGraph with Claude remains the primary choice. ADK demonstrates framework portability — the same tools and RAG pipeline work across both, with model-specific prompt tuning.

## Test Coverage

23 tests + 1 smoke test + 5 eval tests, all passing.

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

**Eval tests**

| Test | What it verifies |
|------|-----------------|
| `test_relevance_threshold` | Average relevance ≥ 3.5/5 |
| `test_accuracy_threshold` | Average accuracy ≥ 3.5/5 |
| `test_faithfulness_threshold` | Average faithfulness ≥ 3.5/5 (excluding metadata queries) |
| `test_precision_threshold` | Retrieval precision@3 ≥ 0.7 |
| `test_zero_hallucinations` | Zero responses with faithfulness < 3 (excluding metadata queries) |

## Observability

Every query is traced through Langfuse at three levels:
- **Retrieval** — query, results, latency
- **Context assembly** — formatted prompt
- **LLM calls** — prompt, completion, tokens

The API returns a `trace_id` with each response for direct lookup in the Langfuse dashboard. Evaluation scores (relevance, accuracy, faithfulness) are attached to traces via the Langfuse SDK, enabling filtered analysis of low-scoring queries.

## How to Run

**Prerequisites**
- Python 3.11+
- Anthropic API key (for Claude backend)
- Google API key (for ADK/Gemini comparison)
- Langfuse account (free tier, for tracing)
- Ollama (optional, for local model backend)

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
export GOOGLE_API_KEY="AIzaSy..."
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
# Unit and integration tests (no API key needed)
PYTHONPATH=. pytest tests/agent -m "not smoke"

# Smoke test (requires API key + ingested data)
PYTHONPATH=. pytest tests/agent/test_smoke.py -m smoke

# Evaluation suite (requires API key + ingested data, ~$3 per run)
PYTHONPATH=. pytest tests/evals -m eval -v -s

# Eval runner standalone (prints summary table)
PYTHONPATH=. python src/evals/runner.py

# ADK agent (requires GOOGLE_API_KEY + ingested data)
PYTHONPATH=. python scripts/adk_query.py
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
- pypdf struggles with chart-heavy PDFs (Sony annual report) — encoded figures are flattened, reducing accuracy on table/chart extraction queries
- Gemini API disconnects on multi-turn tool conversations — cross-company comparisons via ADK require the `compare_companies` workaround tool

**Planned extensions**
- Sliding window memory for conversational follow-up queries
- Reflexion/self-critique node — second LLM pass to verify answer is grounded in sources, critical for financial accuracy
- Hybrid search (vector + BM25 keyword) via reciprocal rank fusion
- Upgrade PDF extraction to pdfplumber or unstructured.io for layout-aware parsing
