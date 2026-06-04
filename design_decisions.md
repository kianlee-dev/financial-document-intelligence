# Financial Document Intelligence — Design Decisions

---

## 1. Ingestion Pipeline

### 1.1 Document Loading

**Options:**
- **Simple extraction** (pypdf, pdfplumber) — extract raw text, fast, minimal dependencies
- **Layout-aware extraction** (unstructured.io, Amazon Textract) — preserves tables, columns, headers
- **OCR-based** (pytesseract) — for scanned/image-based documents

**Considerations:** Documents are Apple 10-K (SEC filing), JPMorgan annual report, HSBC annual report, AIA annual report. These are digitally generated PDFs, contain tables, charts, and multi-column layouts.

**Design Decision:** pypdf

**Reasoning:** Handles data of interest (generated pdfs of 10-K (SEC filing) and annual report) well enough, table layouts are flattened so Layout-aware extraction is unnecessary as downstream chunking step converts everything to plaintext regardless. The pdfs are not scanned so OCR-based tools not applicable. 

**Future Work:** If evaluation shows poorly performing retrieval on table-heavy sections then might consider upgrading to pdfplumber or unstructured.io. 

---

### 1.2 Preprocessing

**Options:**
- **None** — raw text straight to chunking
- **Minimal cleaning** — remove repeated headers/footers, page numbers, table of contents
- **Section splitting** — split by document sections (e.g. "Risk Factors", "MD&A", "Financial Statements") before chunking
- **Metadata extraction** — programmatically pull company name, report year, section headers

**Considerations:** Financial reports have repeating headers/footers on every page, page numbers, and structured sections. Section splitting improves retrieval but assumes consistent document structure across different report formats.

**Design Decision:** Minimal cleaning — remove repeated headers, footers, and page numbers. 

**Reasoning:** Strings get embedded as chunks and pollute retrieval results without cleaning. Section splitting is skipped because documents span different formats — US 10-K, UK annual reports, Asian corporate reports. Assuming consistent section structure would break on at least one format.

**Future Work:** Programmatic metadata extraction (company name, section headers) requires format-specific parsers that would break across diverse document types and is deferred to future improvement. Basic metadata handled in the metadata attachment step.

---

### 1.3 Chunking Strategy

**Options:**
- **Fixed-size** (CharacterTextSplitter) — split at every N characters regardless of content
- **Recursive** (RecursiveCharacterTextSplitter) — split by paragraph → sentence → word hierarchy
- **Semantic** — split based on topic shifts detected by embedding similarity
- **Document-structure-aware** — split by headings/sections first, then chunk within

**Parameters to decide:**
- Chunk size: 250 / 500 / 750 / 1000 tokens
- Overlap: 0 / 50 / 100 / 200 tokens

**Considerations:** Financial text mixes narrative paragraphs (MD&A, Risk Factors) with dense tabular data (financial statements). A chunk size that works for narrative may be wrong for tables.

**Design Decision:**  RecursiveCharacterTextSplitter, 500 tokens ~2000 characters chunk size, 400 character overlap (20% of chunk size)

**Reasoning:**  RecursiveCharacterTextSplitter splits by hierarchy which preserves semantic coherence in structured financial documents. 2000 characters is roughly 1-2 paragraphs, it is small enough for topically focused embeddings that match specific queries, large enough to be self-contained with sufficient context. 400-characters overlap (20% of chunk size) prevents information loss at chunk boundaries. If a key sentence falls exactly at a split point, it would otherwise be cut across two chunks with neither chunk containing the full statement. Overlap ensures the sentence appears intact in at least one chunk.

**Future work:** Validate chunk size empirically by running the eval suite against 1000/2000/4000 characters configurations and comparing retrieval precision metrics.

---

### 1.4 Metadata Attachment

**Options:**
- **Minimal** — source filename, page number
- **Standard** — source filename, page number, company name, report year, document type
- **Rich** — all of the above + section name, chunk summary, extracted keywords
- **Derived** — LLM-generated summary per chunk (expensive, slow, but high quality)

**Considerations:** Metadata enables filtered retrieval at query time. Richer metadata = better retrieval but more ingestion logic.

**Design Decision:** Standard — source filename, page number, company name, report year, document type

**Reasoning:** Agent needs filtered retrieval by company name and must cite source file, date, and page number. All derived from information already available at load time — filename parsing gives company name, year, and document type; pypdf provides page numbers during extraction. No custom parsers needed.

**Future work:** Rich/derived metadata deferred as future improvement.

---

### 1.5 Embedding Model

**Options:**
- **all-MiniLM-L6-v2** — 384 dims, fast, free, local, no API cost. Standard prototyping choice.
- **BGE / E5 / GTE models** — higher quality embeddings, larger models, slower, still free/local
- **OpenAI text-embedding-ada-002** — cloud, paid per call, high quality
- **Cohere embed** — cloud, paid, multilingual strength

**Considerations:** English-language financial reports. Prototype vs production tradeoff. Local models have zero marginal cost. Cloud models add latency and cost but may improve retrieval quality.

**Design Decision:** all-MiniLM-L6-v2

**Reasoning:** all-MiniLM-L6-v2 is lightweight, free, local, well-documented, and sufficient for prototyping. Retrieval quality at this stage is bottlenecked by chunking and metadata strategy, not embedding model quality.

**Future work:** If eval results show poor retrieval precision after tuning other parameters, upgrade to BGE or E5, it is just a matter of single-line config change.

---

### 1.6 Vector Store Configuration

**Options:**
- **Single collection** — all documents in one collection. Simple. Supports cross-document queries natively.
- **Per-document collection** — one collection per PDF. Isolation. Cross-document queries require multiple searches and merging.
- **Per-company collection** — one collection per company. Middle ground.

**Considerations:** Needs to answer both single-document questions ("What was Sony's revenue?") and cross-document questions ("Compare X and Y 's revenue").

**Design Decision:** Single collection — all documents in one collection.

**Reasoning:** Single collection supports cross-document queries natively (e.g. "compare Apple vs HSBC risk factors"). Single-document queries are handled by metadata filtering e.g. company_name=Apple which narrows the search space without needing separate collections. Simpler to maintain than per-document or per-company collections with no loss of functionality.

---

### 1.7 Indexing Strategy

**Options:**
- **Flat** — one embedding per chunk, stored directly. Simple, standard.
- **Parent-child** — embed small chunks for retrieval precision, but return the larger parent chunk for more context in the prompt.
- **Hypothetical questions (HyDE)** — generate questions each chunk could answer, embed the questions instead. Query matches question-to-question rather than question-to-passage.
- **Multi-vector** — store multiple representations per chunk (summary embedding + full text embedding)

**Considerations:** Flat is the standard starting point. Advanced strategies are interview talking points for "what would you improve?" but add complexity.

**Design Decision:** Flat — one embedding per chunk, stored directly.

**Reasoning:** Standard approach, sufficient for prototyping. Each chunk gets one embedding and is retrieved as-is

**Future work:** Parent-child indexing — embed smaller chunks (~200 tokens) for more precise vector matching, but retrieve the larger parent chunk (~1000 tokens) to give the LLM more surrounding context. Trades storage and complexity for better retrieval precision paired with richer answer context.

---

## 2. Retrieval Strategy

### 2.1 Search Method

**Options:**
- **Pure vector search** — cosine similarity between query embedding and stored embeddings
- **Hybrid search** — vector search + BM25 keyword search, results merged (reciprocal rank fusion)
- **Filtered search** — vector search with metadata filters applied first (e.g. company=Apple)

**Considerations:** Pure vector search can miss exact keyword matches — the embedding may not capture the acronym precisely. Hybrid search catches what embeddings miss.

**Design Decision:** Pure vector search with metadata filtering

**Reasoning:**  Cosine similarity handles semantic queries. Metadata filters (company name, report year) narrow the search space when the query targets a specific document. ChromaDB supports metadata filtering natively. ChromaDB doesn't support hybrid search natively, so adding it means building and maintaining a parallel keyword search system. It has marginal benefit at the scale of this project.

**Future work:** Adding BM25 keyword search alongside vector search using reciprocal rank fusion to catch exact term matches that embeddings miss (Hybrid Search).

---

### 2.2 Top-K Selection

**Options:**
- **k=1** — single most relevant chunk. Minimal noise, risk of missing information.
- **k=3** — standard. Balance of relevance and coverage.
- **k=5** — broader coverage, more noise, higher token cost in prompt.
- **Dynamic k** — adjust based on query type or confidence scores.

**Considerations:** More chunks = more context for the LLM but also more noise and token cost.

**Design Decision:** k=3

**Reasoning:**  k=1 risks missing relevant info. k=3 is ~1500 tokens of context which should be enough coverage without dominating the prompt. k=5 adds noise with diminishing returns. Dynamic k would be best but will be overkill for version 1.

---

### 2.3 Similarity Metric

**Options:**
- **Cosine similarity** — measures angle between vectors. Standard for normalized embeddings.
- **Euclidean distance** — measures absolute distance. Sensitive to magnitude.
- **Dot product** — fast, works when embeddings are normalized.

**Design Decision:** Cosine similarity

**Reasoning:** Industry default. sentence-transformers produces normalised vectors so cosine and dot product are equivalent. No reason to deviate.

---

