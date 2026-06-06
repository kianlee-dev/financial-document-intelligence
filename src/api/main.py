import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from src.agent import run
from src.agent.tools import get_documents_list, build_filters, get_store
from src.ingestion.loader import load_single_document
from src.ingestion.chunker import chunk_documents

app = FastAPI(title="Financial Document Intelligence API")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    trace_id: str

class DeleteRequest(BaseModel):
    company_name: str
    report_year: int | None = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Send a question, get an answer with citations."""
    result = run(request.query)
    return QueryResponse(response=result["response"], trace_id=result["trace_id"])

@app.post("/documents", status_code=201)
def upload_document(
    file: UploadFile = File(...),
    company_name: str = Form(...),
    report_year: int = Form(...),
    document_type: str = Form(...),
):
    """Upload a PDF with metadata for ingestion."""

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = Path(tmp.name)
    metadata = {
        "company_name": company_name,
        "report_year": report_year,
        "document_type": document_type,
    }
    pages = load_single_document(tmp_path, metadata)
    chunks = chunk_documents(pages)
    store = get_store()
    store.add_documents(chunks)
    tmp_path.unlink()
    return {"status": "ingested", "chunks_created": len(chunks)}

@app.get("/documents")
def list_documents():
    """List all available documents in the collection."""
    docs = get_documents_list()
    return {"documents": docs}

@app.delete("/documents", status_code=200)
def delete_document(request: DeleteRequest):
    """Remove a document's chunks by metadata filter."""
    filters = build_filters(request.company_name, request.report_year)
    store = get_store()
    store.delete_documents(filters)
    return {"status": "deleted", "deleted": True}