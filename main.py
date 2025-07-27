from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag import RAGEngine  # must be class-based version
from llm import generate_response
import fitz  # PyMuPDF
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

rag = RAGEngine()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        retrieved = rag.retrieve(question, k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {e}")

    context = [chunk["text"] for chunk in retrieved]
    answer = generate_response(question, context)

    return {
        "answer": answer,
        "sources": [chunk["metadata"] for chunk in retrieved],
        "snippets": context
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    if not contents or len(contents) < 100:
        raise HTTPException(status_code=400, detail="Uploaded file is too small or empty.")

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF file: {e}")

    file_chunks = []
    for i, page in enumerate(doc):
        try:
            text = page.get_text().strip()
            if text:
                # Optional: Split long pages into paragraph chunks if needed
                paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
                file_chunks.extend(paragraphs)
        except Exception as e:
            logger.warning(f"Failed to extract page {i}: {e}")

    if not file_chunks:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")

    before = len(rag.texts)
    rag.add_to_index(file.filename, file_chunks)
    added = len(rag.texts) - before

    return {
        "message": f"File '{file.filename}' uploaded and indexed.",
        "chunks_added": added,
        "total_chunks": len(rag.texts)
    }
