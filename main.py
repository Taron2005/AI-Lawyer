from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag import RAGEngine
from llm import generate_response
import fitz
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag = RAGEngine()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        retrieved = rag.retrieve(request.question.strip(), k=3)
        context = [item["text"] for item in retrieved]
        answer = generate_response(request.question.strip(), context)
        return {
            "answer": answer,
            "sources": [item["metadata"] for item in retrieved],
            "snippets": context
        }
    except Exception as e:
        logger.exception("RAG retrieval failed.")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    contents = await file.read()
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="File too small or empty.")

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
    except Exception as e:
        logger.exception("Failed to open uploaded PDF.")
        raise HTTPException(status_code=400, detail="Unable to process uploaded PDF.")

    chunks = []
    for page in doc:
        try:
            text = page.get_text("text")
            for para in text.split('\n\n'):
                if len(para.strip()) > 50:
                    chunks.append(para.strip())
        except Exception as e:
            logger.warning(f"Page read failed: {e}")

    if not chunks:
        raise HTTPException(status_code=400, detail="No readable content found.")

    before = len(rag.texts)
    rag.add_to_index(file.filename, chunks)
    after = len(rag.texts)

    return {
        "message": f"{file.filename} indexed.",
        "new_chunks": after - before,
        "total_chunks": after
    }
