import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
from llm import generate_with_groq
from rag_manager import get_rag_manager
from session_manager import get_session_manager
from rag_utils import extract_text_from_file, smart_chunk_text

app = FastAPI(title="AI Lawyer API", version="5.1.0") # Version Bump

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    get_rag_manager()
    get_session_manager()

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    chat_history: List[Dict[str, str]] = []

class SessionRequest(BaseModel):
    session_id: str

@app.post("/ask", tags=["AI"])
def ask_question(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        rag_manager = get_rag_manager()
        session_manager = get_session_manager()

        # Retrieve now returns a list of dictionaries with source info
        retrieved_chunks = rag_manager.retrieve(question, top_k=10)
        
        temp_chunks = []
        if req.session_id:
            temp_chunks = session_manager.get_temp_chunks(req.session_id)

        answer = generate_with_groq(
            question, 
            retrieved_chunks=retrieved_chunks,
            chat_history=req.chat_history,
            temp_chunks=temp_chunks
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-permanent", tags=["Knowledge Base"])
async def upload_document_permanent(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    try:
        rag_manager = get_rag_manager()
        content = await file.read()
        # The manager now handles associating the filename with the content
        rag_manager.add_document(content, file.filename)
        return {
            "filename": file.filename,
            "message": "File processed and permanently added.",
            "vector_count": rag_manager.index.ntotal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-temp", tags=["Knowledge Base"])
async def upload_document_temp(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    try:
        session_manager = get_session_manager()
        content = await file.read()
        text = extract_text_from_file(content, file.filename)
        chunks = smart_chunk_text(text)
        session_id = str(uuid.uuid4())
        session_manager.add_temp_chunks(session_id, chunks)
        return {
            "filename": file.filename,
            "message": "File is ready for this session.",
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-session", tags=["Knowledge Base"])
def clear_session(req: SessionRequest):
    try:
        session_manager = get_session_manager()
        session_manager.clear_session(req.session_id)
        return {"message": "Session cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))