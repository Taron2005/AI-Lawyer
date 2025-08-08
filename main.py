from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag import KnowledgeGraph
from llm import generate_response
import logging
from fastapi.responses import FileResponse
from speech_api import tts_long_text

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

kg = KnowledgeGraph()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Step 1: Graph Query → top facts
        facts = kg.query(question, k=5)
        if not facts:
            return {"answer": "No relevant facts found in the knowledge graph."}

        # Step 2: Group facts by community
        community_groups = {}
        for f in facts:
            node = f["subject"]
            community = kg.graph.nodes[node].get("community", -1)
            community_groups.setdefault(community, []).append(f"{f['subject']} {f['relation']} {f['object']}")

        # Step 3: LLM summarization per community
        community_summaries = []
        for ctx in community_groups.values():
            summary = generate_response(question, ctx)
            community_summaries.append(summary)

        # Step 4: Global summarization from all summaries
        final_answer = generate_response(question, community_summaries)

        # Step 5: Text-to-Speech
        success, audio_or_err = tts_long_text(final_answer, base_filename="answer")
        if not success:
            raise Exception(audio_or_err)

        return {
            "answer": final_answer,
            "audio_files": [str(p) for p in audio_or_err],
            "sources": list(set(f["source"] for f in facts)),
            "snippets": [f"{f['subject']} {f['relation']} {f['object']}" for f in facts]
        }

    except Exception as e:
        logger.exception("Knowledge-graph query or generation failed.")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    raise HTTPException(status_code=501, detail="Upload → knowledge-graph ingestion not yet implemented.")

@app.get("/audio")
async def get_audio():
    return FileResponse("answer_0.wav", media_type="audio/wav", filename="answer_0.wav")
