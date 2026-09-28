import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import config
from vector_db import VectorDB
from llm import generate_answer_stream
from document_processor import extract_text_from_pdf, chunk_text
import shutil

app = FastAPI()

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

db = VectorDB()

class ChatRequest(BaseModel):
    query: str

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context_str = "\n\n".join(context_chunks)
    return (
        f"You are a helpful study assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

@app.get("/", response_class=HTMLResponse)
async def get_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat(request: ChatRequest):
    relevant_chunks = db.search(request.query, n_results=3)
    if not relevant_chunks:
        prompt = f"Question: {request.query}\nAnswer:"
    else:
        prompt = build_prompt(request.query, relevant_chunks)
    
    return StreamingResponse(generate_answer_stream(prompt), media_type="text/plain")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(config.DOCS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    db.add_chunks(chunks, file.filename)
    
    return {"status": "success", "chunks": len(chunks)}
