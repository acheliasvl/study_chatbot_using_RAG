import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import config
from vector_db import VectorDB
from llm import generate_answer_stream, generate_hypothetical_questions
from document_processor import extract_text_and_metadata_from_pdf, chunk_text_with_context
import shutil

app = FastAPI()

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

CONVERSATIONS_FILE = os.path.join(config.BASE_DIR, "conversations.json")

def load_conversations() -> dict:
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_conversations(convs: dict):
    with open(CONVERSATIONS_FILE, "w") as f:
        json.dump(convs, f, indent=2)

db = VectorDB()

class ChatRequest(BaseModel):
    query: str
    conversation_id: str

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

class NewConversationRequest(BaseModel):
    conversation_id: str = None
    name: str = None

@app.post("/conversations/new")
async def new_conversation(request: NewConversationRequest = None):
    convs = load_conversations()
    conv_id   = (request.conversation_id if request and request.conversation_id else None) or str(uuid.uuid4())
    conv_name = (request.name if request and request.name else None) or f"Chat {len(convs)+1}"
    convs[conv_id] = {"name": conv_name, "messages": []}
    save_conversations(convs)
    return {"conversation_id": conv_id, "name": conv_name}

@app.get("/conversations")
async def list_conversations():
    convs = load_conversations()
    return [{"conversation_id": k, "name": v["name"]} for k, v in convs.items()]

@app.post("/chat")
async def chat(request: ChatRequest):
    relevant_chunks = db.search(request.query, n_results=3, conversation_id=request.conversation_id)
    if not relevant_chunks:
        prompt = f"Question: {request.query}\nAnswer:"
    else:
        prompt = build_prompt(request.query, relevant_chunks)

    convs = load_conversations()
    if request.conversation_id in convs:
        convs[request.conversation_id]["messages"].append({"role": "user", "content": request.query})
        save_conversations(convs)

    return StreamingResponse(generate_answer_stream(prompt), media_type="text/plain")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), conversation_id: str = Form(...)):
    conv_docs_dir = os.path.join(config.DOCS_DIR, conversation_id)
    os.makedirs(conv_docs_dir, exist_ok=True)
    file_path = os.path.join(conv_docs_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pages = extract_text_and_metadata_from_pdf(file_path)
    chunks = chunk_text_with_context(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    for chunk in chunks:
        chunk["hypothetical_questions"] = generate_hypothetical_questions(chunk["text"])
        chunk["metadata"]["conversation_id"] = conversation_id

    db.add_chunks(chunks, file.filename)

    return {"status": "success", "chunks": len(chunks)}

