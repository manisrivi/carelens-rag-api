import io
import os

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai
from pymongo import MongoClient

load_dotenv()

client = genai.Client() 

mongo = MongoClient(os.environ["MONGODB_URI"])
db = mongo["carelens_py"]
chunk_collections = db["report_chunks"]

app = FastAPI();

def embed_texts(text: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return [e.values for e in result.embeddings]

def clean_text(text:str) -> str:
    lines = text.split("\n")
    cleaned = [" ".join(line.split()) for line in lines]
    return "\n".join([line for line in cleaned if line])

def chunk_text(text:str, size: int = 500, overlap: int = 100) -> list[str]:
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]

@app.get("/health")
def health():
    mongo.admin.command("ping")
    return {"status": "ok", "mongodb": "connected"}

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/ask")
def ask(req: AskRequest):
    return { "question": req.question, "top_k": req.top_k }

@app.post("/reports/upload")
async def upload_report(file: UploadFile):
    content = await file.read()

    reader = PdfReader(io.BytesIO(content))
    raw_text = "/n".join([page.extract_text() or "" for page in reader.pages])
    text = clean_text(raw_text)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)
    return {
        "filename": file.filename, 
        "size": len(reader.pages),
        "raw_characters": len(raw_text),
        "characters": len(text),
        "preview": text[:500],
        "chunks": chunks,
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "embedding_dimensions": len(embeddings[0]) 
        }