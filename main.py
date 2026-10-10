import io

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

app = FastAPI();

def clean_text(text:str) -> str:
    lines = text.split("\n")
    cleaned = [" ".join(line.split()) for line in lines]
    return "\n".join([line for line in cleaned if line])

def chunk_text(text:str, size: int = 500, overlap: int = 100) -> list[str]:
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]

@app.get("/health")
def health():
    return {"status": "ok"}

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
    return {
        "filename": file.filename, 
        "size": len(reader.pages),
        "raw_characters": len(raw_text),
        "characters": len(text),
        "preview": text[:500],
        "chunks": chunks
        }