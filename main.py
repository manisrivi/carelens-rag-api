import io

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader

app = FastAPI();

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
    text = "/n".join([page.extract_text() or "" for page in reader.pages])
    return {
        "filename": file.filename, 
        "size": len(reader.pages),
        "characters": len(text),
        "preview": text[:500]
        }