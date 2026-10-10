from fastapi import FastAPI;
from pydantic import BaseModel;

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