import io
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader
from google import genai
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()

client = genai.Client() 

mongo = MongoClient(os.environ["MONGODB_URI"])
db = mongo["carelens_py"]
chunks_collection = db["report_chunks"]

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
    report_id: str
    top_k: int = 5


def generate_answer(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join([c["text"] for c in chunks])

    prompt = f"""You are CareLens, an assistant that explains blood test reports in simple language.
Answer the question using ONLY the report excerpts below.
If the answer is not in the excerpts, say you could not find it in the report.
Do not diagnose. If a value is abnormal, suggest discussing it with a doctor.

Report excerpts:
{context}

Question: {question}
"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
    )
    return response.text

@app.post("/ask")
def ask(req: AskRequest):
    results = search_chunks(req.question, req.report_id, req.top_k)
    answer = generate_answer(req.question, results)
    return {"question": req.question, "answer": answer, "sources": results}
def ensure_vector_index():
    existing = [index["name"] for index in chunks_collection.list_search_indexes()]
    if "vector_index" in existing:
        return

    model = SearchIndexModel(
        name="vector_index",
        type="vectorSearch",
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 3072,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "report_id"},
            ]
        },
    )
    chunks_collection.create_search_index(model=model)
ensure_vector_index()

def search_chunks(question: str, report_id: str, top_k: int = 5) -> list[dict]:
    query_embedding = embed_texts([question])[0]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": top_k,
                "filter": {"report_id": report_id},
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "chunk_index": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(chunks_collection.aggregate(pipeline))

@app.post("/reports/upload")
async def upload_report(file: UploadFile):
    content = await file.read()

    reader = PdfReader(io.BytesIO(content))
    raw_text = "/n".join([page.extract_text() or "" for page in reader.pages])
    text = clean_text(raw_text)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    report_id = str(uuid.uuid4())

    docs = [
        {
            "report_id": report_id,
            "filename": file.filename,
            "chunk_index": i,
            "text": chunk,
            "embedding": embedding
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    chunks_collection.insert_many(docs)

    return {
        "report_id": report_id,
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