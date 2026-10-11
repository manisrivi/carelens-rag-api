from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import mongo
from app.routes import ask, reports
from app.services.vector_store import ensure_vector_index, index_status


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_vector_index()
    except Exception as e:
        print(f"Warning: could not create vector index at startup: {e}")
    yield


app = FastAPI(title="CareLens RAG API", lifespan=lifespan)

app.include_router(reports.router)
app.include_router(ask.router)


@app.get("/health", tags=["health"])
def health():
    mongo.admin.command("ping")
    return {"status": "ok", "mongodb": "connected", "search_indexes": index_status()}