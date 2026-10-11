import uuid

from pymongo.operations import SearchIndexModel

from app.config import VECTOR_INDEX_NAME, chunks_collection, db
from app.services.embeddings import embed_texts


def ensure_vector_index():
    if chunks_collection.name not in db.list_collection_names():
        db.create_collection(chunks_collection.name)

    existing = [index["name"] for index in chunks_collection.list_search_indexes()]
    if VECTOR_INDEX_NAME in existing:
        return

    model = SearchIndexModel(
        name=VECTOR_INDEX_NAME,
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


def index_status() -> list[dict]:
    return [
        {
            "name": index["name"],
            "status": index.get("status"),
            "queryable": index.get("queryable"),
        }
        for index in chunks_collection.list_search_indexes()
    ]


def save_chunks(filename: str, chunks: list[str], embeddings: list[list[float]]) -> str:
    report_id = str(uuid.uuid4())
    docs = [
        {
            "report_id": report_id,
            "filename": filename,
            "chunk_index": i,
            "text": chunk,
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    chunks_collection.insert_many(docs)
    ensure_vector_index()
    return report_id


def search_chunks(question: str, report_id: str, top_k: int = 5) -> list[dict]:
    query_embedding = embed_texts([question])[0]
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
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


def latest_report_id() -> str | None:
    doc = chunks_collection.find_one(sort=[("_id", -1)], projection={"report_id": 1})
    return doc["report_id"] if doc else None