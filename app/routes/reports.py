from fastapi import APIRouter, Depends, UploadFile

from app.security import require_api_key
from app.services.embeddings import embed_texts
from app.services.pdf import chunk_text, clean_text, extract_text
from app.services.vector_store import save_chunks

router = APIRouter(prefix="/reports", tags=["reports"], dependencies=[Depends(require_api_key)])


@router.post("/upload")
def upload_report(file: UploadFile):
    content = file.file.read()

    raw_text, pages = extract_text(content)
    text = clean_text(raw_text)
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)
    report_id = save_chunks(file.filename, chunks, embeddings)

    return {
        "report_id": report_id,
        "filename": file.filename,
        "pages": pages,
        "chunk_count": len(chunks),
    }