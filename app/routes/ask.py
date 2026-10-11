from fastapi import APIRouter

from app.schemas import AskRequest
from app.services.agent import run_agent
from app.services.langchain_rag import ask_with_langchain
from app.services.llm import generate_answer
from app.services.vector_store import latest_report_id, search_chunks

router = APIRouter(tags=["ask"])


@router.post("/ask")
def ask(req: AskRequest):
    report_id = req.report_id or latest_report_id()

    if report_id is None:
        return {
            "question": req.question,
            "answer": "No report uploaded yet. Please upload a report first.",
            "sources": [],
        }

    results = search_chunks(req.question, report_id, req.top_k)

    if not results:
        return {
            "question": req.question,
            "report_id": report_id,
            "answer": "I could not find this in your report.",
            "sources": [],
        }

    answer = generate_answer(req.question, results)
    return {
        "question": req.question,
        "report_id": report_id,
        "answer": answer,
        "sources": results,
    }


@router.post("/ask-langchain")
def ask_langchain(req: AskRequest):
    report_id = req.report_id or latest_report_id()

    if report_id is None:
        return {
            "question": req.question,
            "answer": "No report uploaded yet. Please upload a report first.",
            "sources": [],
        }

    answer, sources = ask_with_langchain(req.question, report_id, req.top_k)
    return {
        "question": req.question,
        "report_id": report_id,
        "answer": answer,
        "sources": sources,
        "engine": "langchain",
    }


@router.post("/agent")
def agent(req: AskRequest):
    report_id = req.report_id or latest_report_id()

    if report_id is None:
        return {
            "question": req.question,
            "answer": "No report uploaded yet. Please upload a report first.",
            "sources": [],
        }

    result = run_agent(req.question, report_id)
    return {
        "question": req.question,
        "report_id": report_id,
        "route": result["route"],
        "answer": result["answer"],
        "sources": result["sources"],
        "engine": "langgraph",
    }