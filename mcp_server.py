from mcp.server.mcpserver import MCPServer

from app.config import chunks_collection
from app.services.llm import generate_answer
from app.services.vector_store import latest_report_id, search_chunks

mcp = MCPServer("CareLens")


@mcp.tool()
def ask_report(question: str) -> str:
    """Answer a question about the user's latest blood test report,
    such as a lab value, its reference range, or which results are abnormal."""
    report_id = latest_report_id()
    if report_id is None:
        return "No report uploaded yet."

    results = search_chunks(question, report_id)
    if not results:
        return "I could not find this in the report."

    return generate_answer(question, results)


@mcp.tool()
def latest_report_info() -> str:
    """Get basic info about the user's latest uploaded blood test report."""
    report_id = latest_report_id()
    if report_id is None:
        return "No report uploaded yet."

    doc = chunks_collection.find_one({"report_id": report_id}, projection={"filename": 1})
    count = chunks_collection.count_documents({"report_id": report_id})
    return f"Latest report: {doc['filename']} (report ID {report_id}, {count} chunks)."


if __name__ == "__main__":
    mcp.run()