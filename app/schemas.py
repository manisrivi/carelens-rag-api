from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    report_id: str | None = None
    top_k: int = 5