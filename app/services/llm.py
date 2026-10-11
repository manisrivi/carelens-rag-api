from app.config import GEMINI_MODEL, client

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
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text