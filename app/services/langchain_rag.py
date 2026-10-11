import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

from app.config import GEMINI_MODEL, VECTOR_INDEX_NAME, chunks_collection

api_key = os.environ["GEMINI_API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
)

vector_store = MongoDBAtlasVectorSearch(
    collection=chunks_collection,
    embedding=embeddings,
    index_name=VECTOR_INDEX_NAME,
    text_key="text",
    embedding_key="embedding",
)

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=0)

prompt = ChatPromptTemplate.from_template(
    """You are CareLens, an assistant that explains blood test reports in simple language.
Answer the question using ONLY the report excerpts below.
If the answer is not in the excerpts, say you could not find it in the report.
For each value you mention, always state the reference range and whether it is normal, low, or high.
Do not diagnose. If any value is abnormal, end by suggesting the user discuss it with a doctor.

Report excerpts:
{context}

Question: {question}
"""
)


def ask_with_langchain(question: str, report_id: str, top_k: int = 5) -> tuple[str, list[dict]]:
    retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k, "pre_filter": {"report_id": {"$eq": report_id}}}
    )
    docs = retriever.invoke(question)

    if not docs:
        return "I could not find this in your report.", []

    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = [
        {"chunk_index": doc.metadata.get("chunk_index"), "text": doc.page_content}
        for doc in docs
    ]
    return answer, sources