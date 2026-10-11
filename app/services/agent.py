from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from app.services.langchain_rag import ask_with_langchain, llm


class AgentState(TypedDict):
    question: str
    report_id: str
    route: str
    answer: str
    sources: list


classify_prompt = ChatPromptTemplate.from_template(
    """Classify the user's message for a blood report assistant.
Reply with exactly one word: report, chitchat, or unclear.

- report: names a specific test or value, or asks about the whole report
- chitchat: a greeting, thanks, or small talk
- unclear: refers to "it", "this", or "that" without saying which test

Examples:
"What is my hemoglobin?" -> report
"Summarize my report" -> report
"hi" -> chitchat
"thanks!" -> chitchat
"is it okay?" -> unclear
"what does this mean?" -> unclear

Message: {question}"""
)

chitchat_prompt = ChatPromptTemplate.from_template(
    """You are CareLens, a friendly assistant for blood test reports.
Reply briefly and warmly to this message, and mention you can answer questions about their report.

Message: {question}"""
)


# ---------- Nodes (each step of the agent) ----------

def classify(state: AgentState) -> dict:
    chain = classify_prompt | llm | StrOutputParser()
    route = chain.invoke({"question": state["question"]}).strip().lower()
    if route not in ("report", "chitchat", "unclear"):
        route = "report"
    return {"route": route}


def answer_from_report(state: AgentState) -> dict:
    answer, sources = ask_with_langchain(state["question"], state["report_id"])
    return {"answer": answer, "sources": sources}


def chitchat(state: AgentState) -> dict:
    chain = chitchat_prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": state["question"]})
    return {"answer": answer, "sources": []}


def clarify(state: AgentState) -> dict:
    answer = (
        "Could you tell me which test or value you mean? "
        "For example: hemoglobin, platelets, or WBC count."
    )
    return {"answer": answer, "sources": []}


def pick_route(state: AgentState) -> str:
    return state["route"]


# ---------- The map (graph) ----------

graph = StateGraph(AgentState)

graph.add_node("classify", classify)
graph.add_node("answer_from_report", answer_from_report)
graph.add_node("chitchat", chitchat)
graph.add_node("clarify", clarify)

graph.set_entry_point("classify")

graph.add_conditional_edges(
    "classify",
    pick_route,
    {
        "report": "answer_from_report",
        "chitchat": "chitchat",
        "unclear": "clarify",
    },
)

graph.add_edge("answer_from_report", END)
graph.add_edge("chitchat", END)
graph.add_edge("clarify", END)

agent = graph.compile()


def run_agent(question: str, report_id: str) -> dict:
    result = agent.invoke({"question": question, "report_id": report_id})
    return {
        "route": result["route"],
        "answer": result["answer"],
        "sources": result.get("sources", []),
    }