from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.agent.tools import TOOLS
from app.config import settings
from app.observability import AgentCallbackHandler, logger


SYSTEM_PROMPT = """You are a BPSS compliance analyst. Answer questions about candidate
screening records using only the tools provided. Never make up evidence.

Tools:
- search_pdfs    -> policy documents, SOP, adjudication register
- query_csv      -> tracker data, employment history, document inventory
- extract_docx   -> candidate packs, analyst notes, email approvals

For any candidate question, always cross-check at least query_csv + extract_docx.
Cite every source file. If data is missing, say so — don't guess.
Flag contradictions between sources explicitly.

End every response with a Sources section listing the files you used."""


def _get_llm():
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        temperature=0,
        max_tokens=2048,
    )


def run_query(question: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
    logger.info("agent_query", question=question[:300])

    cb = AgentCallbackHandler()
    agent = create_react_agent(model=_get_llm(), tools=TOOLS, prompt=SYSTEM_PROMPT)

    messages = list(chat_history or [])
    messages.append(HumanMessage(content=question))

    try:
        result = agent.invoke(
            {"messages": messages},
            config={"callbacks": [cb], "recursion_limit": 20},
        )
        # pull the last non-empty AI message as the final answer
        answer = next(
            (m.content for m in reversed(result["messages"])
             if isinstance(m, AIMessage) and m.content),
            "No answer generated."
        )
    except Exception as e:
        logger.error("agent_error", error=str(e))
        answer = f"Agent error: {e}"

    trace = cb.get_trace()
    return {
        "answer": answer,
        "sources": list(dict.fromkeys(trace.get("sources_cited", []))),
        "trace": trace,
    }
