import logging
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from ..state import AgentState
from ..tools.portfolio import retrieve_portfolio_info

logger = logging.getLogger(__name__)
rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([retrieve_portfolio_info])

# WE MOVED THE FRONTEND SCHEMA HERE!
class FinalResponseSchema(BaseModel):
    reply: str = Field(description="A friendly, concise answer (Max 3 sentences).")
    ui_action: Literal["NONE", "PROMPT_USER", "AUTO_NAVIGATE"] = Field(...)
    target_route: Optional[str] = Field(description="e.g., '/projects', '/experience'")

# We give the RAG agent instructions on how to format the UI directly
RAG_PROMPT = """You are the Portfolio Expert. Search the database to answer the user's query.
Write a friendly, concise response (Max 3 sentences).

Determine UI Navigation:
- 'NONE': Normal chat.
- 'PROMPT_USER': If you explain a project or skill, ask if they want to visit that page.
- 'AUTO_NAVIGATE': If the user explicitly commanded "Go to the page"."""

async def rag_agent(state: AgentState):
    logger.info("🟢 WORKER: RAG Agent executing...")
    
    messages = [SystemMessage(content=RAG_PROMPT)] + state["messages"] + [HumanMessage(content=state["current_user_input"])]
    response = await rag_llm.ainvoke(messages)
    
    # 1. Execute Tool if needed
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_output = retrieve_portfolio_info.invoke(tool_call["args"]["query"])
        tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"], name=tool_call["name"])
        
        # Give the data back to the LLM
        messages.extend([response, tool_message])
    else:
        messages.append(response)

    # 2. Force the LLM to output the clean JSON for the frontend
    formatter = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).with_structured_output(FinalResponseSchema)
    final_output = await formatter.ainvoke(messages)
    
    active_target = "rag_agent" if final_output.ui_action == "PROMPT_USER" else None
    
    logger.info(f"✅ RAG COMPLETE. Saving JSON to final_response.")
    return {
        "final_response": final_output.model_dump(),
        "active_dialogue_target": active_target
    }