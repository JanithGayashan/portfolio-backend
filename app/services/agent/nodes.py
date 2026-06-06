import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from typing import Literal

from .state import AgentState
from .prompts import SUPERVISOR_PROMPT, CONVERSATIONAL_PROMPT, RAG_PROMPT, TRANSACTIONAL_PROMPT
# Notice it says .tools.portfolio instead of .tools.portfolio_tools
from .tools.portfolio import navigate_website, retrieve_portfolio_info, execute_loan_prediction

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"))

class RouteSchema(BaseModel):
    next_node: Literal["rag_agent", "transactional_agent", "conversational_agent"] = Field(...)

supervisor_llm = llm.with_structured_output(RouteSchema)

async def supervisor_node(state: AgentState):
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    decision = await supervisor_llm.ainvoke(messages)
    return {"next_agent": decision.next_node}

async def conversational_agent(state: AgentState):
    agent = llm.bind_tools([navigate_website])
    response = await agent.ainvoke([SystemMessage(content=CONVERSATIONAL_PROMPT)] + state["messages"])
    return {"messages": [response]}

# ... (Include rag_agent and transactional_agent exactly like this) ...