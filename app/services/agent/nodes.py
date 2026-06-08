import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from .state import AgentState
from .prompts import SUPERVISOR_PROMPT, CONVERSATIONAL_PROMPT, RAG_PROMPT, TRANSACTIONAL_PROMPT
from .tools.portfolio import navigate_website, retrieve_portfolio_info, execute_loan_prediction

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

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

async def rag_agent(state: AgentState):
    agent = llm.bind_tools([retrieve_portfolio_info])
    response = await agent.ainvoke([SystemMessage(content=RAG_PROMPT)] + state["messages"])
    return {"messages": [response]}

async def transactional_agent(state: AgentState):
    agent = llm.bind_tools([execute_loan_prediction])
    response = await agent.ainvoke([SystemMessage(content=TRANSACTIONAL_PROMPT)] + state["messages"])
    return {"messages": [response]}