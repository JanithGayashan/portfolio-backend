# app/services/agent/nodes/conversational.py

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import AgentState
from ..prompts import CONVERSATIONAL_PROMPT

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

async def conversational_agent(state: AgentState):
    task = state["plan"][0] if state.get("plan") else "Respond to small talk"
    logger.info(f"🟢 WORKER: Conversational Agent executing task: '{task}'")
    
    response = await llm.ainvoke([SystemMessage(content=CONVERSATIONAL_PROMPT), HumanMessage(content=task)])
    return {"past_steps": [(task, response.content)]}