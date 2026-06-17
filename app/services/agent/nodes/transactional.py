# app/services/agent/nodes/transactional.py

import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import AgentState
from ..prompts import TRANSACTIONAL_PROMPT
from ..tools.portfolio import execute_loan_prediction

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

async def transactional_agent(state: AgentState):
    task = state["plan"][0] if state.get("plan") else "Run loan prediction"
    logger.info(f"🟢 WORKER: Transactional Agent executing task: '{task}'")
    
    agent = llm.bind_tools([execute_loan_prediction])
    response = await agent.ainvoke([SystemMessage(content=TRANSACTIONAL_PROMPT), HumanMessage(content=task)])
    
    return {"past_steps": [(task, response.content)]}