import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import AgentState
from ..prompts import SUPERVISOR_PROMPT

logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

class RouteDecision(BaseModel):
    # Only picking ONE primary intent now
    detected_intent: Literal["VECTOR_SEARCH", "LOAN_CALCULATION", "CONVERSATIONAL"] = Field(...)

async def supervisor_node(state: AgentState):
    logger.info("🧠 ACTIVATED: Supervisor Node -> Classifying intent...")
    
    supervisor = llm.with_structured_output(RouteDecision)
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"] + [HumanMessage(content=f"Current User Message: {state['current_user_input']}")]
    
    try:
        decision = await supervisor.ainvoke(messages)
        intent = decision.detected_intent
    except Exception as e:
        logger.error(f"⚠️ SUPERVISOR FAILED: {e}. Falling back to conversational.")
        intent = "CONVERSATIONAL"
        
    logger.info(f"📊 INTENT DETECTED: {intent}")
    
    routing_table = {
        "VECTOR_SEARCH": "rag_agent",
        "LOAN_CALCULATION": "transactional_agent",
        "CONVERSATIONAL": "conversational_agent"
    }
    
    target_agent = routing_table.get(intent, "conversational_agent")
    logger.info(f"🔀 ROUTING TO: {target_agent}")
    
    # We save the target agent in the state so the graph knows where to go
    return {"target_agent": target_agent}