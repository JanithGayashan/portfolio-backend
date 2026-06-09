import os
import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from .state import AgentState
from .prompts import SUPERVISOR_PROMPT, CONVERSATIONAL_PROMPT, RAG_PROMPT, TRANSACTIONAL_PROMPT, OUT_OF_BOUNDS_PROMPT
from .tools.portfolio import navigate_website, retrieve_portfolio_info, execute_loan_prediction

# Configure the terminal logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

class RouteSchema(BaseModel):
    next_node: Literal["rag_agent", "transactional_agent", "conversational_agent", "OUT_OF_BOUNDS"] = Field(...)

supervisor_llm = llm.with_structured_output(RouteSchema)

async def supervisor_node(state: AgentState):
    logger.info("🟢 ACTIVATED: Supervisor Node -> Analyzing user intent...")
    
    # --- 💾 NEW MONGODB LOGGING TRACE ---
    # state["messages"] contains the history loaded from MongoDB + the current message
    total_messages = len(state["messages"])
    logger.info(f"💾 MONGODB MEMORY: Loaded {total_messages} total message(s) from database.")
    
    # If there is history (more than just the 1 current user message), log the last 3
    if total_messages > 1:
        logger.info("🕒 HISTORY TRACE (Previous 3 messages):")
        
        # Python slicing: get up to 3 previous messages (excluding the current new one)
        previous_messages = state["messages"][-4:-1] 
        
        for msg in previous_messages:
            # Check who sent it (User or AI)
            role = "USER" if msg.type == "user" else ("AI" if msg.type == "ai" else "TOOL")
            
            # Truncate the text so it doesn't flood the terminal if it's super long
            content = msg.content
            if len(content) > 75:
                content = content[:75] + "..."
                
            logger.info(f"   -> [{role}]: {content}")
    else:
        logger.info("🕒 HISTORY TRACE: 0 previous messages. This is a brand new conversation.")
    # ------------------------------------

    # Now proceed with the standard routing logic
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    decision = await supervisor_llm.ainvoke(messages)
    
    logger.info(f"🔀 ROUTING: Supervisor decided to route to -> [{decision.next_node}]")
    return {"next_agent": decision.next_node}

async def conversational_agent(state: AgentState):
    logger.info("🟢 ACTIVATED: Conversational Agent -> Processing small talk/navigation...")
    agent = llm.bind_tools([navigate_website])
    response = await agent.ainvoke([SystemMessage(content=CONVERSATIONAL_PROMPT)] + state["messages"])
    return {"messages": [response]}

async def rag_agent(state: AgentState):
    logger.info("🟢 ACTIVATED: RAG Agent -> Preparing to search portfolio context...")
    agent = llm.bind_tools([retrieve_portfolio_info])
    response = await agent.ainvoke([SystemMessage(content=RAG_PROMPT)] + state["messages"])
    return {"messages": [response]}

async def transactional_agent(state: AgentState):
    logger.info("🟢 ACTIVATED: Transactional Agent -> Preparing loan prediction model...")
    agent = llm.bind_tools([execute_loan_prediction])
    response = await agent.ainvoke([SystemMessage(content=TRANSACTIONAL_PROMPT)] + state["messages"])
    return {"messages": [response]}

async def out_of_bounds_node(state: AgentState):
    """Dynamically handles greetings or off-topic rejections."""
    logger.info("🛑 ACTIVATED: Out of Bounds Node -> Handling rejected off-topic query or greeting...")
    # We do not bind tools here because this agent is never allowed to execute actions
    messages = [SystemMessage(content=OUT_OF_BOUNDS_PROMPT)] + state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}