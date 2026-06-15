import os
import logging
from typing import Literal, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage

from .state import AgentState
from .prompts import PLANNER_PROMPT, REPLANNER_PROMPT, CONVERSATIONAL_PROMPT, RAG_PROMPT, TRANSACTIONAL_PROMPT
from .tools.portfolio import retrieve_portfolio_info, execute_loan_prediction

# Configure the terminal logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([retrieve_portfolio_info])

# --- PYDANTIC SCHEMAS ---
class Plan(BaseModel):
    steps: List[str] = Field(description="Sequential steps to complete the user's request.")

class FinalResponseSchema(BaseModel):
    reply: str = Field(description="The final synthesized conversational reply to the user.")
    ui_action: Literal["NONE", "PROMPT_USER", "AUTO_NAVIGATE"] = Field(...)
    target_route: Optional[str] = Field(description="e.g., '/projects', '/experience', '/education', '/contact'")

class Act(BaseModel):
    action: Literal["update_plan", "finish"] = Field(...)
    final_response: Optional[FinalResponseSchema] = Field(None, description="Provide ONLY if action is 'finish'")
    plan: List[str] = Field(default=[], description="Provide remaining steps ONLY if action is 'update_plan'")

# --- THE MANAGERS ---
async def planner_node(state: AgentState):
    logger.info("🧠 ACTIVATED: Planner Node -> Formulating execution plan...")
    
    # --- MONGODB LOGGING TRACE (Unchanged) ---
    total_messages = len(state.get("messages", []))
    logger.info(f"💾 DATABASE MEMORY: Loaded {total_messages} total message(s) from state.")
    
    if total_messages > 1:
        logger.info("🕒 HISTORY TRACE (Previous 3 messages):")
        previous_messages = state["messages"][-4:-1] 
        for msg in previous_messages:
            role = "USER" if msg.type == "user" else ("AI" if msg.type == "ai" else "TOOL")
            content = msg.content
            if len(content) > 75:
                content = content[:75] + "..."
            logger.info(f"   -> [{role}]: {content}")
    else:
        logger.info("🕒 HISTORY TRACE: 0 previous messages. This is a brand new conversation.")
    # ------------------------------------

    planner_agent = llm.with_structured_output(Plan)
    
    messages = [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=state["current_user_input"])]
    plan_output = await planner_agent.ainvoke(messages)
    
    logger.info(f"📋 PLAN CREATED: {plan_output.steps}")
    
    # THE FIX: Add "final_response": {} here to wipe out the previous turn's memory!
    return {
        "plan": plan_output.steps, 
        "past_steps": [], 
        "final_response": {} 
    }

async def replanner_node(state: AgentState):
    logger.info("🔄 ACTIVATED: Re-Planner Node -> Evaluating progress...")
    replanner_agent = llm.with_structured_output(Act)
    
    history_str = "\n".join(f"Task: {task}\nResult: {result}" for task, result in state.get("past_steps", []))
    prompt = f"{REPLANNER_PROMPT}\n\nCurrent Plan:\n{state.get('plan', [])}\n\nPast Steps:\n{history_str}"
    
    action = await replanner_agent.ainvoke([SystemMessage(content=prompt)])
    
    if action.action == "finish":
        logger.info(f"✅ PLAN COMPLETE: UI Action -> [{action.final_response.ui_action}] Target -> [{action.final_response.target_route}]")
        return {
            "final_response": action.final_response.model_dump(), 
            "messages": [AIMessage(content=action.final_response.reply)]
        }
    else:
        logger.info(f"⏳ PLAN UPDATED: Remaining steps -> {action.plan}")
        return {"plan": action.plan}

# --- THE WORKERS ---
async def rag_agent(state: AgentState):
    task = state["plan"][0] if state.get("plan") else "General portfolio search"
    logger.info(f"🟢 WORKER: RAG Agent executing task: '{task}'")
    
    messages = [SystemMessage(content=RAG_PROMPT), HumanMessage(content=task)]
    response = await rag_llm.ainvoke(messages)
    
    result_text = response.content
    
    # Manual tool execution to preserve detailed logging
    if hasattr(response, 'tool_calls') and len(response.tool_calls) > 0:
        tool_call = response.tool_calls[0]
        
        if tool_call["name"] == "retrieve_portfolio_info":
            optimized_query = tool_call["args"]["query"]
            logger.info(f"🔎 OPTIMIZED SEARCH: LLM translated input to: '{optimized_query}'")
            
            tool_output = retrieve_portfolio_info.invoke(optimized_query)
            
            tool_message = ToolMessage(
                content=str(tool_output), 
                tool_call_id=tool_call["id"], 
                name=tool_call["name"]
            )
            
            logger.info("✅ Pinecone search complete. Summarizing findings for the Re-Planner...")
            final_response = await rag_llm.ainvoke(messages + [response, tool_message])
            result_text = final_response.content

    return {"past_steps": [(task, result_text)]}

async def transactional_agent(state: AgentState):
    task = state["plan"][0] if state.get("plan") else "Run loan prediction"
    logger.info(f"🟢 WORKER: Transactional Agent executing task: '{task}'")
    
    agent = llm.bind_tools([execute_loan_prediction])
    response = await agent.ainvoke([SystemMessage(content=TRANSACTIONAL_PROMPT), HumanMessage(content=task)])
    
    return {"past_steps": [(task, response.content)]}

async def conversational_agent(state: AgentState):
    task = state["plan"][0] if state.get("plan") else "Respond to small talk"
    logger.info(f"🟢 WORKER: Conversational Agent executing task: '{task}'")
    
    response = await llm.ainvoke([SystemMessage(content=CONVERSATIONAL_PROMPT), HumanMessage(content=task)])
    return {"past_steps": [(task, response.content)]}