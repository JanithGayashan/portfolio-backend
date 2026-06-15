from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import planner_node, replanner_node, rag_agent, transactional_agent, conversational_agent

def route_to_executor(state: AgentState):
    """
    Reads the first step of the plan and dynamically sends it to the right worker.
    """
    # Failsafe: If the planner returned an empty plan, end the graph safely.
    if not state.get("plan"):
        return END 
        
    next_step = state["plan"][0].lower()
    
    if "loan" in next_step:
        return "transactional_agent"
    elif any(keyword in next_step for keyword in ["portfolio", "experience", "search", "database", "education", "project", "janith"]):
        return "rag_agent"
    else:
        return "conversational_agent"

def route_after_replanner(state: AgentState):
    """
    Checks if the Re-Planner outputted a final response.
    If yes, stop. If no, dynamically route to the next task in the plan.
    """
    # 1. STRICT CHECK: Only end if the final_response dictionary actually exists AND has content
    if state.get("final_response") and state["final_response"].get("reply"):
        return END
    
    # 2. CONTINUATION: If not finished, use the router logic to find the NEXT worker
    return route_to_executor(state)

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("transactional_agent", transactional_agent)
    workflow.add_node("conversational_agent", conversational_agent)

    # We reuse this mapping dictionary for both conditional edges
    worker_mapping = {
        "rag_agent": "rag_agent",
        "transactional_agent": "transactional_agent",
        "conversational_agent": "conversational_agent",
        END: END
    }

    # 2. Wire the Plan and Execute Loop
    workflow.add_edge(START, "planner")
    
    # Route the FIRST task from the planner to a worker
    workflow.add_conditional_edges("planner", route_to_executor, worker_mapping)

    # 3. All Workers report back to the Re-Planner after finishing their task
    workflow.add_edge("rag_agent", "replanner")
    workflow.add_edge("transactional_agent", "replanner")
    workflow.add_edge("conversational_agent", "replanner")

    # 4. The Re-Planner decides if we are done, or loops to the NEXT worker
    workflow.add_conditional_edges("replanner", route_after_replanner, worker_mapping)

    # 5. Attach In-Memory Database
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

graph = build_graph()