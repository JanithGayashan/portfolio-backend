from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import supervisor_node, conversational_agent, rag_agent, transactional_agent, out_of_bounds_node
from .tools.portfolio import navigate_website, retrieve_portfolio_info, execute_loan_prediction

def route_after_agent(state: AgentState):
    last_message = state["messages"][-1]
    
    # 1. Safely check if the LLM natively decided to call a tool
    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        return "tools"
        
    # 2. THE GUARDRAIL LOOP: If the RAG Agent manually intercepted a failure,
    # route it back to the supervisor so it can hand off to OUT_OF_BOUNDS!
    content = getattr(last_message, "content", "")
    if "GUARDRAIL_TRIGGERED" in content:
        return "supervisor"
        
    # 3. Otherwise, the agent successfully answered the user. Stop the graph.
    return END

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("conversational_agent", conversational_agent)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("transactional_agent", transactional_agent)
    workflow.add_node("out_of_bounds", out_of_bounds_node)
    
    tools = [navigate_website, retrieve_portfolio_info, execute_loan_prediction]
    workflow.add_node("tools", ToolNode(tools))

    # 2. Wire the Router
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        {
            "conversational_agent": "conversational_agent",
            "rag_agent": "rag_agent",
            "transactional_agent": "transactional_agent",
            "OUT_OF_BOUNDS": "out_of_bounds"
        }
    )

    # 3. Wire the Tool Callbacks
    # NOTE: We MUST include "supervisor" in this map so the guardrail handoff works!
    edge_map = {
        "tools": "tools", 
        "supervisor": "supervisor", 
        END: END
    }
    
    workflow.add_conditional_edges("conversational_agent", route_after_agent, edge_map)
    workflow.add_conditional_edges("rag_agent", route_after_agent, edge_map)
    workflow.add_conditional_edges("transactional_agent", route_after_agent, edge_map)
    
    workflow.add_edge("tools", "supervisor")

    # Route the guardrail directly to END so it immediately stops without tools
    workflow.add_edge("out_of_bounds", END)

    # 4. Attach In-Memory Database
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

graph = build_graph()