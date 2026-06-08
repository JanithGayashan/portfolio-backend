from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.mongodb import MongoDBSaver

from app.db.database import get_client
from .state import AgentState
from .nodes import supervisor_node, conversational_agent, rag_agent, transactional_agent
from .tools.portfolio import navigate_website, retrieve_portfolio_info, execute_loan_prediction

def route_after_agent(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("conversational_agent", conversational_agent)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("transactional_agent", transactional_agent)
    
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
            "transactional_agent": "transactional_agent"
        }
    )

    # 3. Wire the Tool Callbacks
    workflow.add_conditional_edges("conversational_agent", route_after_agent, {"tools": "tools", END: END})
    workflow.add_conditional_edges("rag_agent", route_after_agent, {"tools": "tools", END: END})
    workflow.add_conditional_edges("transactional_agent", route_after_agent, {"tools": "tools", END: END})
    workflow.add_edge("tools", "supervisor")

    # 4. Attach Database Memory asynchronously
    memory = MongoDBSaver(client=get_client())
    return workflow.compile(checkpointer=memory)

# Lazy initialize portfolio graph to avoid requiring MongoDB at import time
_portfolio_graph = None

def get_portfolio_graph():
    global _portfolio_graph
    if _portfolio_graph is None:
        _portfolio_graph = build_graph()
    return _portfolio_graph