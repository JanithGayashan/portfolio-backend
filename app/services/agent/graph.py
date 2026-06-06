from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from app.db.database import get_client
from .state import AgentState
from .nodes import supervisor_node, conversational_agent, rag_agent, transactional_agent
# Notice it says .tools.portfolio instead of .tools.portfolio_tools
from .tools.portfolio import navigate_website, retrieve_portfolio_info, execute_loan_prediction

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("conversational_agent", conversational_agent)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("transactional_agent", transactional_agent)
    
    tools = [navigate_website, retrieve_portfolio_info, execute_loan_prediction]
    workflow.add_node("tools", ToolNode(tools))

    # 2. Add Routing Edges (Exactly as mapped out previously)
    workflow.add_edge(START, "supervisor")
    # ... (Include the conditional edges here) ...
    workflow.add_edge("tools", "supervisor")

    # 3. Compile with Database Memory
    memory = AsyncMongoDBSaver(get_client())
    return workflow.compile(checkpointer=memory)

# Export the compiled graph to be used in your API router
portfolio_graph = build_graph()