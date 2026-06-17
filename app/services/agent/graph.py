import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import supervisor_node, rag_agent, transactional_agent, conversational_agent

logger = logging.getLogger(__name__)

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add all nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("transactional_agent", transactional_agent)
    workflow.add_node("conversational_agent", conversational_agent)

    # 2. Start -> Supervisor (Every message goes to the LLM classifier first)
    workflow.add_edge(START, "supervisor")

    # 3. Supervisor -> Specific Agent based on LLM decision
    workflow.add_conditional_edges(
        "supervisor", 
        lambda state: state.get("target_agent", "conversational_agent"), 
        {
            "rag_agent": "rag_agent",
            "transactional_agent": "transactional_agent",
            "conversational_agent": "conversational_agent"
        }
    )

    # 4. Agents -> END (They format their own UI JSON, so they end the graph)
    workflow.add_edge("rag_agent", END)
    workflow.add_edge("transactional_agent", END)
    workflow.add_edge("conversational_agent", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

graph = build_graph()