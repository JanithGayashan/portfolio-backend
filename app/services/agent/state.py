import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_user_input: str
    
    # Plan and Execute Memory
    plan: list[str]
    past_steps: Annotated[list[tuple], operator.add] 
    
    # The Final Output holding both text and UI commands
    final_response: dict