from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_user_input: str
    final_response: dict
    activate_dialogue_target: Optional[str]