from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="Unique ID for the user's browser session.")
    message: str = Field(..., description="The user's input message.")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="The AI's text response.")
    action: str = Field(..., description="Will be 'NAVIGATE' or 'NONE'.")
    target: str = Field(..., description="The frontend route to push to, if applicable.")