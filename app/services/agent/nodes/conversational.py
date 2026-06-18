import logging
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..state import AgentState
from ..prompts import CONVERSATIONAL_PROMPT

logger = logging.getLogger(__name__)

# 1. THE STRUCTURE GUARDRAIL
# This forces the LLM to output the exact JSON your Next.js UI expects, 
# preventing broken responses.
class UIResponse(BaseModel):
    reply: str = Field(description="The conversational response to the user.")
    ui_action: str = Field(description="UI action: 'NONE', 'AUTO_NAVIGATE', or 'PROMPT_USER'")
    target_route: str = Field(description="The route to navigate to, e.g., '/projects', or ''")

# We attach the strict structure directly to the model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
structured_llm = llm.with_structured_output(UIResponse)

async def conversational_agent(state: AgentState):
    logger.info("🟢 WORKER: Conversational Agent executing...")
    
    # 2. Grab the full conversation history from the state
    messages = state.get("messages", [])
    
    # 3. THE BEHAVIOR GUARDRAIL
    # We explicitly define the boundaries of the portfolio chatbot.
    # The LLM will combine this with your existing CONVERSATIONAL_PROMPT.
    GUARDRAIL_INSTRUCTION = """
    CRITICAL GUARDRAIL: You are an AI assistant representing a professional engineering portfolio.
    You MUST explicitly refuse to answer any questions that fall outside these domains:
    - The professional experience, education, or skills of the portfolio owner (AI, Web Development, etc.).
    - The technical architecture of this chatbot or portfolio website.
    - Professional networking or hiring inquiries.
    
    If the user asks an off-topic question (e.g., cooking recipes, general trivia, writing essays, coding tasks unrelated to this portfolio), 
    politely decline. Briefly explain that you are a portfolio assistant and pivot the conversation back to technical skills or projects.
    """
    
    # Combine your base persona with the strict guardrails
    system_message = SystemMessage(content=f"{CONVERSATIONAL_PROMPT}\n\n{GUARDRAIL_INSTRUCTION}")
    
    # We pass the system prompt followed by the exact history of the conversation
    invocation_messages = [system_message] + messages
    
    try:
        # 4. Invoke the LLM
        # Because we use `structured_llm`, response_data will be a perfect UIResponse object
        response_data = await structured_llm.ainvoke(invocation_messages)
        
        # 5. Return the data to state in the exact format your chat.py endpoint needs
        return {
            "final_response": {
                "reply": response_data.reply,
                "ui_action": response_data.ui_action,
                "target_route": response_data.target_route
            }
        }
        
    except Exception as e:
        # 6. THE FALLBACK GUARDRAIL
        # If the LLM times out or the OpenAI API crashes, we fail gracefully
        logger.error(f"❌ Conversational Agent Failed: {e}")
        return {
            "final_response": {
                "reply": "I apologize, but I am having trouble connecting to my linguistic core right now. Can we try that again?",
                "ui_action": "NONE",
                "target_route": ""
            }
        }