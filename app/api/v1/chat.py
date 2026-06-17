import logging
import traceback
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.services.agent.graph import graph
from app.db.database import get_client

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatRequest(BaseModel):
    message:str
    thread_id:str

def save_chat_to_db(thread_id: str, user_text: str, bot_text: str):
    """Saves the conversation to MongoDB in the background without making the user wait."""
    try:
        db_client = get_client()
        db = db_client["portfolio_chatbot_db"]
        collection = db["chat_history"]
        timestamp = datetime.utcnow()

        collection.insert_many([
            {
                "thread_id": thread_id,
                "sender": "user",
                "text": user_text,
                "timestamp": timestamp
            },
            {
                "thread_id": thread_id,
                "sender": "bot",
                "text": bot_text,
                "timestamp": timestamp
            }
        ])
        logger.info(f"✅ Background task: Chat saved for thread {thread_id}")
    except Exception as e:
        logger.error(f"❌ Failed to save history in background: {e}")
        traceback.print_exc()

@router.post("/message")
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        db_client = get_client()
        db = db_client["portfolio_chatbot_db"]
        collection =db["chat_history"]

        message_history = list(collection.find({"thread_id": request.thread_id}))
        
        formatted_history = []
        for doc in message_history:
            if doc.get("sender") == "user":
                formatted_history.append(HumanMessage(content=doc.get("text", "")))
            elif doc.get("sender") == "bot":
                formatted_history.append(AIMessage(content=doc.get("text", "")))
        
        # Append the new user message to the historical chain
        current_user_message = HumanMessage(content = request.message)
        full_message_history = formatted_history + [current_user_message]

        initial_state = {
            "messages": full_message_history,
            "current_user_input": request.message,
            "final_response":{}
        }

        config = {"configurable": {"thread_id": request.thread_id}}
        final_state = await graph.ainvoke(initial_state, config= config)

        final_data = final_state.get("final_response",{})

        if not final_data:
            final_message = final_state.get("messages",[])
            fallback_text = final_message[-1].content if final_message else "I encounterd an error."
            final_data = {
                "reply": fallback_text,
                "ui_action": "NONE",
                "target_route": ""
            }
        
        final_bot_response = final_data.get("reply", "I encountered an error.")
        action_type = final_data.get("ui_action", "NONE")
        target_route = final_data.get("target_route", "") or ""

        background_tasks.add_task(
            save_chat_to_db, 
            request.thread_id, 
            request.message, 
            final_bot_response
        )

        return {
            "reply": final_bot_response,
            "action": action_type,
            "target": target_route
        }
    
    except Exception as e:
        logger.error("🔥 EXCEPTION ENCOUNTERED IN CHAT LIFECYCLE 🔥")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )



