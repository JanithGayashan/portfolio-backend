import logging
import traceback
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

# 1. FIXED IMPORTS: Using your exact graph name and database client!
from app.services.agent.graph import graph
from app.db.database import get_client  

logger = logging.getLogger(__name__)
router = APIRouter()

# 2. FIXED SCHEMA: Using 'message' to perfectly match your Next.js frontend
class ChatRequest(BaseModel):
    message: str
    thread_id: str

@router.post("/message")
async def chat_endpoint(request: ChatRequest):
    try:
        # 3. Setup DB Connection
        # Using the exact same get_client() and db_name from your graph.py!
        db_client = get_client()
        db = db_client["portfolio_chatbot_db"]
        collection = db["chat_history"]

        # 4. Fetch historical messages from MongoDB
        history_docs = list(collection.find({"thread_id": request.thread_id}))
        
        formatted_history = []
        for doc in history_docs:
            if doc.get("sender") == "user":
                formatted_history.append(HumanMessage(content=doc.get("text", "")))
            elif doc.get("sender") == "bot":
                formatted_history.append(AIMessage(content=doc.get("text", "")))

        # 5. Append the brand new user message to the historical chain
        current_user_message = HumanMessage(content=request.message)
        full_message_history = formatted_history + [current_user_message]

        # 6. Construct the initial state for the graph execution
        initial_state = {
            "messages": full_message_history,
            "current_user_input": request.message
        }

        # 7. Invoke the graph (Runs completely in-memory via MemorySaver)
        config = {"configurable": {"thread_id": request.thread_id}}
        final_state = await graph.ainvoke(initial_state, config=config)

        # 8. Extract the final response message
        final_messages = final_state.get("messages", [])
        if not final_messages:
            raise HTTPException(status_code=500, detail="Graph failed to produce messages.")
        
        final_bot_response = final_messages[-1].content

        # 9. INTERCEPT UI NAVIGATION COMMANDS
        action_type = "NONE"
        target_route = ""
        
        if "SYSTEM_COMMAND_NAVIGATE:" in final_bot_response:
            action_type = "NAVIGATE"
            target_route = "/" + final_bot_response.split("SYSTEM_COMMAND_NAVIGATE:")[1].strip()
            final_bot_response = f"Navigating to the {target_route.replace('/', '')} section now!"

        # 10. POST-RESPONSE PERSISTENCE: Save both items safely to MongoDB
        timestamp = datetime.utcnow()
        collection.insert_many([
            {
                "thread_id": request.thread_id,
                "sender": "user",
                "text": request.message,
                "timestamp": timestamp
            },
            {
                "thread_id": request.thread_id,
                "sender": "bot",
                "text": final_bot_response,
                "timestamp": timestamp
            }
        ])

        # 11. Return perfectly formatted data directly to your ChatWidget.tsx
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