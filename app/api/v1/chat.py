from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent.graph import get_portfolio_graph
from langchain_core.messages import HumanMessage

router = APIRouter()

@router.post("/message", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        portfolio_graph = get_portfolio_graph()
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": request.thread_id}}
        
        result = await portfolio_graph.ainvoke(inputs, config=config)
        final_message = result["messages"][-1].content
        
        # Intercept the UI Navigation Command
        if "SYSTEM_COMMAND_NAVIGATE" in final_message:
            target = final_message.split(":")[1].strip()
            return ChatResponse(
                reply=f"Navigating to {target} now!",
                action="NAVIGATE",
                target=f"/{target}"
            )
            
        return ChatResponse(
            reply=final_message,
            action="NONE",
            target=""
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))