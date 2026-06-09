from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.api.v1 import loan
from app.api.v1 import chat

app = FastAPI(title="Janith's Portfolio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "online", "message": "Backend is active. Terminal listening."}

# Mount routers
app.include_router(loan.router, prefix="/api/v1/loan", tags=["Machine Learning"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["AI Chatbot"])