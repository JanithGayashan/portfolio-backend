from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import loan

# 1. Initialize the FastAPI application first!
app = FastAPI(title="Janith's Portfolio API")

# 2. Configure CORS so your Next.js frontend is allowed to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. A simple health check route
@app.get("/")
async def health_check():
    return {"status": "online", "message": "Backend is active. Terminal listening."}

# 4. NOW we can attach the Machine Learning router!
app.include_router(loan.router, prefix="/api/v1/loan", tags=["Machine Learning"])