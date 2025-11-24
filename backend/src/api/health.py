# api/health.py

from fastapi import APIRouter

health_router = APIRouter()

@health_router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Vehicle Data Chatbot API is running"}
