"""Main FastAPI application entry point."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv

from src.llm.router import router as llm_router, handle_query, QueryRequest

# Load environment variables
load_dotenv()

app = FastAPI(title="Vehicle Data Chatbot API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount routers
app.include_router(llm_router)

# Backward compatibility: redirect /query to /llm/query
@app.post("/query")
async def query_compat(request: Request):
    """Backward compatibility endpoint that forwards to /llm/query"""
    body = await request.json()
    return await handle_query(QueryRequest(**body))

# Request/Response models
class LogRequest(BaseModel):
    error: str
    timestamp: str
    userAgent: str
    url: str


@app.post("/log")
async def log_error(request: LogRequest):
    """Log frontend errors"""
    try:
        logger.error(f"Frontend Error - {request.timestamp}: {request.error}")
        logger.error(f"User Agent: {request.userAgent}")
        logger.error(f"URL: {request.url}")
        return {"success": True, "message": "Error logged successfully"}
    except Exception as e:
        logger.error(f"Failed to log error: {e}")
        return {"success": False, "message": "Failed to log error"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Vehicle Data Chatbot API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

