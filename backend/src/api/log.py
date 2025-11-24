# api/log.py

from fastapi import APIRouter
import logging
from pydantic import BaseModel

log_router = APIRouter()
logger = logging.getLogger(__name__)

class LogRequest(BaseModel):
    error: str
    timestamp: str
    userAgent: str
    url: str

@log_router.post("/log")
async def log_error(request: LogRequest):
    try:
        logger.error(f"Frontend Error - {request.timestamp}: {request.error}")
        logger.error(f"User Agent: {request.userAgent}")
        logger.error(f"URL: {request.url}")
        return {"success": True, "message": "Error logged successfully"}
    except Exception as e:
        logger.error(f"Failed to log error: {e}")
        return {"success": False, "message": "Failed to log error"}
