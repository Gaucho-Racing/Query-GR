"""Router for LLM query endpoint."""
import re
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.llm.agent import call_gemini
from src.tools.run_queries import process_query
from src.tools.utils import (
    extract_trip_id,
    is_vehicle_data_query,
    expand_trip_ids,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])

# Script generation cache (to reduce OpenRouter calls)
SCRIPT_CACHE_TTL_SECONDS = 3600  # 1 hour
_script_cache: Dict[str, Dict[str, Any]] = {}


def clear_script_cache():
    """Clear the script generation cache."""
    global _script_cache
    count = len(_script_cache)
    _script_cache.clear()
    return count


@router.post("/clear-cache", tags=["admin"])
@router.get("/clear-cache", tags=["admin"])
async def clear_cache():
    """Clear all caches (script cache and signals cache).
    
    Can be called via:
    - GET: curl http://localhost:8000/llm/clear-cache
    - POST: curl -X POST http://localhost:8000/llm/clear-cache
    """
    from src.tools.utils import DB_SIGNALS_CACHE
    
    script_cache_count = clear_script_cache()
    signals_cache_count = len(DB_SIGNALS_CACHE)
    DB_SIGNALS_CACHE.clear()
    
    logger.info(f"[CACHE CLEAR] Cleared {script_cache_count} script cache entries and {signals_cache_count} signals cache entries")
    
    return {
        "success": True,
        "message": "Caches cleared successfully",
        "cleared": {
            "script_cache_entries": script_cache_count,
            "signals_cache_entries": signals_cache_count
        }
    }


class QueryRequest(BaseModel):
    message: str


class QueryResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any] = None
    error: str = None


@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Handle user queries and process vehicle data"""
    try:
        message = request.message.strip()
        
        if not message:
            return QueryResponse(
                success=False,
                message="Please provide a valid query.",
                error="Empty message"
            )
        
        # Extract trip_id early to use in cache lookups
        chosen_trip = extract_trip_id(message)
        
        # Check if it's a vehicle data query (only use cache if trip_id is available)
        # If trip_id is None, we'll ask for clarification after this check
        if not is_vehicle_data_query(message, trip_id=chosen_trip, default_trip_id="4"):
            return QueryResponse(
                success=True,
                message="Sorry, I can't help you with that. I can only assist with vehicle data queries."
            )

        # Clarify ambiguous cell queries without a cell number (e.g., "give me cell temp")
        ml = message.lower()
        if re.search(r"\bcell\b", ml) and (
            re.search(r"\btemp(?:erature)?\b", ml) or re.search(r"\bvolt(?:age)?\b", ml)
        ) and not re.search(r"\b\d{1,3}\b", ml):
            metric = "voltage" if re.search(r"\bvolt", ml) else "temperature"
            return QueryResponse(
                success=True,
                message=f"Which cell number for {metric}? e.g., 16 or 110",
                data={"intent": "clarify_cell_metric", "metric": metric}
            )

        # Clarify trip/run if user did not specify any trip id (do this after vehicle query check)
        # Don't ask if "all" or range is specified
        if chosen_trip is None:
            return QueryResponse(
                success=True,
                message="Which trip (run) number? e.g., 3, or 'all' for all trips, or 'trips 3-5' for a range",
                data={"intent": "clarify_trip"}
            )
        
        # Process query using the tools
        try:
            result, debug_info, image_b64, table_data, suggested_pairs, top_explained = await process_query(
                message,
                call_gemini,
                _script_cache,
                SCRIPT_CACHE_TTL_SECONDS,
            )
            
            # Log if result indicates no data
            result_lower = result.lower() if result else ""
            if any(phrase in result_lower for phrase in ["no data found", "no instances found", "no matching", "could not find", "no data available"]):
                logger.warning(
                    f"[API RESPONSE] Query returned no data: '{message[:100]}...' "
                    f"Result message: {result[:200]}, "
                    f"Trip IDs: {expand_trip_ids(chosen_trip) if chosen_trip else ['4']}, "
                    f"Selected signals: {[p[0] for p in suggested_pairs] if suggested_pairs else []}"
                )
            
            # Log if table_data is empty
            if table_data and isinstance(table_data, dict):
                rows = table_data.get("rows", [])
                if not rows or len(rows) == 0:
                    logger.warning(
                        f"[API RESPONSE] Empty table_data returned for query: '{message[:100]}...' "
                        f"Columns: {table_data.get('columns', [])}, "
                        f"Trip IDs: {expand_trip_ids(chosen_trip) if chosen_trip else ['4']}"
                    )
            
            # Build response payload
            signal_scoring_payload = [
                {"signal": sig, **details} for sig, score, details in (top_explained if top_explained else [])
            ]
            data_payload: Dict[str, Any] = {
                "debug": debug_info,
                "signal_scoring": {
                    "selected": suggested_pairs if suggested_pairs else [],
                    "top": signal_scoring_payload,
                },
                "trip_id_used": chosen_trip or "4",
                "trip_ids_processed": expand_trip_ids(chosen_trip) if chosen_trip else ["4"],
            }
            if image_b64:
                data_payload["image_base64"] = image_b64
            if table_data:
                data_payload["table_data"] = table_data
            
            response = QueryResponse(
                success=True,
                message=result,
                data=data_payload
            )
            logger.info(f"[API RESPONSE] Returning success response. Message length: {len(result)}, Has image: {image_b64 is not None}, Has table_data: {table_data is not None}")
            if table_data and isinstance(table_data, dict):
                rows_count = len(table_data.get("rows", []))
                cols_count = len(table_data.get("columns", []))
                logger.info(f"[API RESPONSE] Table data: {rows_count} rows, {cols_count} columns")
            return response
        except ValueError as e:
            # No matching signals found
            return QueryResponse(
                success=True,
                message="Sorry, I can't help you with that. I can only assist with vehicle data queries.",
            )
        except HTTPException as e:
            # Propagate HTTP exceptions (e.g., rate limits)
            raise e
        except Exception as e:
            logger.error(f"[QUERY ERROR] Failed to process query: {e}", exc_info=True)
            logger.error(f"[QUERY ERROR] Query was: '{message[:200]}'")
            logger.error(f"[QUERY ERROR] Exception type: {type(e).__name__}")
            return QueryResponse(
                success=False,
                message="Sorry, I couldn't process your query.",
                error=str(e)
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions (e.g., rate limits)
        raise
    except Exception as e:
        logger.error(f"[ROUTER ERROR] Unexpected error in query handler: {e}", exc_info=True)
        logger.error(f"[ROUTER ERROR] Query was: '{request.message[:200] if hasattr(request, 'message') else 'N/A'}'")
        logger.error(f"[ROUTER ERROR] Exception type: {type(e).__name__}")
        return QueryResponse(
            success=False,
            message="Sorry, an unexpected error occurred.",
            error=str(e)
        )

