from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import pandas as pd
import json
import logging
from typing import Dict, Any
from time import time, sleep
import signal
import os
from dotenv import load_dotenv
import re

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

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any] = None
    error: str = None

class LogRequest(BaseModel):
    error: str
    timestamp: str
    userAgent: str
    url: str

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
GEMINI_RETRY_BACKOFF = float(os.getenv("GEMINI_RETRY_BACKOFF", "1.0"))
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "300"))

# Debug diagnostics toggle
DEBUG_ANALYSIS = os.getenv("DEBUG_ANALYSIS", "false").lower() == "true"

# Vehicle data API configuration
VEHICLE_API_URL = "https://mapache.gauchoracing.com/api/query/signals"
VEHICLE_API_TOKEN = "01b3939d-678f-44ac-93ff-0d54e09ba3d6"
VEHICLE_ID = "gr24-main"
TRIP_ID = "4"
VEHICLE_DATA_TIMEOUT = float(os.getenv("VEHICLE_DATA_TIMEOUT", "30"))
SCRIPT_TIMEOUT = int(os.getenv("SCRIPT_TIMEOUT", "20"))

# Script generation cache (to reduce OpenRouter calls)
SCRIPT_CACHE_TTL_SECONDS = 3600  # 1 hour
_script_cache: Dict[str, Dict[str, Any]] = {}

SIGNALS = ""
df = pd.read_csv("signals.csv")
for index, row in df.iterrows():
    SIGNALS += row["name"] + ", "
logger.info("-- -- -- -- json dumps: " + json.dumps(SIGNALS))

async def call_gemini(prompt: str) -> str:
    """Call Gemini API to generate content"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
    params = {"key": GEMINI_API_KEY}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        attempt = 0
        while True:
            try:
                logger.info(f"Gemini request model={GEMINI_MODEL} prompt_len={len(prompt)} attempt={attempt}")
                response = await client.post(url, params=params, json=payload, timeout=GEMINI_TIMEOUT)
                logger.info(response)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "60")
                    logger.warning(f"Gemini rate limited. Retry-After: {retry_after}")
                    raise HTTPException(status_code=429, detail="Rate limited by AI service", headers={"Retry-After": retry_after})

                if response.status_code == 404:
                    # Likely invalid model or wrong base URL version
                    raise HTTPException(status_code=404, detail=f"Gemini model not found or unavailable: {GEMINI_MODEL}")

                # Retry on transient 5xx
                if 500 <= response.status_code < 600 and attempt < GEMINI_MAX_RETRIES:
                    backoff = GEMINI_RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"Gemini 5xx {response.status_code}, retrying in {backoff:.1f}s")
                    sleep(backoff)
                    attempt += 1
                    continue

                response.raise_for_status()
                data = response.json()
                # Extract text from candidates
                candidates = data.get("candidates", [])
                if not candidates:
                    raise HTTPException(status_code=500, detail="No candidates returned by Gemini")
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts)
                logger.info(f"Gemini response chars={len(text)}")
                return text
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response else 500
                if status == 429:
                    retry_after = e.response.headers.get("Retry-After", "60") if e.response else "60"
                    raise HTTPException(status_code=429, detail="Rate limited by AI service", headers={"Retry-After": retry_after})
                if status == 404:
                    raise HTTPException(status_code=404, detail=f"Gemini model not found or unavailable: {GEMINI_MODEL}")
                if 500 <= status < 600 and attempt < GEMINI_MAX_RETRIES:
                    backoff = GEMINI_RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"Gemini HTTP error {status}, retrying in {backoff:.1f}s")
                    sleep(backoff)
                    attempt += 1
                    continue
                body_preview = ""
                try:
                    body_preview = (e.response.text or "")[:500]
                except Exception:
                    body_preview = ""
                logger.error(f"Gemini HTTP error: {e} body={body_preview}")
                detail = f"AI service error: {str(e)}"
                if body_preview:
                    detail += f" | body: {body_preview}"
                raise HTTPException(status_code=status, detail=detail)
            except httpx.RequestError as e:
                if attempt < GEMINI_MAX_RETRIES:
                    backoff = GEMINI_RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"Gemini request error {e.__class__.__name__}, retrying in {backoff:.1f}s")
                    sleep(backoff)
                    attempt += 1
                    continue
                logger.error(f"Gemini request error: {repr(e)}")
                raise HTTPException(status_code=503, detail=f"AI service network error: {e.__class__.__name__}: {str(e)}")
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

async def fetch_vehicle_data(signals: str = "mobile_speed") -> Dict[str, Any]:
    """Deprecated direct fetch; retained for reference. The analysis script now fetches the data itself."""
    params = {
        "vehicle_id": VEHICLE_ID,
        "trip_id": TRIP_ID,
        "signals": signals,
        "token": VEHICLE_API_TOKEN
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(VEHICLE_API_URL, params=params, timeout=30.0)
        response.raise_for_status()
        return response.json()

def is_vehicle_data_query(message: str) -> bool:
    """Simple keyword-based detection for vehicle data queries"""
    vehicle_keywords = [
        "speed", "mobile_speed", "average", "max", "min", "mean", "median",
        "vehicle", "data", "telemetry", "trip", "signal", "sensor",
        "acceleration", "brake", "throttle", "rpm", "fuel", "temperature"
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in vehicle_keywords)

async def generate_pandas_script(query: str) -> str:
    """Generate a Pandas script to process vehicle data.
    The generated script MUST fetch the JSON from the vehicle API itself using httpx (or requests),
    parse into pandas, compute the requested metric, and print a concise result.
    """
    prompt = f"""
You are a data analysis expert. Generate a Python script that:

1. Infers the relevant signal names from the user request (e.g., mobile_speed, acu_cellX_temp, acu_cellY_voltage, etc.).
   Build the signals parameter as a string list.
   These are the only allowed signals (separated by ", "): {SIGNALS}
   - Use the user query to find the most relevant signals to query. ONLY use these signals when fetching the API
   Use the provided helper `build_url(signals: list[str])` to construct the URL. Do not use data from any other API. Do not create mock data. Do not make up data without fetching the API. If you do not use the build_url and http_get helpers then do return any data.
   Then use `http_get(url)` to fetch the JSON.
2. Computes the metric requested by the user query below
3. Prints a clear, single-line result string for the user

User Query: "{json.dumps(query)}" (This variable is not defined, you have to define it yourself with this string value)

Support advanced queries like:
- "top 10 mobile_speed values"
- "compare average mobile_speed between trip 4 and trip 5"
- "min/median/max of mobile_speed"
 - "correlation between temperature and voltage signals"

Make your script resilient and efficient.

Requirements:
- Use: import pandas as pd
- Use the provided helper: http_get(url) instead of calling httpx directly (this logs the URL and returns a Response for .json()).
- Make the HTTP GET within the script to fetch the JSON data.
- Do not use any external variables; construct the URL with params inside the script.
- The only defined global variables/functions are: pd json httpx print set_result http_get build_url, do not call any other undefined variables/functions
- Parse the JSON robustly. Possible shapes include:
  * {{"signals": {{"mobile_speed": [...]}}}}
  * {{"data": {{"mobile_speed": [...]}}}}
  * {{"data": {{"data": [{{...}}] }}}}
  * {{"mobile_speed": [...]}}
  * A list of dicts containing fields like "mobile_speed", "value" or similar.
  Try these paths in order and handle missing keys gracefully. Flatten into a numeric series.
- Clean the series: coerce to numeric, drop nulls, then compute the metric requested by the user query (e.g., average/mean, min, max, median).
 - For correlation tasks, align signals on produced_at timestamps when applicable; compute Pearson correlation.
- At the end, set a variable named `result` to the final one-line string AND call both `print(result)` and `set_result(result)` (available at runtime).
- Do not include explanations or markdown. Output ONLY executable Python code.
"""

    return prompt

    # script = await call_gemini(prompt)
    # if DEBUG_ANALYSIS:
    #     logger.info(f"Generated script length={len(script)} for query='{query[:60]}'...")
    # return script

async def execute_pandas_script(script: str) -> tuple[str, Dict[str, Any]]:
    """Execute the generated Pandas script safely"""
    def sanitize_generated_code(raw: str) -> str:
        # Extract code from triple backtick fences if present, drop language tag
        matches = re.findall(r"```(?:[\w+-]*)?\n([\s\S]*?)```", raw)
        if matches:
            code = "\n\n".join(matches)
        else:
            code = raw
        # Remove any stray triple backticks and leading 'python' markers
        code = code.replace("```", "")
        code = re.sub(r"^\s*python\n", "", code, flags=re.IGNORECASE)
        return code.strip()

    try:
        sanitized = sanitize_generated_code(script)
        # Create a safe execution environment
        captured: Dict[str, Any] = {"__captured_result": None}
        def set_result(value: Any) -> None:
            try:
                captured["__captured_result"] = value
            except Exception:
                captured["__captured_result"] = None

        def http_get(url: str):
            logger.info(f"AI script HTTP GET: {url}")
            resp = httpx.get(url, timeout=VEHICLE_DATA_TIMEOUT)
            return resp

        def build_url(signals: list[str]):
            # Construct signals query (comma-separated, no spaces)
            sig_param = ",".join(s.strip() for s in signals if s and isinstance(s, str))
            base = (
                f"{VEHICLE_API_URL}?vehicle_id={VEHICLE_ID}&trip_id={TRIP_ID}"
                f"&signals={sig_param}&token={VEHICLE_API_TOKEN}"
            )
            logger.info(f"AI script build_url: {base}")
            return base
        
        # def get_possible_signals():
        #     df = pd.read_csv("signals.csv")
        #     return df

        safe_globals = {
            'pd': pd,
            'json': json,
            'httpx': httpx,
            'print': print,
            'set_result': set_result,
            'http_get': http_get,
            'build_url': build_url,
            # 'get_possible_signals': get_possible_signals
        }
        
        # Capture the output
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        output = io.StringIO()
        err_output = io.StringIO()
        # Compile first to surface SyntaxError clearly
        code_object = compile(sanitized, '<generated>', 'exec')
        start_ts = time()
        # Temporarily disable built-in exit/quit to prevent SystemExit from scripts
        import builtins as _bi
        def _no_op(*args, **kwargs):
            return None
        old_exit = getattr(_bi, 'exit', None)
        old_quit = getattr(_bi, 'quit', None)
        _bi.exit = _no_op  # type: ignore
        _bi.quit = _no_op  # type: ignore
        # Setup a hard timeout using SIGALRM (Unix-only)
        def _on_timeout(signum, frame):  # type: ignore[no-redef]
            raise TimeoutError("Script execution timed out")
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.alarm(SCRIPT_TIMEOUT)
        try:
            with redirect_stdout(output), redirect_stderr(err_output):
                exec(code_object, safe_globals)
        finally:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
            if old_exit is not None:
                _bi.exit = old_exit  # type: ignore
            if old_quit is not None:
                _bi.quit = old_quit  # type: ignore
        duration_ms = int((time() - start_ts) * 1000)
        result = output.getvalue().strip()
        stderr = err_output.getvalue().strip()
        debug: Dict[str, Any] = {
            "sanitized_len": len(sanitized),
            "stdout_len": len(result),
            "stderr_len": len(stderr),
            "duration_ms": duration_ms,
        }
        if result:
            return result, debug

        # Fallback: try to read common result variables if nothing was printed
        for var_name in ("result", "answer", "output"):
            if var_name in safe_globals and safe_globals[var_name] is not None:
                try:
                    debug["fallback_var"] = var_name
                    return str(safe_globals[var_name]), debug
                except Exception:
                    continue
        # Fallback 2: captured result via set_result
        if captured.get("__captured_result") is not None:
            debug["fallback_var"] = "__captured_result"
            return str(captured["__captured_result"]), debug

        debug["reason"] = "no stdout and no fallback variable"
        return "No output generated", debug
        
    except SyntaxError as e:
        logger.error(f"Pandas script syntax error: {e}\nOriginal script:\n{script}")
        raise HTTPException(status_code=500, detail=f"Script execution failed: {e}")
    except TimeoutError as e:
        logger.error(f"Pandas script execution timeout: {e}")
        raise HTTPException(status_code=504, detail="Script execution timed out")
    except Exception as e:
        logger.error(f"Pandas script execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Script execution failed: {str(e)}")

@app.post("/query", response_model=ChatResponse)
async def handle_query(request: ChatRequest):
    """Handle user queries and process vehicle data"""
    try:
        message = request.message.strip()
        
        if not message:
            return ChatResponse(
                success=False,
                message="Please provide a valid query.",
                error="Empty message"
            )
        
        # Check if it's a vehicle data query
        if not is_vehicle_data_query(message):
            return ChatResponse(
                success=True,
                message="Sorry, I can't help you with that. I can only assist with vehicle data queries."
            )
        
        # Fetch vehicle data
        try:
            # Do not fetch data here to avoid extra API calls; the generated script will fetch directly
            vehicle_data = None
        except HTTPException as e:
            # Propagate HTTPExceptions (e.g., 429 with Retry-After)
            raise e
        except Exception as e:
            logger.error(f"Failed to fetch vehicle data: {e}")
            return ChatResponse(
                success=False,
                message="Sorry, I couldn't fetch the data.",
                error=str(e)
            )
        
        # Generate Pandas script
        try:
            # Cache by normalized query to reduce model usage
            normalized = message.strip().lower()
            cached = _script_cache.get(normalized)
            now = time()
            if cached and (now - cached["ts"]) < SCRIPT_CACHE_TTL_SECONDS:
                pandas_script = cached["script"]
            else:
                prompt = await generate_pandas_script(message)
                pandas_script = await call_gemini(prompt)
                if DEBUG_ANALYSIS:
                    logger.info(f"Generated script length={len(pandas_script)} for query='{message[:60]}'...")
                _script_cache[normalized] = {"script": pandas_script, "ts": now}
        except HTTPException as e:
            # Propagate rate limit or other HTTP errors with headers
            raise e
        except Exception as e:
            logger.error(f"Failed to generate Pandas script: {e}")
            return ChatResponse(
                success=False,
                message="Sorry, I couldn't process your query.",
                error=str(e)
            )
        
        # Execute the script
        try:
            result, debug_info = await execute_pandas_script(pandas_script)
            data_payload: Dict[str, Any] = {"script": pandas_script, "debug": debug_info}
            return ChatResponse(
                success=True,
                message=result,
                data=data_payload
            )
        except Exception as e:
            logger.error(f"Failed to execute Pandas script: {e}")
            return ChatResponse(
                success=False,
                message="Sorry, I couldn't process your query.",
                error=str(e)
            )
            
    except Exception as e:
        logger.error(f"Unexpected error in query handler: {e}")
        return ChatResponse(
            success=False,
            message="Sorry, an unexpected error occurred.",
            error=str(e)
        )

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
