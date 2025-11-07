from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import pandas as pd
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from time import time, sleep
import signal
import os
from dotenv import load_dotenv
import re
import csv

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
VEHICLE_DATA_TIMEOUT = float(os.getenv("VEHICLE_DATA_TIMEOUT", "60"))
SCRIPT_TIMEOUT = int(os.getenv("SCRIPT_TIMEOUT", "60"))

# Script generation cache (to reduce OpenRouter calls)
SCRIPT_CACHE_TTL_SECONDS = 3600  # 1 hour
_script_cache: Dict[str, Dict[str, Any]] = {}

# Load known signal names from CSV for fuzzy matching
SIGNALS_CSV_PATH = os.getenv("SIGNALS_CSV_PATH", os.path.join(os.path.dirname(__file__), "signals.csv"))
KNOWN_SIGNALS: List[str] = []
try:
    with open(SIGNALS_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                cell = cell.strip()
                if cell:
                    KNOWN_SIGNALS.append(cell)
    KNOWN_SIGNALS = sorted(set(KNOWN_SIGNALS))
    logger.info(f"Loaded {len(KNOWN_SIGNALS)} signals from {SIGNALS_CSV_PATH}")
except Exception as e:
    logger.warning(f"Failed to load signals from {SIGNALS_CSV_PATH}: {e}")


def score_signal(query: str, signal_name: str) -> float:
    """Return a float score in [0,200]: 0 best match, 200 worst.
    Uses substring cues (keywords, numbers) and difflib ratio.
    """
    q = query.lower()
    s = signal_name.lower()

    # Base: substring similarity
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, q, s).ratio()  # 0..1
    score = (1.0 - ratio) * 150.0  # 0..150

    # Keyword presence bonuses (lower score)
    keywords = [
        ("voltage", 40.0), ("volt", 35.0), ("temp", 35.0), ("temperature", 40.0),
        ("speed", 30.0), ("current", 30.0), ("cell", 20.0), ("acu", 10.0),
        ("inverter", 10.0), ("magnetometer", 10.0)
    ]
    for kw, bonus in keywords:
        if kw in q and kw in s:
            score -= bonus

    # Numeric cues (e.g., cell index like 16)
    nums = re.findall(r"\d+", q)
    for n in nums:
        if n in s:
            score -= 30.0

    # Penalize if signal_name shares no tokens with query at all
    qtokens = set(re.findall(r"[a-zA-Z0-9_]+", q))
    stokens = set(re.findall(r"[a-zA-Z0-9_]+", s))
    if qtokens and stokens and qtokens.isdisjoint(stokens):
        score += 20.0

    # Clamp to 0..200
    if score < 0.0:
        score = 0.0
    if score > 200.0:
        score = 200.0
    return float(score)


def best_signals_for_query(query: str, max_signals: int = 6, threshold: float = 100.0) -> List[Tuple[str, float]]:
    if not KNOWN_SIGNALS:
        return []
    scored = [(sig, score_signal(query, sig)) for sig in KNOWN_SIGNALS]
    scored.sort(key=lambda x: x[1])
    return [(sig, sc) for sig, sc in scored[:max_signals] if sc <= threshold]


def score_signal_with_details(query: str, signal_name: str) -> Tuple[float, Dict[str, Any]]:
    """Detailed scoring breakdown for logging and debugging."""
    q = query.lower()
    s = signal_name.lower()

    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, q, s).ratio()  # 0..1
    base = (1.0 - ratio) * 150.0
    score = base

    bonuses_hit: List[str] = []
    keywords = [
        ("voltage", 40.0), ("volt", 35.0), ("temp", 35.0), ("temperature", 40.0),
        ("speed", 30.0), ("current", 30.0), ("cell", 20.0), ("acu", 10.0),
        ("inverter", 10.0), ("magnetometer", 10.0)
    ]
    for kw, bonus in keywords:
        if kw in q and kw in s:
            score -= bonus
            bonuses_hit.append(kw)

    numeric_hits: List[str] = []
    nums = re.findall(r"\d+", q)
    for n in nums:
        if n in s:
            score -= 30.0
            numeric_hits.append(n)

    qtokens = set(re.findall(r"[a-zA-Z0-9_]+", q))
    stokens = set(re.findall(r"[a-zA-Z0-9_]+", s))
    disjoint = bool(qtokens and stokens and qtokens.isdisjoint(stokens))
    if disjoint:
        score += 20.0

    # Clamp
    score = 0.0 if score < 0.0 else (200.0 if score > 200.0 else score)

    details = {
        "signal": signal_name,
        "ratio": round(ratio, 4),
        "base": round(base, 2),
        "bonuses_hit": bonuses_hit,
        "numeric_hits": numeric_hits,
        "disjoint": disjoint,
        "final": round(float(score), 2),
    }
    return float(score), details

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
        response = await client.get(VEHICLE_API_URL, params=params, timeout=60.0)
        response.raise_for_status()
        return response.json()

def is_vehicle_data_query(message: str) -> bool:
    """Determine via fuzzy matching against known signals and query intent terms."""
    # Quick intent words
    intents = ["average", "mean", "median", "min", "max", "top", "bottom", "compare", "correlation", "corr", "speed", "temperature", "voltage", "current"]
    ml = message.lower()
    intent_hit = any(w in ml for w in intents)
    # Fuzzy signal matches
    matches = best_signals_for_query(message, max_signals=3, threshold=100.0)
    return intent_hit or len(matches) > 0

async def generate_pandas_script(query: str, suggested_signals: List[str]) -> str:
    """Generate a Pandas script to process vehicle data.
    The generated script MUST fetch the JSON from the vehicle API itself using httpx (or requests),
    parse into pandas, compute the requested metric, and print a concise result.
    """
    suggested = ", ".join(suggested_signals) if (suggested_signals := suggested_signals) else ""
    prompt = f"""
You are a data analysis expert. Generate a Python script that:

1. Use ONLY signals from the allowed list (separated by ", "): {SIGNALS}
   Suggested signals based on the user query: {suggested}
   Build the signals parameter as a string list (comma-separated, no spaces) and use the suggested ones when appropriate.
   Use the provided helper `build_url(signals: list[str])` to construct the URL. Do not use data from any other API. Do not create mock data. Do not make up data without fetching the API. If you do not use the build_url and http_get helpers then do not return any data.
   Then use `http_get(url)` to fetch the JSON.
2. Computes the metric requested by the user query below
3. Prints a clear, single-line result string for the user

User Query: "{json.dumps(query)}" (This variable is not defined, you have to define it yourself with this string value)

Support advanced queries like:
- "top 10 mobile_speed values"
- "compare average mobile_speed between trip 4 and trip 5"
- "min/median/max of mobile_speed"
 - "correlation between temperature and voltage signals"

Graphing:
- If the user asks to graph/plot (keywords: graph, plot, vs), create a Matplotlib plot (import matplotlib.pyplot as plt).
- Label axes with units when applicable (Voltage (V), Temperature (C), Speed (m/s) if derivable).
- Use reasonable figure size and grid; add title.
- Save the plot to a PNG buffer (io.BytesIO), base64-encode it, and call set_image_base64(encoded_str).
- Still print a brief one-line caption summarizing what was plotted.

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

    script = await call_gemini(prompt)
    if DEBUG_ANALYSIS:
        logger.info(f"Generated script length={len(script)} for query='{query[:60]}'... suggested={suggested}")
    return script

async def execute_pandas_script(script: str) -> tuple[str, Dict[str, Any], Optional[str]]:
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
        captured: Dict[str, Any] = {"__captured_result": None, "__image_base64": None}
        def set_result(value: Any) -> None:
            try:
                captured["__captured_result"] = value
            except Exception:
                captured["__captured_result"] = None
        def set_image_base64(value: Any) -> None:
            try:
                captured["__image_base64"] = value
            except Exception:
                captured["__image_base64"] = None

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
            'set_image_base64': set_image_base64,
            'http_get': http_get,
            'build_url': build_url,
            # 'get_possible_signals': get_possible_signals,
            'io': __import__('io'),
            'base64': __import__('base64'),
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
            return result, debug, captured.get("__image_base64")

        # Fallback: try to read common result variables if nothing was printed
        for var_name in ("result", "answer", "output"):
            if var_name in safe_globals and safe_globals[var_name] is not None:
                try:
                    debug["fallback_var"] = var_name
                    return str(safe_globals[var_name]), debug, captured.get("__image_base64")
                except Exception:
                    continue
        # Fallback 2: captured result via set_result
        if captured.get("__captured_result") is not None:
            debug["fallback_var"] = "__captured_result"
            return str(captured["__captured_result"]), debug, captured.get("__image_base64")

        debug["reason"] = "no stdout and no fallback variable"
        return "No output generated", debug, captured.get("__image_base64")
        
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

        # Clarify ambiguous cell queries without a cell number (e.g., "give me cell temp")
        ml = message.lower()
        if re.search(r"\bcell\b", ml) and (
            re.search(r"\btemp(?:erature)?\b", ml) or re.search(r"\bvolt(?:age)?\b", ml)
        ) and not re.search(r"\b\d{1,3}\b", ml):
            metric = "voltage" if re.search(r"\bvolt", ml) else "temperature"
            return ChatResponse(
                success=True,
                message=f"Which cell number for {metric}? e.g., 16 or 110",
                data={"intent": "clarify_cell_metric", "metric": metric}
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
                # Select suggested signals for the query
                suggested_pairs = best_signals_for_query(message, max_signals=6, threshold=100.0)
                suggested_signals = [s for s, sc in suggested_pairs]
                # Build detailed explanations for top matches (for logging)
                explained_all = [
                    (sig,)+score_signal_with_details(message, sig) for sig in KNOWN_SIGNALS
                ] if KNOWN_SIGNALS else []
                explained_all.sort(key=lambda x: x[1])  # sort by score
                top_explained = explained_all[:10]
                if top_explained:
                    logger.info(
                        "Signal scoring top matches: " +
                        ", ".join(f"{sig}={details['final']} (ratio={details['ratio']}, bonuses={details['bonuses_hit']}, nums={details['numeric_hits']})" for sig, score, details in top_explained)
                    )
                if not suggested_signals:
                    return ChatResponse(
                        success=True,
                        message="Sorry, I can't help you with that. I can only assist with vehicle data queries.",
                    )
                pandas_script = await generate_pandas_script(message, suggested_signals)
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
            result, debug_info, image_b64 = await execute_pandas_script(pandas_script)
            # Include signal scoring info for transparency
            if 'top_explained' not in locals():
                explained_all = []
                top_explained = []
            signal_scoring_payload = [
                {"signal": sig, **details} for sig, score, details in (top_explained if top_explained else [])
            ]
            data_payload: Dict[str, Any] = {
                "script": pandas_script,
                "debug": debug_info,
                "signal_scoring": {
                    "selected": suggested_pairs if 'suggested_pairs' in locals() else [],
                    "top": signal_scoring_payload,
                },
            }
            if image_b64:
                data_payload["image_base64"] = image_b64
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
