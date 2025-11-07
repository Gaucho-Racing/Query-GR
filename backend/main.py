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
import mysql.connector  # requires mysql-connector-python
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

def extract_trip_id(message: str) -> Optional[str]:
    ml = message.lower()
    # patterns: "trip 3", "run 3", "in run 3", "3rd run", "third trip"
    m = re.search(r"\btrip\s+(\d{1,3})\b", ml)
    if m:
        return m.group(1)
    m = re.search(r"\brun\s+(\d{1,3})\b", ml)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{1,3})(?:st|nd|rd|th)\s+(?:run|trip)\b", ml)
    if m:
        return m.group(1)
    ord_map = {
        'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'sixth':6,'seventh':7,'eighth':8,'ninth':9,'tenth':10,
        'eleventh':11,'twelfth':12,'thirteenth':13,'fourteenth':14,'fifteenth':15,'sixteenth':16,'seventeenth':17,
        'eighteenth':18,'nineteenth':19,'twentieth':20
    }
    m = re.search(r"\b(" + "|".join(ord_map.keys()) + r")\s+(?:run|trip)\b", ml)
    if m:
        return str(ord_map[m.group(1)])
    return None

# Script generation cache (to reduce OpenRouter calls)
SCRIPT_CACHE_TTL_SECONDS = 3600  # 1 hour
_script_cache: Dict[str, Dict[str, Any]] = {}

"""Signals are sourced from the database and cached in-memory."""

# DB configuration for signals list (optional; used after clarifications)
DB_HOST = os.getenv("DATABASE_HOST", "")
DB_PORT = int(os.getenv("DATABASE_PORT", "3306"))
DB_USER = os.getenv("DATABASE_USER", "")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD", "")
DB_NAME = os.getenv("DATABASE_NAME", "")

DB_SIGNALS_CACHE: List[str] = []

def refresh_db_signals_cache() -> int:
    """Fetch distinct signal names into DB_SIGNALS_CACHE. Returns count or 0 on failure."""
    global DB_SIGNALS_CACHE
    if not (DB_HOST and DB_USER and DB_NAME):
        return 0
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            connection_timeout=10,
        )
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT name FROM `signal` LIMIT 9999;")
            rows = cursor.fetchall()
            names = [r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip()]
            DB_SIGNALS_CACHE = sorted(set(names))
            logger.info(f"Loaded {len(DB_SIGNALS_CACHE)} signals from DB {DB_NAME}")
            return len(DB_SIGNALS_CACHE)
        finally:
            try:
                cursor.close()
            except Exception:
                pass
            conn.close()
    except Exception as e:
        logger.warning(f"Failed to load signals from DB: {e}")
        return 0

def get_known_signals() -> List[str]:
    if not DB_SIGNALS_CACHE:
        refresh_db_signals_cache()
    return DB_SIGNALS_CACHE


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


def best_signals_for_query(
    query: str,
    max_signals: int = 6,
    threshold: float = 100.0,
    pool: Optional[List[str]] = None,
    filter_pred: Optional[callable] = None,
) -> List[Tuple[str, float]]:
    signals_source = pool if pool is not None else get_known_signals()
    if not signals_source:
        return []
    if filter_pred:
        signals_source = [s for s in signals_source if filter_pred(s)]
    scored = [(sig, score_signal(query, sig)) for sig in signals_source]
    scored.sort(key=lambda x: x[1])
    return [(sig, sc) for sig, sc in scored[:max_signals] if sc <= threshold]


def infer_specific_signals(message: str, signals_pool: List[str]) -> List[str]:
    ml = message.lower()
    # detect cell number
    m = re.search(r"\bcell\s*(\d{1,3})\b", ml)
    cell_num = m.group(1) if m else None
    wants_temp = bool(re.search(r"\btemp(?:erature)?\b", ml))
    wants_volt = bool(re.search(r"\bvolt(?:age)?\b", ml))
    selected: List[str] = []
    if cell_num:
        if wants_temp:
            candidate = f"acu_cell{cell_num}_temp"
            if candidate in signals_pool:
                selected.append(candidate)
        if wants_volt:
            candidate = f"acu_cell{cell_num}_voltage"
            if candidate in signals_pool:
                selected.append(candidate)
    return selected


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

SIGNALS = ", ".join(get_known_signals())
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
    suggested = ", ".join(suggested_signals) if suggested_signals else ""
    prompt = f"""
You are a data analysis expert. Generate a Python script that:

1. Use EXACTLY these signals (comma-separated, no spaces): {suggested}
   Do not invent or rename signals; if a suggested signal is empty, return a one-line message that no matching signals were found.
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
- The only defined globals are: pd json httpx print set_result set_image_base64 http_get build_url parse_series parse_series_df io base64
- Parse the JSON robustly. Possible shapes include:
  * {{"signals": {{"mobile_speed": [...]}}}}
  * {{"data": {{"mobile_speed": [...]}}}}
  * {{"data": {{"data": [{{...}}] }}}}
  * {{"mobile_speed": [...]}}
  * A list of dicts containing fields like "mobile_speed", "value" or similar.
  Use the provided parse_series(payload, [signals]) helper to flatten robustly into numeric series.
- Clean the series: coerce to numeric, drop nulls, then compute the metric requested by the user query (e.g., average/mean, min, max, median).
 - For correlation tasks, align signals on produced_at timestamps when applicable; compute Pearson correlation.
- At the end, set a variable named `result` to the final one-line string AND call both `print(result)` and `set_result(result)` (available at runtime).
- Do not include explanations or markdown. Output ONLY executable Python code.
    """

    script = await call_gemini(prompt)
    if DEBUG_ANALYSIS:
        logger.info(f"Generated script length={len(script)} for query='{query[:60]}'... suggested={suggested}")
    return script

async def execute_pandas_script(script: str, trip_id_override: Optional[str] = None) -> tuple[str, Dict[str, Any], Optional[str]]:
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

        def build_url(signals: list[str] | str, *args, **kwargs):
            # Accept optional trip_id passed by generated code, but prefer override
            incoming_trip = kwargs.get('trip_id')
            # Normalize signals input
            if isinstance(signals, str):
                sig_param = ",".join([p.strip() for p in signals.split(',') if p.strip()])
            else:
                sig_param = ",".join(s.strip() for s in signals if s and isinstance(s, str))
            trip_to_use = trip_id_override or incoming_trip or TRIP_ID
            base = (
                f"{VEHICLE_API_URL}?vehicle_id={VEHICLE_ID}&trip_id={trip_to_use}"
                f"&signals={sig_param}&token={VEHICLE_API_TOKEN}"
            )
            logger.info(f"AI script build_url: {base}")
            return base

        def parse_series(payload: Any, signals: list[str] | str):
            """Parse payload into pandas Series.
            - If a single signal name (string without commas) is provided, returns a pandas.Series
            - If multiple signals (list or comma-separated string), returns a dict {signal: Series}
            All Series are coerced to numeric and NaNs dropped.
            """
            try:
                if isinstance(signals, str):
                    raw_parts = [p.strip() for p in signals.split(',') if p.strip()]
                    single = len(raw_parts) == 1
                    sigs = raw_parts
                else:
                    single = len(signals) == 1
                    sigs = [s.strip() for s in signals if s]
            except Exception:
                sigs = []
                single = False
            rows = None
            if isinstance(payload, dict):
                try:
                    if isinstance(payload.get('data'), dict) and isinstance(payload['data'].get('data'), list):
                        rows = payload['data']['data']
                    elif isinstance(payload.get('data'), list):
                        rows = payload['data']
                    elif isinstance(payload.get('signals'), dict):
                        # flatten signals dict of arrays into row-wise table if lengths match
                        sig_dict = payload['signals']
                        max_len = max((len(v) for v in sig_dict.values() if isinstance(v, list)), default=0)
                        rows = [ {k: (sig_dict.get(k)[i] if i < len(sig_dict.get(k, [])) else None) for k in sig_dict.keys()} for i in range(max_len) ]
                    else:
                        rows = []
                except Exception:
                    rows = []
            elif isinstance(payload, list):
                rows = payload
            else:
                rows = []
            try:
                df = pd.DataFrame(rows)
            except Exception:
                df = pd.DataFrame()
            # build outputs
            if single and sigs:
                sig = sigs[0]
                series = None
                if sig in df.columns:
                    series = pd.to_numeric(df[sig], errors='coerce').dropna()
                else:
                    for col in df.columns:
                        if isinstance(col, str) and col.lower() == sig.lower():
                            series = pd.to_numeric(df[col], errors='coerce').dropna()
                            break
                return series if series is not None else pd.Series(dtype='float64')
            else:
                out: Dict[str, pd.Series] = {}
                for sig in sigs:
                    if sig in df.columns:
                        s = pd.to_numeric(df[sig], errors='coerce').dropna()
                        out[sig] = s
                    else:
                        # try case-insensitive column match
                        for col in df.columns:
                            if isinstance(col, str) and col.lower() == sig.lower():
                                s = pd.to_numeric(df[col], errors='coerce').dropna()
                                out[sig] = s
                                break
                return out

        def parse_series_df(payload: Any, signals: list[str] | str):
            """Return a DataFrame from parse_series. Columns are signal names.
            Returns empty DataFrame if nothing parsed.
            """
            sd = parse_series(payload, signals)
            try:
                df = pd.DataFrame(sd)
            except Exception:
                df = pd.DataFrame()
            return df
        
        # signals are sourced from DB cache only; no CSV fallback here

        safe_globals = {
            'pd': pd,
            'json': json,
            'httpx': httpx,
            'print': print,
            'set_result': set_result,
            'parse_series': parse_series,
            'parse_series_df': parse_series_df,
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

        # Clarify trip/run if user did not specify any trip id
        chosen_trip = extract_trip_id(message)
        if chosen_trip is None:
            return ChatResponse(
                success=True,
                message="Which trip (run) number? e.g., 3",
                data={"intent": "clarify_trip"}
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
            # After clarifications, initialize signals from DB (once per process)
            if not DB_SIGNALS_CACHE:
                refresh_db_signals_cache()

            # Cache by normalized query to reduce model usage
            normalized = message.strip().lower()
            cached = _script_cache.get(normalized)
            now = time()
            if cached and (now - cached["ts"]) < SCRIPT_CACHE_TTL_SECONDS:
                pandas_script = cached["script"]
            else:
                # Select suggested signals for the query
                pool = get_known_signals()
                # Try exact inference for cell+metric first
                exact = infer_specific_signals(message, pool)
                suggested_pairs: List[Tuple[str, float]] = []
                if exact:
                    suggested_signals = exact
                else:
                    # metric filtering
                    wants_temp = bool(re.search(r"\btemp(?:erature)?\b", message.lower()))
                    wants_volt = bool(re.search(r"\bvolt(?:age)?\b", message.lower()))
                    pred = None
                    if wants_temp and not wants_volt:
                        pred = lambda s: isinstance(s, str) and s.endswith("_temp")
                    elif wants_volt and not wants_temp:
                        pred = lambda s: isinstance(s, str) and s.endswith("_voltage")
                    # special: correlation or vs chooses top 2
                    wants_two = bool(re.search(r"\bvs\b|correl", message.lower()))
                    pairs = best_signals_for_query(message, max_signals=(2 if wants_two else 6), threshold=100.0, filter_pred=pred, pool=pool)
                    suggested_pairs = pairs
                    # Choose the lowest score first (enforce top-1 unless two-signals case)
                    suggested_signals = [p[0] for p in pairs[: (2 if wants_two else 1)]] or []
                # Build detailed explanations for top matches (for logging)
                signals_source = pool
                explained_all = [
                    (sig,) + score_signal_with_details(message, sig)
                    for sig in signals_source
                ] if signals_source else []
                explained_all.sort(key=lambda x: x[1])  # sort by score
                top_explained = explained_all[:10]
                if top_explained:
                    logger.info(
                        "Signal scoring top matches: "
                        + ", ".join(
                            f"{sig}={details['final']} (ratio={details['ratio']}, bonuses={details['bonuses_hit']}, nums={details['numeric_hits']})"
                            for sig, score, details in top_explained
                        )
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
            # use clarified/parsed trip id
            chosen_trip = extract_trip_id(message) or TRIP_ID
            result, debug_info, image_b64 = await execute_pandas_script(pandas_script, trip_id_override=chosen_trip)
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
                "trip_id_used": chosen_trip,
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
