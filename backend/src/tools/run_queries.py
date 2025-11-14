"""Query execution logic for vehicle data analysis."""
import re
import json
import logging
import httpx
import pandas as pd
import signal
import io
import sys
from time import time
from typing import Dict, Any, List, Tuple, Optional
from contextlib import redirect_stdout, redirect_stderr
import os
from dotenv import load_dotenv

from .utils import (
    extract_trip_id,
    get_known_signals,
    best_signals_for_query,
    infer_specific_signals,
    score_signal,
    score_signal_with_details,
    expand_trip_ids,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Vehicle data API configuration
VEHICLE_API_URL = "https://mapache.gauchoracing.com/api/query/signals"
VEHICLE_API_TOKEN = "01b3939d-678f-44ac-93ff-0d54e09ba3d6"
VEHICLE_ID = "gr24-main"
TRIP_ID = "4"
VEHICLE_DATA_TIMEOUT = float(os.getenv("VEHICLE_DATA_TIMEOUT", "120"))  # 120 seconds (2 minutes)
SCRIPT_TIMEOUT = int(os.getenv("SCRIPT_TIMEOUT", "120"))  # 120 seconds (2 minutes)
DEBUG_ANALYSIS = os.getenv("DEBUG_ANALYSIS", "false").lower() == "true"


async def generate_pandas_script(query: str, suggested_signals: List[str], gemini_call_fn, trip_ids: List[str] = None) -> str:
    """Generate a Pandas script to process vehicle data.
    The generated script MUST fetch the JSON from the vehicle API itself using httpx (or requests),
    parse into pandas, compute the requested metric, and print a concise result.
    """
    suggested = ", ".join(suggested_signals) if suggested_signals else ""
    
    # Log signal mappings for debugging
    if "tcm_power_draw" in suggested_signals:
        logger.info(f"[SIGNAL MAPPING] tcm_power_draw provided for query mentioning motor current/current draw")
    if any("acu_cell" in sig and "_temp" in sig for sig in suggested_signals):
        logger.info(f"[SIGNAL MAPPING] ACU cell temp signals provided for query mentioning battery temperature")
    trip_ids_str = ", ".join(trip_ids) if trip_ids and len(trip_ids) > 1 else ""
    multi_trip_instruction = ""
    if trip_ids and len(trip_ids) > 1:
        multi_trip_instruction = f"""
MULTIPLE TRIPS: This query needs to process data from {len(trip_ids)} trip(s): {trip_ids_str}
- Loop through each trip_id and fetch data using build_url(signals, trip_id=trip_id) for each trip
- IMPORTANT: After fetching data for each trip, add a 'trip_id' column to the DataFrame with the current trip_id value
  Example: df['trip_id'] = trip_id  # Add trip_id column before combining
- Combine all data from all trips into a single DataFrame using pd.concat() or similar
- Then apply filters and compute metrics across all trips
- The trip_id column will help users see which trip each row belongs to
- Example for multiple trips:
  all_dfs = []
  for trip_id in [{trip_ids_str}]:
      url = build_url(signals, trip_id=trip_id)
      resp = http_get(url)
      df = parse_full_df(resp.json())
      df['trip_id'] = trip_id  # Add trip_id column
      all_dfs.append(df)
  combined_df = pd.concat(all_dfs, ignore_index=True)
"""
    
    prompt = f"""
You are a data analysis expert. Generate a Python script that:

1. Use EXACTLY these signals (comma-separated, no spaces): {suggested}
   
   ⚠️ CRITICAL: SIGNAL SELECTION IS NOT YOUR JOB ⚠️
   
   - Signal selection has ALREADY been done by our signal scoring system (NOT by you).
   - Your ONLY job is to use the provided signals to write a pandas script.
   - You do NOT choose signals. You do NOT validate signals. You do NOT check if signals match the query.
   - The signals above have been PRE-SELECTED by our system - they ARE correct, even if names don't match query text.
   
   ⚠️ YOU MUST USE THE PROVIDED SIGNALS - DO NOT:
   - Check if signal names match the query text (signal selection is NOT your job)
   - Generate error messages like "no matching signals" or "Query cannot be completed"
   - Try to find or choose different signals
   - Validate whether signals are correct
   - Return errors about missing signals
   - Return generic "No data found for the specified trips and signals" messages
   - Say "Voltage data was not available" or "Temperature data was not available" - if signals were provided, the data EXISTS
   - Check if columns exist before using them - just use the provided signals directly (they ARE in the data)
   
   ✅ YOUR JOB IS TO:
   - Take the signals provided above
   - Use them to fetch data from the API
   - Write pandas code to analyze the data
   - Check the ACTUAL DATA VALUES (not signal names)
   - IMPORTANT: The signals provided above ARE in the fetched data - use them directly (e.g., df['acu_cell1_voltage'], df['tcm_power_draw'])
   - DO NOT check if columns exist - if signals were provided, they ARE in the DataFrame columns
   - DO NOT say "voltage data was not available" or "temperature data was not available" - the data EXISTS if signals were provided
   - IMPORTANT: Even 1 data point is VALID DATA and should be shown. Only say "no data" if there are ZERO data points.
   - If data exists but no rows meet criteria (e.g., temp > 45°C AND voltage > 3V), say "No instances found where [condition]"
   - If data is truly empty (ZERO rows), say "No data available for the selected signals"
   - For "show all" queries, use set_table_data() even if there's only 1 row - display ALL available data
   - Return results based on ACTUAL DATA ANALYSIS, not signal name matching
   
   SIGNAL MAPPINGS (for your reference - signals are already selected):
   - If query mentions "motor current draw" → tcm_power_draw is provided (they are the same thing)
   - If query mentions "battery temperature" → acu_cell*_temp signals are provided (battery = ACU cells)
   - Signal names may not match query text - THIS IS EXPECTED. Use the provided signals as-is.
   
   EXAMPLE:
   - Query: "motor current draw was above 300A"
   - Signals provided: ["tcm_power_draw", ...]
   - Your job: Use tcm_power_draw to fetch data and check if values > 300
   - NOT your job: Check if "motor current draw" matches "tcm_power_draw" (already done by our system)
   
   IMPORTANT: Do NOT include metadata fields like 'run_id', 'produced_at', 'trip_id', 'vehicle_id', or 'token' in the signals list.
   These are NOT signals - they are metadata or query parameters. Only use actual sensor signal names provided above.
   
   CONTEXT FOR ANALYSIS (not signal selection - signals already chosen):
   - Numbers in queries like "45°C" or "300A" are THRESHOLDS, not cell numbers
   - Extract the NUMERIC VALUE from thresholds: "45°C" → 45, "300A" → 300, "45 C" → 45, "300 A" → 300
   - The data values from the API are NUMERIC (unitless), so compare numeric values directly
   - Example: Query says "temperature exceeded 45°C", signal is "acu_cell1_temp" → extract 45, then check if acu_cell1_temp > 45
   - Example: Query says "current draw was above 300A", signal is "tcm_power_draw" → extract 300, then check if tcm_power_draw > 300
   - IMPORTANT: Data values are already numeric - do NOT look for units in the data, just compare the numeric values
   
   Use the provided helper `build_url(signals: list[str], trip_id=X)` to construct the URL. You can pass trip_id as a keyword argument to build_url.
   Then use `http_get(url)` to fetch the JSON.
   Do not use data from any other API. Do not create mock data. Do not make up data without fetching the API.
{multi_trip_instruction}
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

Tables for "Show all" queries:
- If the query asks to "show all", "list all", "find all", or returns ANY rows of data (even just 1 row), use set_table_data() instead of just printing.
- IMPORTANT: Use set_table_data() even if there's only 1 data point - display ALL available data, no matter how small.
- set_table_data() accepts a pandas DataFrame or list of dicts. For example:
  * set_table_data(df) where df is a pandas DataFrame
  * set_table_data([{{"col1": val1, "col2": val2}}, {{"col1": val3, "col2": val4}}])
- The table will be displayed nicely formatted in the UI. Still print a brief summary message.
- Format numeric values appropriately (round to reasonable precision, format dates/times clearly).
- DO NOT say "no data" if there are ANY rows (even 1) - only say "no data" if DataFrame is completely empty (0 rows).

EXAMPLE: If you filter data and get 1 row that meets criteria:
  - CORRECT: 
    threshold = 45  # Extract numeric value from "45°C"
    filtered_df = df[df['temp'] > threshold]  # Compare numeric values
    set_table_data(filtered_df)
    print(f"Found {{len(filtered_df)}} instance(s) where temperature exceeded 45°C")
  - WRONG: filtered_df = df[df['temp'] > "45°C"]  # Don't compare strings!
  - WRONG: filtered_df = df[df['temp'] > 45]; if len(filtered_df) == 0: print("No data available")  # Don't do this!
  
EXAMPLE: Query "motor current draw was above 300A":
  - CORRECT:
    threshold = 300  # Extract numeric value from "300A"
    filtered_df = df[df['tcm_power_draw'] > threshold]  # Compare numeric values
    set_table_data(filtered_df)
    print(f"Found {{len(filtered_df)}} instance(s) where motor current draw exceeded 300A")
  - WRONG: df[df['tcm_power_draw'] > "300A"]  # Data is numeric, not strings with units!
  - WRONG: if 'tcm_power_draw' not in df.columns: print("Current data not available")  # Don't check - signal was provided, so it EXISTS!

EXAMPLE: Query "battery temperature > 25°C AND cell voltages > 3V":
  - CORRECT:
    temp_threshold = 25  # Extract from "25°C"
    volt_threshold = 3  # Extract from "3V"
    # Use the provided signals directly - they ARE in the DataFrame
    temp_cols = [col for col in df.columns if col.endswith('_temp') and 'acu_cell' in col]
    volt_cols = [col for col in df.columns if col.endswith('_voltage') and 'acu_cell' in col]
    # Filter rows where ANY temp > threshold AND ANY voltage > threshold
    filtered_df = df[(df[temp_cols].max(axis=1) > temp_threshold) & (df[volt_cols].max(axis=1) > volt_threshold)]
    set_table_data(filtered_df)
    print(f"Found {{len(filtered_df)}} instance(s) where battery temp > 25°C AND cell voltage > 3V")
  - WRONG: if not volt_cols: print("Voltage data was not available")  # Don't do this! Signals were provided, data EXISTS!
  - WRONG: if 'acu_cell1_voltage' not in df.columns: print("Voltage data was not available")  # Don't check columns!
  
EXAMPLE: If you have 2 signals with 1 data point each:
  - CORRECT: Combine into DataFrame with 1 row, call set_table_data(df), print summary
  - WRONG: Say "no data available" - 1 data point IS data and should be shown!

Make your script resilient and efficient.

Requirements:
- Use: import pandas as pd
- Use the provided helper: http_get(url) instead of calling httpx directly (this logs the URL and returns a Response for .json()).
- Make the HTTP GET within the script to fetch the JSON data.
- Do not use any external variables; construct the URL with params inside the script.
- The only defined globals are: pd json httpx print set_result set_image_base64 set_table_data http_get build_url parse_series parse_series_df parse_full_df io base64
- Parse the JSON robustly. Possible shapes include:
  * {{"signals": {{"mobile_speed": [...]}}}}
  * {{"data": {{"mobile_speed": [...]}}}}
  * {{"data": {{"data": [{{...}}] }}}}
  * {{"mobile_speed": [...]}}
  * A list of dicts containing fields like "mobile_speed", "value" or similar.
  
Parsing helpers:
- parse_series(payload, signals): Returns numeric Series (single signal) or dict of Series (multiple signals). Use for simple numeric operations.
- parse_series_df(payload, signals): Returns DataFrame with only signal columns. Use when you need a DataFrame but don't need timestamps.
- parse_full_df(payload): Returns DataFrame with ALL columns including 'produced_at' (timestamp) and metadata. USE THIS when:
  * Joining data from multiple trips (need timestamps to align)
  * Doing time-based operations
  * Need to sort/filter by time
  * Need 'produced_at' for correlation or time alignment
  
IMPORTANT: 'produced_at' is a timestamp column that exists in the data but should NOT be in the signals list for build_url().
- When you fetch data, the response includes 'produced_at' automatically - you don't need to request it as a signal.
- Use parse_full_df() to access 'produced_at' for time-based operations.
- Clean numeric columns: coerce to numeric, drop nulls, then compute the metric requested by the user query (e.g., average/mean, min, max, median).
- IMPORTANT: When filtering by thresholds (e.g., "> 45°C" or "> 300A"), extract the numeric value from the query:
  * Use: threshold_temp = 45  # extracted from "45°C" or "45 C"
  * Use: threshold_current = 300  # extracted from "300A" or "300 A"
  * Then filter: df[df['signal_name'] > threshold_temp]  # Compare numeric values, not strings
  * The data values are already numeric (unitless), so compare directly: df['temp'] > 45, not df['temp'] > "45°C"
  
- IMPORTANT: When multiple ACU cell signals are provided (e.g., acu_cell0_temp, acu_cell1_temp, ..., acu_cell49_temp):
  * Find all matching columns: temp_cols = [col for col in df.columns if col.endswith('_temp') and 'acu_cell' in col]
  * Find all voltage columns: volt_cols = [col for col in df.columns if col.endswith('_voltage') and 'acu_cell' in col]
  * Use these column lists directly - they WILL exist if signals were provided
  * DO NOT check if columns exist - if signals were provided, they ARE in the DataFrame
  * Example: df[temp_cols].max(axis=1) > threshold  # Check if ANY temp exceeds threshold
- For correlation tasks, align signals on produced_at timestamps when applicable; compute Pearson correlation.
- At the end, set a variable named `result` to the final one-line string AND call both `print(result)` and `set_result(result)` (available at runtime).
- Do not include explanations or markdown. Output ONLY executable Python code.

⚠️ FINAL REMINDER: 
   - Signal selection is NOT your job - it's already done by our signal scoring system (in tools/utils.py).
   - Your ONLY job is to write pandas code using the provided signals.
   - DO NOT generate error messages about missing signals or "no data found for specified trips and signals".
   - DO NOT check if signals match query text.
   - Use the provided signals directly - signal names not matching query text is EXPECTED and NORMAL.
   - FETCH THE DATA FIRST, then check if values meet criteria (e.g., temp > 45°C, current > 300A).
   - IMPORTANT: The signals provided above ARE in the DataFrame - use them directly without checking if columns exist.
   - DO NOT say "voltage data was not available" or "temperature data was not available" - if signals were provided, the data EXISTS.
   - IMPORTANT: Even 1 data point is VALID DATA. Display it using set_table_data() or print it.
   - If data exists but no rows meet criteria, report that clearly (e.g., "No instances found where battery temperature exceeded 45°C AND cell voltage exceeded 3V").
   - Only say "no data" if the API returns ZERO data points (empty DataFrame with 0 rows), not if there are 1+ data points.
   - For "show all" queries: If there are ANY rows (even 1), use set_table_data() to display them. Do NOT say "no data available".
   - Example: Query says "motor current draw", signal provided is "tcm_power_draw" → USE tcm_power_draw directly (signal selection already done by our system).
   - Example: Query mentions "cell voltages", signals provided include "acu_cell*_voltage" → USE those columns directly, don't check if they exist.
    """

    script = await gemini_call_fn(prompt)
    if DEBUG_ANALYSIS:
        logger.info(f"Generated script length={len(script)} for query='{query[:60]}'... suggested={suggested}")
    return script


async def execute_pandas_script(script: str, trip_id_override: Optional[str] = None) -> Tuple[str, Dict[str, Any], Optional[str], Optional[Dict[str, Any]]]:
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
        captured: Dict[str, Any] = {"__captured_result": None, "__image_base64": None, "__table_data": None}
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
        def set_table_data(value: Any) -> None:
            """Set table data for "show all" queries. Accepts pandas DataFrame or list of dicts."""
            try:
                if hasattr(value, 'to_dict'):  # pandas DataFrame
                    # Convert DataFrame to list of dicts with proper formatting
                    df_dict = value.to_dict('records')
                    captured["__table_data"] = {
                        "columns": list(value.columns),
                        "rows": df_dict
                    }
                elif isinstance(value, list):
                    # List of dicts
                    if value and isinstance(value[0], dict):
                        captured["__table_data"] = {
                            "columns": list(value[0].keys()),
                            "rows": value
                        }
                    else:
                        captured["__table_data"] = None
                elif isinstance(value, dict) and "columns" in value and "rows" in value:
                    # Already formatted table data
                    captured["__table_data"] = value
                else:
                    captured["__table_data"] = None
            except Exception:
                captured["__table_data"] = None

        def http_get(url: str):
            logger.info(f"AI script HTTP GET: {url}")
            try:
                resp = httpx.get(url, timeout=VEHICLE_DATA_TIMEOUT)
                logger.info(f"[HTTP RESPONSE] Status: {resp.status_code}, Content-Length: {len(resp.content)} bytes")
                resp.raise_for_status()  # Raise exception for HTTP errors
                
                # Check if response contains data
                try:
                    data = resp.json()
                    logger.info(f"[HTTP DATA] Response type: {type(data).__name__}")
                    
                    # Log data structure details
                    if isinstance(data, dict):
                        logger.info(f"[HTTP DATA] Dict keys: {list(data.keys())[:10]}")
                        # Check various empty data patterns
                        signals_data = data.get('signals', {}) or data.get('data', {})
                        if isinstance(signals_data, dict):
                            signal_count = len(signals_data)
                            total_rows = sum(len(v) if isinstance(v, list) else 1 for v in signals_data.values() if v)
                            logger.info(f"[HTTP DATA] Found {signal_count} signals, total data points: {total_rows}")
                            if not signals_data or all(not v or (isinstance(v, list) and len(v) == 0) for v in signals_data.values()):
                                logger.warning(f"[EMPTY DATA] HTTP GET returned empty data for URL: {url}")
                            else:
                                # Log sample signal names and row counts
                                sample_signals = list(signals_data.keys())[:5]
                                sample_info = {sig: len(v) if isinstance(v, list) else "non-list" for sig, v in list(signals_data.items())[:5]}
                                logger.info(f"[HTTP DATA] Sample signals: {sample_signals}, row counts: {sample_info}")
                        elif isinstance(signals_data, list):
                            logger.info(f"[HTTP DATA] List length: {len(signals_data)}")
                            if len(signals_data) == 0:
                                logger.warning(f"[EMPTY DATA] HTTP GET returned empty list for URL: {url}")
                            else:
                                logger.info(f"[HTTP DATA] First item type: {type(signals_data[0]).__name__}, sample keys: {list(signals_data[0].keys())[:5] if isinstance(signals_data[0], dict) else 'N/A'}")
                    elif isinstance(data, list):
                        logger.info(f"[HTTP DATA] List length: {len(data)}")
                        if len(data) == 0:
                            logger.warning(f"[EMPTY DATA] HTTP GET returned empty list for URL: {url}")
                        else:
                            logger.info(f"[HTTP DATA] First item type: {type(data[0]).__name__}, sample keys: {list(data[0].keys())[:5] if isinstance(data[0], dict) else 'N/A'}")
                    else:
                        logger.info(f"[HTTP DATA] Response data type: {type(data).__name__}, value preview: {str(data)[:200]}")
                except Exception as json_err:
                    logger.error(f"[HTTP PARSE ERROR] Could not parse JSON response from {url}: {json_err}")
                    logger.error(f"[HTTP PARSE ERROR] Response text preview: {resp.text[:500]}")
                
                return resp
            except httpx.HTTPStatusError as e:
                logger.error(f"[HTTP ERROR] HTTP {e.response.status_code} error for URL {url}")
                logger.error(f"[HTTP ERROR] Response text: {e.response.text[:500]}")
                raise
            except httpx.TimeoutException as e:
                logger.error(f"[HTTP TIMEOUT] Request timed out for URL {url}")
                raise
            except Exception as e:
                logger.error(f"[HTTP ERROR] Unexpected error fetching {url}: {e}")
                raise

        def build_url(signals: list[str] | str, *args, **kwargs):
            # Accept optional trip_id passed by generated code, but prefer override
            incoming_trip = kwargs.get('trip_id')
            # Normalize signals input
            if isinstance(signals, str):
                sig_list = [p.strip() for p in signals.split(',') if p.strip()]
            else:
                sig_list = [s.strip() for s in signals if s and isinstance(s, str)]
            # Filter out metadata fields that are NOT signals
            # These are query parameters or metadata fields, NOT actual sensor signals
            metadata_fields = [
                'trip_id', 'trip', 'tripid',
                'run_id', 'runid', 'run',
                'vehicle_id', 'vehicleid', 'vehicle',
                'produced_at', 'producedat', 'timestamp', 'time',
                'token'
            ]
            sig_list = [s for s in sig_list if s.lower() not in metadata_fields]
            
            if not sig_list:
                raise ValueError("No valid signals provided after filtering metadata fields")
            
            sig_param = ",".join(sig_list)
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
        
        def parse_full_df(payload: Any):
            """Parse payload into a full DataFrame with ALL columns including metadata like 'produced_at'.
            This preserves timestamps and other metadata fields that are useful for time-based operations.
            Returns a DataFrame with all columns from the payload.
            """
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
                # Convert produced_at to datetime if it exists
                if 'produced_at' in df.columns:
                    df['produced_at'] = pd.to_datetime(df['produced_at'], errors='coerce')
            except Exception:
                df = pd.DataFrame()
            return df
        
        safe_globals = {
            'pd': pd,
            'json': json,
            'httpx': httpx,
            'print': print,
            'set_result': set_result,
            'parse_series': parse_series,
            'parse_series_df': parse_series_df,
            'parse_full_df': parse_full_df,
            'set_image_base64': set_image_base64,
            'set_table_data': set_table_data,
            'http_get': http_get,
            'build_url': build_url,
            'io': __import__('io'),
            'base64': __import__('base64'),
        }
        
        # Capture the output
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
        
        # Log script execution results
        logger.info(f"[SCRIPT EXECUTION] Completed in {duration_ms}ms")
        logger.info(f"[SCRIPT EXECUTION] Stdout length: {len(result)} chars, Stderr length: {len(stderr)} chars")
        if result:
            logger.info(f"[SCRIPT EXECUTION] Result preview: {result[:300]}")
        if stderr:
            logger.warning(f"[SCRIPT EXECUTION] Stderr: {stderr[:500]}")
        
        debug: Dict[str, Any] = {
            "sanitized_len": len(sanitized),
            "stdout_len": len(result),
            "stderr_len": len(stderr),
            "duration_ms": duration_ms,
        }
        
        # Check for error messages from generated script (script shouldn't generate these)
        result_lower = result.lower()
        stderr_lower = stderr.lower()
        error_indicators = [
            "query cannot be completed",
            "no matching signals",
            "cannot find signals",
            "signals not found",
            "no signals found",
            "missing signals",
            "could not find signals",
            "voltage data was not available",
            "temperature data was not available",
            "data was not available to check",
            "not available to check"
        ]
        if any(indicator in result_lower for indicator in error_indicators) or any(indicator in stderr_lower for indicator in error_indicators):
            logger.error(f"[SCRIPT ERROR] Generated script returned error message instead of using provided signals.")
            logger.error(f"[SCRIPT ERROR] Result: {result[:500]}")
            logger.error(f"[SCRIPT ERROR] Stderr: {stderr[:500]}")
            logger.error(f"[SCRIPT ERROR] This suggests the AI script ignored the instruction to use provided signals.")
            # Check specifically for voltage/temperature "not available" errors
            if "voltage" in result_lower and "not available" in result_lower:
                logger.error(f"[SCRIPT ERROR] Script incorrectly reported voltage data as not available. Voltage signals were provided and should exist in the DataFrame.")
            if "temperature" in result_lower and "not available" in result_lower:
                logger.error(f"[SCRIPT ERROR] Script incorrectly reported temperature data as not available. Temperature signals were provided and should exist in the DataFrame.")
        
        # Check if table_data exists
        table_data = captured.get("__table_data")
        
        # Check for empty data indicators in result
        empty_indicators = [
            "no data found",
            "no data available",
            "no instances found",
            "empty",
            "no results",
            "no matching",
            "could not find",
            "did not find",
            "voltage data was not available",
            "temperature data was not available",
            "data was not available",
            "not available to check"
        ]
        has_empty_indicator = any(indicator in result_lower for indicator in empty_indicators)
        
        # If script says "no data" but table_data exists with rows, log a warning
        if has_empty_indicator and table_data and isinstance(table_data, dict):
            rows = table_data.get("rows", [])
            if rows and len(rows) > 0:
                logger.warning(f"[DATA MISMATCH] Script says 'no data' but table_data has {len(rows)} rows. This is incorrect - data exists and should be shown.")
                logger.warning(f"[DATA MISMATCH] Result message: {result[:200]}")
                logger.warning(f"[DATA MISMATCH] Table columns: {table_data.get('columns', [])}")
        
        if has_empty_indicator:
            logger.warning(f"[EMPTY RESULT] Script returned empty data indicator. Result: {result[:200]}, stderr: {stderr[:200]}")
        
        # Check if table_data is empty
        if table_data and isinstance(table_data, dict):
            rows = table_data.get("rows", [])
            if not rows or len(rows) == 0:
                logger.warning(f"[EMPTY TABLE] Table data has no rows. Columns: {table_data.get('columns', [])}")
        
        if result:
            logger.info(f"[SCRIPT RESULT] Returning result: {result[:200]}...")
            logger.info(f"[SCRIPT RESULT] Has image: {captured.get('__image_base64') is not None}, Has table_data: {table_data is not None}")
            if table_data:
                rows_count = len(table_data.get("rows", [])) if isinstance(table_data, dict) else 0
                cols_count = len(table_data.get("columns", [])) if isinstance(table_data, dict) else 0
                logger.info(f"[SCRIPT RESULT] Table data: {rows_count} rows, {cols_count} columns")
            return result, debug, captured.get("__image_base64"), table_data

        # Fallback: try to read common result variables if nothing was printed
        for var_name in ("result", "answer", "output"):
            if var_name in safe_globals and safe_globals[var_name] is not None:
                try:
                    debug["fallback_var"] = var_name
                    return str(safe_globals[var_name]), debug, captured.get("__image_base64"), captured.get("__table_data")
                except Exception:
                    continue
        # Fallback 2: captured result via set_result
        if captured.get("__captured_result") is not None:
            debug["fallback_var"] = "__captured_result"
            result_str = str(captured["__captured_result"])
            result_lower = result_str.lower()
            empty_indicators = [
                "no data found",
                "no data available",
                "no instances found",
                "empty",
                "no results"
            ]
            if any(indicator in result_lower for indicator in empty_indicators):
                logger.warning(f"[EMPTY RESULT] Captured result indicates empty data: {result_str[:200]}")
            return result_str, debug, captured.get("__image_base64"), captured.get("__table_data")

        debug["reason"] = "no stdout and no fallback variable"
        logger.warning(f"[NO OUTPUT] Script generated no output. stderr: {stderr[:500]}")
        return "No output generated", debug, captured.get("__image_base64"), captured.get("__table_data")
        
    except SyntaxError as e:
        logger.error(f"Pandas script syntax error: {e}\nOriginal script:\n{script}")
        raise Exception(f"Script execution failed: {e}")
    except TimeoutError as e:
        logger.error(f"Pandas script execution timeout: {e}")
        raise Exception("Script execution timed out")
    except Exception as e:
        logger.error(f"Pandas script execution error: {e}")
        raise Exception(f"Script execution failed: {str(e)}")


async def process_query(
    message: str,
    gemini_call_fn,
    script_cache: Dict[str, Dict[str, Any]],
    script_cache_ttl: int = 3600,
) -> Tuple[str, Dict[str, Any], Optional[str], Optional[Dict[str, Any]], List[Tuple[str, float]], List[Tuple[str, float, Dict[str, Any]]]]:
    """Process a user query and return results with metadata."""
    from time import time
    
    # Extract trip_id specification (can be "all", "range:X-Y", or single trip ID)
    trip_spec = extract_trip_id(message)
    
    # Expand trip specification to list of trip IDs
    if trip_spec:
        trip_ids = expand_trip_ids(trip_spec)
        logger.info(f"[TRIP EXPANSION] Query '{message[:60]}...' -> trip_spec='{trip_spec}' -> trip_ids={trip_ids}")
    else:
        trip_ids = [TRIP_ID]  # Default to single trip
    
    # For signal discovery, aggregate signals from all trips if multiple trips requested
    # This ensures we have a comprehensive signal list
    if len(trip_ids) > 1:
        # Aggregate signals from all trips
        all_signals = set()
        for trip_id in trip_ids:
            trip_signals = get_known_signals(trip_id)
            all_signals.update(trip_signals)
        pool = sorted(list(all_signals))
        logger.info(f"[SIGNAL POOL] Aggregated {len(pool)} unique signals from {len(trip_ids)} trips")
        chosen_trip = trip_ids[0]  # Use first trip for logging purposes
    else:
        # Use first trip for signal discovery (signals should be similar across trips)
        chosen_trip = trip_ids[0] if trip_ids else TRIP_ID
        pool = None  # Will be fetched in the cache check below
    
    # Cache by normalized query to reduce model usage
    normalized = message.strip().lower()
    cached = script_cache.get(normalized)
    now = time()
    
    if cached and (now - cached["ts"]) < script_cache_ttl:
        pandas_script = cached["script"]
        suggested_pairs = cached.get("suggested_pairs", [])
        top_explained = cached.get("top_explained", [])
    else:
        # Select suggested signals for the query
        # If pool wasn't set above (single trip case), fetch it now
        if pool is None:
            pool = get_known_signals(chosen_trip)
        # Try exact inference for cell+metric first
        exact = infer_specific_signals(message, pool)
        suggested_pairs: List[Tuple[str, float]] = []
        if exact:
            # Score the exact signals to preserve scoring information
            suggested_pairs = [(sig, score_signal(message, sig)) for sig in exact]
            suggested_pairs.sort(key=lambda x: x[1])  # Sort by score (lower is better)
            suggested_signals = exact
            logger.info(f"[SIGNAL SELECTION] Using exact inference: {suggested_signals} with scores: {[(sig, f'{score:.2f}') for sig, score in suggested_pairs]}")
        else:
            # metric filtering - detect multiple metrics (e.g., "temperature AND current")
            ml = message.lower()
            wants_temp = bool(re.search(r"\btemp(?:erature)?s?\b", ml))
            wants_volt = bool(re.search(r"\bvolt(?:age|ages)?\b", ml))  # Matches "volt", "voltage", "voltages"
            wants_current = bool(re.search(r"\bcurrent\b|motor\s+current|current\s+draw", ml))
            
            # Log detected metrics for debugging
            logger.info(f"[SIGNAL SELECTION] Detected metrics - temp: {wants_temp}, volt: {wants_volt}, current: {wants_current}")
            
            # Detect qualifiers for more specific matching
            wants_battery = bool(re.search(r"\bbattery\b", ml))
            wants_cell = bool(re.search(r"\bcell\b", ml))  # "cell voltages" = all ACU cell voltages (like battery)
            wants_pack = bool(re.search(r"\bpack\b", ml))
            wants_motor = bool(re.search(r"\bmotor\b", ml))
            wants_inverter = bool(re.search(r"\binverter\b", ml))
            logger.info(f"[SIGNAL SELECTION] Detected qualifiers - battery: {wants_battery}, cell: {wants_cell}, pack: {wants_pack}, motor: {wants_motor}")
            
            # Extract ALL battery numbers mentioned (e.g., "battery 1", "battery 22", "battery 18")
            # Battery X = ACU cell X, so we need to find all battery numbers
            battery_number_matches = re.findall(r"\bbattery\s+(\d{1,3})\b", ml)
            battery_numbers = list(set(battery_number_matches)) if battery_number_matches else []  # Remove duplicates
            has_battery_number = len(battery_numbers) > 0
            
            # Check if cell number is mentioned - if not, prefer general signals over cell-specific
            cell_number_matches = re.findall(r"\bcell\s+(\d{1,3})\b", ml)
            cell_numbers = list(set(cell_number_matches)) if cell_number_matches else []
            has_cell_number = len(cell_numbers) > 0
            
            # If battery numbers are mentioned, treat them as cell numbers
            if has_battery_number:
                has_cell_number = True
                # Combine battery numbers and cell numbers
                all_cell_numbers = list(set(battery_numbers + cell_numbers))
            else:
                all_cell_numbers = cell_numbers
            
            # Count how many different metrics are requested
            metric_count = sum([wants_temp, wants_volt, wants_current])
            # Also check for "AND" or "&" which indicates multiple conditions
            has_and = bool(re.search(r"\band\b|&", ml))
            
            # Determine how many signals we need
            if metric_count > 1 or has_and:
                # Multiple metrics requested - need multiple signals
                num_signals_needed = min(metric_count + (1 if has_and else 0), 6)
            else:
                num_signals_needed = 1
            
            # special: correlation or vs chooses top 2
            wants_two = bool(re.search(r"\bvs\b|correl", ml))
            if wants_two:
                num_signals_needed = 2
            
            # Get signals for each metric type with qualifier awareness
            # Collect (signal, score) pairs to preserve scoring information
            suggested_pairs: List[Tuple[str, float]] = []
            suggested_signal_names = set()  # Track signal names to avoid duplicates
            
            if wants_temp:
                # Build temperature predicate based on qualifiers
                def temp_pred(s):
                    if not isinstance(s, str) or not s.endswith("_temp"):
                        return False
                    
                    # If specific battery numbers mentioned (e.g., "battery 22", "battery 18"), select those ACU cells
                    if has_battery_number:
                        # Battery X = ACU cell X, so check if signal matches any of the mentioned battery numbers
                        for bat_num in battery_numbers:
                            if f"acu_cell{bat_num}_temp" == s.lower():
                                return True
                        return False  # Signal doesn't match any mentioned battery number
                    
                    # If battery OR cell mentioned WITHOUT number: battery/cell = ALL ACU cells
                    if (wants_battery or wants_cell) and not has_battery_number and not has_cell_number:
                        # Battery/cell temperature means any ACU cell temperature
                        return "acu_cell" in s.lower() and re.search(r"cell\d+", s.lower())
                    
                    # If specific cell numbers mentioned (not battery), select those cells
                    if all_cell_numbers:
                        for cell_num in all_cell_numbers:
                            if f"acu_cell{cell_num}_temp" == s.lower():
                                return True
                        return False
                    
                    # If pack mentioned, prioritize pack signals
                    if wants_pack:
                        return "pack" in s.lower()
                    
                    # Exclude cell-specific unless cell number mentioned (but battery cases handled above)
                    if not has_cell_number and re.search(r"cell\d+", s.lower()):
                        return False
                    
                    # If motor mentioned, prioritize motor temp (but not if battery was mentioned)
                    if wants_motor and not (wants_battery or wants_pack):
                        return "motor" in s.lower()
                    
                    # Otherwise, prefer general temp signals (not cell-specific, not motor-specific unless motor mentioned)
                    if wants_motor:
                        return "motor" in s.lower()
                    
                    # Prefer battery/pack over motor if neither qualifier specified
                    return "battery" in s.lower() or "pack" in s.lower() or (not "motor" in s.lower())
                
                # For battery/cell queries without number, get ALL ACU cell temp signals since battery/cell = all ACU cells
                if (wants_battery or wants_cell) and not has_battery_number and not has_cell_number:
                    # Get all ACU cell temp signals (up to reasonable limit)
                    max_temp_signals = 50  # Get many signals to capture all ACU cells
                    temp_pairs = best_signals_for_query(message, max_signals=max_temp_signals, threshold=100.0, filter_pred=temp_pred, pool=pool, default_trip_id=TRIP_ID)
                    if temp_pairs:
                        # Add ALL ACU cell temp signals, not just the first one
                        for temp_pair in temp_pairs:
                            if temp_pair[0] not in suggested_signal_names:
                                suggested_pairs.append(temp_pair)
                                suggested_signal_names.add(temp_pair[0])
                        logger.info(f"[SIGNAL SELECTION] Battery temperature (all): selected {len(temp_pairs)} ACU cell temp signals (battery = all ACU cells)")
                else:
                    # For specific battery numbers or cell numbers, get signals for each mentioned number
                    if has_battery_number or all_cell_numbers:
                        # Get signals for each mentioned battery/cell number
                        target_numbers = battery_numbers if has_battery_number else all_cell_numbers
                        for num in target_numbers:
                            # Find signal matching this specific cell number
                            target_signal = f"acu_cell{num}_temp"
                            # Score this specific signal
                            if target_signal in pool:
                                score = score_signal(message, target_signal)
                                if target_signal not in suggested_signal_names:
                                    suggested_pairs.append((target_signal, score))
                                    suggested_signal_names.add(target_signal)
                        if has_battery_number:
                            logger.info(f"[SIGNAL SELECTION] Battery numbers {battery_numbers}: selected ACU cell temp signals for batteries {battery_numbers}")
                        else:
                            logger.info(f"[SIGNAL SELECTION] Cell numbers {all_cell_numbers}: selected ACU cell temp signals")
                    else:
                        # For non-battery queries, get the best match
                        temp_pairs = best_signals_for_query(message, max_signals=5, threshold=100.0, filter_pred=temp_pred, pool=pool, default_trip_id=TRIP_ID)
                        if temp_pairs:
                            # Use the best scoring signal (first in sorted list)
                            best_temp = temp_pairs[0]
                            suggested_pairs.append(best_temp)
                            suggested_signal_names.add(best_temp[0])
                            logger.info(f"[SIGNAL SELECTION] Temperature signal selected: {best_temp[0]} (score: {best_temp[1]:.2f}, from {len(temp_pairs)} candidates)")
            
            if wants_current:
                # Build current predicate based on qualifiers
                def current_pred(s):
                    if not isinstance(s, str):
                        return False
                    s_lower = s.lower()
                    # Exclude cell-specific
                    if re.search(r"cell\d+", s_lower):
                        return False
                    # Exclude error flags
                    if "error" in s_lower or "flag" in s_lower:
                        return False
                    # PRIORITY 1: tcm_power_draw is motor current draw (highest priority)
                    if "tcm_power_draw" in s_lower:
                        return True
                    # PRIORITY 2: Must have current, draw, or power_draw
                    if not ("current" in s_lower or "draw" in s_lower or "power_draw" in s_lower):
                        return False
                    # PRIORITY 3: If motor mentioned, prioritize motor current
                    if wants_motor:
                        return "motor" in s_lower or "inverter" in s_lower or "tcm" in s_lower
                    # Otherwise, prefer motor/inverter/TCM current over other types
                    return "motor" in s_lower or "inverter" in s_lower or "tcm" in s_lower
                
                current_pairs = best_signals_for_query(message, max_signals=5, threshold=100.0, filter_pred=current_pred, pool=pool, default_trip_id=TRIP_ID)
                if current_pairs:
                    # Use the best scoring signal (first in sorted list)
                    best_current = current_pairs[0]
                    if best_current[0] not in suggested_signal_names:
                        suggested_pairs.append(best_current)
                        suggested_signal_names.add(best_current[0])
                        logger.info(f"[SIGNAL SELECTION] Current signal selected: {best_current[0]} (score: {best_current[1]:.2f}, from {len(current_pairs)} candidates)")
            
            if wants_volt:
                logger.info(f"[SIGNAL SELECTION] Processing voltage signals - wants_battery: {wants_battery}, wants_cell: {wants_cell}, has_battery_number: {has_battery_number}, has_cell_number: {has_cell_number}")
                logger.info(f"[SIGNAL SELECTION] Pool size for voltage selection: {len(pool) if pool else 'None'}")
                # Build voltage predicate based on qualifiers
                # DEFAULT: All voltages are ACU cell voltages (like temperatures)
                def volt_pred(s):
                    if not isinstance(s, str) or not s.endswith("_voltage"):
                        return False
                    
                    # CRITICAL: By default, ALL voltages are ACU cell voltages (like temperatures)
                    # Only exclude if pack is explicitly mentioned (pack voltage != cell voltage)
                    
                    # If specific battery numbers mentioned (e.g., "battery 22", "battery 18"), select those ACU cells
                    if has_battery_number:
                        # Battery X = ACU cell X, so check if signal matches any of the mentioned battery numbers
                        for bat_num in battery_numbers:
                            if f"acu_cell{bat_num}_voltage" == s.lower():
                                return True
                        return False  # Signal doesn't match any mentioned battery number
                    
                    # If specific cell numbers mentioned (e.g., "cell 1 voltage"), select those cells
                    if all_cell_numbers:
                        for cell_num in all_cell_numbers:
                            if f"acu_cell{cell_num}_voltage" == s.lower():
                                return True
                        return False
                    
                    # If pack mentioned, prioritize pack signals (pack voltage is different from cell voltage)
                    if wants_pack:
                        return "pack" in s.lower()
                    
                    # DEFAULT: All voltage queries (without pack) = ACU cell voltages
                    # This matches how temperature works: "battery temperature" = all ACU cell temps
                    # "cell voltages" = all ACU cell voltages
                    # "voltage" (general) = all ACU cell voltages
                    return "acu_cell" in s.lower() and re.search(r"cell\d+", s.lower())
                
                # DEFAULT: All voltage queries = ACU cell voltages (like temperatures)
                # Only exception: if pack is explicitly mentioned (pack voltage != cell voltage)
                # If specific cell/battery numbers mentioned, select those specific cells
                if has_battery_number or all_cell_numbers:
                    # For specific battery/cell numbers, get signals for each mentioned number
                    target_numbers = battery_numbers if has_battery_number else all_cell_numbers
                    for num in target_numbers:
                        # Find signal matching this specific cell number
                        target_signal = f"acu_cell{num}_voltage"
                        # Score this specific signal
                        if target_signal in pool:
                            score = score_signal(message, target_signal)
                            if target_signal not in suggested_signal_names:
                                suggested_pairs.append((target_signal, score))
                                suggested_signal_names.add(target_signal)
                    if has_battery_number:
                        logger.info(f"[SIGNAL SELECTION] Battery numbers {battery_numbers}: selected ACU cell voltage signals for batteries {battery_numbers}")
                    else:
                        logger.info(f"[SIGNAL SELECTION] Cell numbers {all_cell_numbers}: selected ACU cell voltage signals")
                elif wants_pack:
                    # Pack voltage is different from cell voltage - get pack signals
                    volt_pairs = best_signals_for_query(message, max_signals=5, threshold=100.0, filter_pred=volt_pred, pool=pool, default_trip_id=TRIP_ID)
                    if volt_pairs:
                        best_volt = volt_pairs[0]
                        if best_volt[0] not in suggested_signal_names:
                            suggested_pairs.append(best_volt)
                            suggested_signal_names.add(best_volt[0])
                            logger.info(f"[SIGNAL SELECTION] Pack voltage signal selected: {best_volt[0]} (score: {best_volt[1]:.2f})")
                else:
                    # DEFAULT: All voltage queries = ALL ACU cell voltage signals (like temperatures)
                    # Get all ACU cell voltage signals (up to reasonable limit)
                    max_volt_signals = 50  # Get many signals to capture all ACU cells
                    logger.info(f"[SIGNAL SELECTION] Voltage query (default): selecting ALL ACU cell voltage signals (voltage = ACU cell voltage, like temperature)")
                    logger.info(f"[SIGNAL SELECTION] Calling best_signals_for_query for voltage with max_signals={max_volt_signals}, threshold=100.0")
                    # Test the predicate on a few signals to debug
                    if pool:
                        test_signals = [s for s in pool if s.endswith('_voltage') and 'acu_cell' in s][:5]
                        logger.info(f"[SIGNAL SELECTION] Testing voltage predicate on sample ACU cell signals: {test_signals}")
                        for test_sig in test_signals:
                            pred_result = volt_pred(test_sig)
                            logger.info(f"[SIGNAL SELECTION] volt_pred('{test_sig}') = {pred_result}")
                    volt_pairs = best_signals_for_query(message, max_signals=max_volt_signals, threshold=100.0, filter_pred=volt_pred, pool=pool, default_trip_id=TRIP_ID)
                    logger.info(f"[SIGNAL SELECTION] best_signals_for_query returned {len(volt_pairs)} voltage signal pairs")
                    if volt_pairs:
                        # Add ALL ACU cell voltage signals, not just the first one
                        added_count = 0
                        for volt_pair in volt_pairs:
                            if volt_pair[0] not in suggested_signal_names:
                                suggested_pairs.append(volt_pair)
                                suggested_signal_names.add(volt_pair[0])
                                added_count += 1
                        logger.info(f"[SIGNAL SELECTION] Voltage (all ACU cells): selected {added_count} ACU cell voltage signals")
                        logger.info(f"[SIGNAL SELECTION] Voltage signals: {[pair[0] for pair in volt_pairs[:10]]}{'...' if len(volt_pairs) > 10 else ''}")
                    else:
                        logger.warning(f"[SIGNAL SELECTION] No voltage signals found! This is unexpected. Pool has {len(pool) if pool else 0} signals.")
                        # Debug: check what voltage signals exist in pool
                        if pool:
                            volt_in_pool = [s for s in pool if s.endswith('_voltage')]
                            acu_volt_in_pool = [s for s in pool if s.endswith('_voltage') and 'acu_cell' in s]
                            logger.warning(f"[SIGNAL SELECTION] Total voltage signals in pool: {len(volt_in_pool)}")
                            logger.warning(f"[SIGNAL SELECTION] ACU cell voltage signals in pool: {len(acu_volt_in_pool)}")
                            logger.warning(f"[SIGNAL SELECTION] Sample ACU voltage signals: {acu_volt_in_pool[:10]}{'...' if len(acu_volt_in_pool) > 10 else ''}")
            
            # Log final signal selection summary
            logger.info(f"[SIGNAL SELECTION] Final signal count: {len(suggested_pairs)} signals selected")
            logger.info(f"[SIGNAL SELECTION] Selected signals: {[pair[0] for pair in suggested_pairs[:20]]}{'...' if len(suggested_pairs) > 20 else ''}")
            
            # If we still don't have enough signals, get more general matches
            if len(suggested_pairs) < num_signals_needed:
                # Fallback: get top signals without specific filters (using scoring)
                pairs = best_signals_for_query(message, max_signals=num_signals_needed * 2, threshold=100.0, pool=pool, default_trip_id=TRIP_ID)
                for pair in pairs:
                    if pair[0] not in suggested_signal_names:
                        suggested_pairs.append(pair)
                        suggested_signal_names.add(pair[0])
                        if len(suggested_pairs) >= num_signals_needed:
                            break
            
            # Sort by score (lower is better) to ensure best matches are first
            suggested_pairs.sort(key=lambda x: x[1])
            
            # Extract signal names for logging and script generation
            suggested_signals = [sig for sig, _ in suggested_pairs]
            logger.info(f"[SIGNAL SELECTION] Selected {len(suggested_signals)} signals with scores: {[(sig, f'{score:.2f}') for sig, score in suggested_pairs]}")
        
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
            logger.error(f"[SIGNAL SELECTION] No signals selected for query: '{message[:100]}...'")
            raise ValueError("No matching signals found for query")
        
        # Log signal mappings for debugging
        signal_mappings = []
        if any("tcm_power_draw" in sig for sig in suggested_signals):
            signal_mappings.append("tcm_power_draw → motor current draw")
        if any("acu_cell" in sig and "_temp" in sig for sig in suggested_signals):
            signal_mappings.append("acu_cell*_temp → battery temperature")
        if signal_mappings:
            logger.info(f"[SIGNAL MAPPINGS] {', '.join(signal_mappings)}")
        
        logger.info(f"[SCRIPT GENERATION] Generating script with {len(suggested_signals)} signals: {suggested_signals[:10]}{'...' if len(suggested_signals) > 10 else ''}, for {len(trip_ids)} trip(s): {trip_ids}")
        pandas_script = await generate_pandas_script(message, suggested_signals, gemini_call_fn, trip_ids=trip_ids if len(trip_ids) > 1 else None)
        script_cache[normalized] = {
            "script": pandas_script,
            "ts": now,
            "suggested_pairs": suggested_pairs,
            "top_explained": top_explained,
        }
    
    # Execute the script
    # For single trip, pass trip_id_override. For multiple trips, the script handles it internally
    trip_id_override = trip_ids[0] if len(trip_ids) == 1 else None
    result, debug_info, image_b64, table_data = await execute_pandas_script(pandas_script, trip_id_override=trip_id_override)
    
    # Log if result indicates no data found
    result_lower = result.lower() if result else ""
    if any(phrase in result_lower for phrase in ["no data found", "no instances found", "no matching", "could not find"]):
        logger.warning(
            f"[QUERY RESULT] No data found for query: '{message[:100]}...' "
            f"Signals: {suggested_signals}, Trip IDs: {trip_ids}, Result: {result[:200]}"
        )
    
    # Log if table_data is empty
    if table_data and isinstance(table_data, dict):
        rows = table_data.get("rows", [])
        if not rows or len(rows) == 0:
            logger.warning(
                f"[QUERY RESULT] Empty table data for query: '{message[:100]}...' "
                f"Signals: {suggested_signals}, Trip IDs: {trip_ids}, Columns: {table_data.get('columns', [])}"
            )
    
    return result, debug_info, image_b64, table_data, suggested_pairs, top_explained

