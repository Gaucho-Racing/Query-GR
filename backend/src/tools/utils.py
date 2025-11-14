"""Utility functions for signal matching, scoring, and database operations."""
import re
import logging
import mysql.connector
import os
from typing import List, Tuple, Optional, Dict, Any, Callable
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# DB configuration for signals list
DB_HOST = os.getenv("DATABASE_HOST", "")
DB_PORT = int(os.getenv("DATABASE_PORT", "3306"))
DB_USER = os.getenv("DATABASE_USER", "")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD", "")
DB_NAME = os.getenv("DATABASE_NAME", "")

DB_SIGNALS_CACHE: Dict[str, List[str]] = {}  # Keyed by trip_id


def extract_trip_id(message: str) -> Optional[str]:
    """Extract trip ID from user message using various patterns.
    Returns:
    - Single trip ID as string (e.g., "3")
    - "all" if user wants all trips
    - "range:X-Y" for trip ranges (e.g., "range:3-5")
    - None if no trip specified
    """
    ml = message.lower()
    
    # Check for "all" keywords - improved to catch "all test runs", "show all runs", etc.
    if (re.search(r"\b(all|every|each)\s+(?:trip|run|test|runs|trips)", ml) or 
        re.search(r"(?:trip|run|test|runs|trips)\s+(?:all|every|each)", ml) or
        re.search(r"\bshow\s+all\s+(?:trip|run|test|runs|trips)", ml) or
        re.search(r"\ball\s+(?:test\s+)?(?:runs?|trips?)", ml)):
        return "all"
    
    # Check for ranges like "trips 3-4" or "runs 3 to 5"
    range_match = re.search(r"(?:trip|run)s?\s+(\d{1,3})\s*[-–—to]\s*(\d{1,3})", ml)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        return f"range:{start}-{end}"
    
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
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6,
        'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
        'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14,
        'fifteenth': 15, 'sixteenth': 16, 'seventeenth': 17,
        'eighteenth': 18, 'nineteenth': 19, 'twentieth': 20
    }
    m = re.search(r"\b(" + "|".join(ord_map.keys()) + r")\s+(?:run|trip)\b", ml)
    if m:
        return str(ord_map[m.group(1)])
    return None


def get_all_trip_ids() -> List[str]:
    """Fetch all distinct trip IDs from the database."""
    if not (DB_HOST and DB_USER and DB_NAME):
        logger.warning("[SQL QUERY] Cannot fetch trip IDs: DB credentials not configured")
        return []
    try:
        logger.info(f"[SQL QUERY] Connecting to MySQL {DB_HOST}:{DB_PORT}/{DB_NAME} to fetch trip IDs")
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
            # Try trip table first (most common schema), then fall back to signal table
            # Order: trip table -> signal table -> discover tables
            alternatives = [
                ("SELECT DISTINCT id FROM `trip` ORDER BY CAST(id AS UNSIGNED)", "trip table"),
                ("SELECT DISTINCT trip_id FROM `trip` ORDER BY CAST(trip_id AS UNSIGNED)", "trip table with trip_id"),
                ("SELECT DISTINCT trip_id FROM `signal` WHERE trip_id IS NOT NULL ORDER BY CAST(trip_id AS UNSIGNED)", "signal table"),
                ("SHOW TABLES LIKE 'trip%'", "check trip tables"),
            ]
            for query, desc in alternatives:
                try:
                    if "SHOW TABLES" in query:
                        # First check what tables exist
                        cursor.execute(query)
                        tables = cursor.fetchall()
                        logger.info(f"[SQL QUERY] Found trip-related tables: {tables}")
                        if tables:
                            # Try to query the first trip table
                            table_name = tables[0][0]
                            try:
                                cursor.execute(f"SELECT DISTINCT id FROM `{table_name}` ORDER BY CAST(id AS UNSIGNED)")
                                rows = cursor.fetchall()
                                trip_ids = [str(r[0]) for r in rows if r and r[0] is not None]
                                if trip_ids:
                                    logger.info(f"[SQL QUERY] Found {len(trip_ids)} trip IDs from {table_name}")
                                    return trip_ids
                            except Exception as e3:
                                logger.debug(f"[SQL QUERY] Failed to query {table_name}: {e3}")
                                pass
                    else:
                        logger.debug(f"[SQL QUERY] Trying: {desc}")
                        cursor.execute(query)
                        rows = cursor.fetchall()
                        trip_ids = [str(r[0]) for r in rows if r and r[0] is not None]
                        if trip_ids:
                            logger.info(f"[SQL QUERY] Found {len(trip_ids)} trip IDs from {desc}: {trip_ids[:10]}..." if len(trip_ids) > 10 else f"[SQL QUERY] Found {len(trip_ids)} trip IDs from {desc}: {trip_ids}")
                            return trip_ids
                except Exception as e2:
                    logger.debug(f"[SQL QUERY] Query failed ({desc}): {e2}")
                    continue
            logger.warning(f"[SQL QUERY] All queries failed. Cannot fetch trip IDs.")
            return []
        finally:
            try:
                cursor.close()
            except Exception:
                pass
            conn.close()
    except Exception as e:
        logger.warning(f"[SQL QUERY] Failed to fetch trip IDs: {e}")
        return []


def expand_trip_ids(trip_spec: str) -> List[str]:
    """Expand trip specification into list of trip IDs.
    
    Args:
        trip_spec: Can be:
            - "all" -> returns all trip IDs from DB
            - "range:X-Y" -> returns list of trip IDs from X to Y inclusive
            - Single trip ID -> returns [trip_id]
    
    Returns:
        List of trip ID strings
    """
    if trip_spec == "all":
        return get_all_trip_ids()
    elif trip_spec.startswith("range:"):
        # Parse range like "range:3-5"
        match = re.match(r"range:(\d+)-(\d+)", trip_spec)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            return [str(i) for i in range(start, end + 1)]
        return []
    else:
        # Single trip ID
        return [trip_spec]


def refresh_db_signals_cache(trip_id: str) -> int:
    """Fetch distinct signal names into DB_SIGNALS_CACHE for a specific trip_id. Returns count or 0 on failure."""
    global DB_SIGNALS_CACHE
    logger.info(f"[CACHE MISS] Running SQL query for trip_id={trip_id} (not found in cache)")
    if not (DB_HOST and DB_USER and DB_NAME):
        logger.warning(f"[CACHE MISS] Cannot run SQL query for trip_id={trip_id}: DB credentials not configured")
        return 0
    try:
        logger.info(f"[SQL QUERY] Connecting to MySQL {DB_HOST}:{DB_PORT}/{DB_NAME} for trip_id={trip_id}")
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
            query = "SELECT DISTINCT name FROM `signal` LIMIT 9999;"
            logger.info(f"[SQL QUERY] Executing: {query} for trip_id={trip_id}")
            cursor.execute(query)
            rows = cursor.fetchall()
            names = [r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip()]
            signals = sorted(set(names))
            DB_SIGNALS_CACHE[trip_id] = signals
            logger.info(f"[SQL QUERY] Successfully loaded {len(signals)} signals from DB {DB_NAME} for trip_id {trip_id} and cached")
            return len(signals)
        finally:
            try:
                cursor.close()
            except Exception:
                pass
            conn.close()
    except Exception as e:
        logger.warning(f"[SQL QUERY] Failed to load signals from DB for trip_id {trip_id}: {e}")
        return 0


def get_known_signals(trip_id: str) -> List[str]:
    """Get cached signals for trip_id, or fetch from DB if not cached."""
    if trip_id not in DB_SIGNALS_CACHE:
        logger.info(f"[CACHE] trip_id={trip_id} not in cache, fetching from DB...")
        refresh_db_signals_cache(trip_id)
    else:
        cached_count = len(DB_SIGNALS_CACHE.get(trip_id, []))
        logger.info(f"[CACHE HIT] trip_id={trip_id} found in cache with {cached_count} signals (no SQL query needed)")
    signals = DB_SIGNALS_CACHE.get(trip_id, [])
    logger.info(f"[CACHE] Returning {len(signals)} signals for trip_id={trip_id}")
    return signals


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

    # Check if battery number is mentioned (e.g., "battery 1" = ACU cell 1)
    # Initialize early so it can be used throughout the function
    battery_number_match = re.search(r"\bbattery\s+(\d{1,3})\b", q)
    battery_number = battery_number_match.group(1) if battery_number_match else None

    # Keyword presence bonuses (lower score)
    keywords = [
        ("voltage", 40.0), ("volt", 35.0), ("temp", 35.0), ("temperature", 40.0),
        ("speed", 30.0), ("current", 30.0), ("cell", 20.0), ("acu", 10.0),
        ("inverter", 10.0), ("magnetometer", 10.0), ("motor", 30.0), ("battery", 30.0),
        ("pack", 25.0), ("draw", 20.0), ("power_draw", 25.0), ("tcm", 20.0)
    ]
    
    # Special case: "motor current draw" or "current draw" matches "tcm_power_draw"
    # TCM power draw is motor current draw - give VERY strong bonus
    if ("motor" in q or "current" in q) and "draw" in q and "tcm_power_draw" in s:
        score -= 60.0  # Very strong bonus for motor current draw -> tcm_power_draw match
    
    # Penalize tcm_current_draw when motor current draw is mentioned (tcm_power_draw is preferred)
    if ("motor" in q or "current" in q) and "draw" in q and "tcm_current_draw" in s and "tcm_power_draw" not in s:
        score += 30.0  # Penalty to prefer tcm_power_draw over tcm_current_draw
    for kw, bonus in keywords:
        if kw in q and kw in s:
            score -= bonus
    
    # Check if query mentions a specific cell number (must be before checking general "cell")
    cell_mention = re.search(r"\bcell\s*(\d{1,3})\b", q)
    
    # Special case: "battery" or "cell" in query matches "acu_cell" in signal name
    # Battery/cell refers to ACU cells, so battery/cell temperature/voltage = ACU cell temperatures/voltages
    
    # Check for "cell" mentioned without a number (e.g., "cell voltage", "cell temperature")
    cell_mention_general = re.search(r"\bcell\s+(?:volt|temp|current)", q) or re.search(r"\b(?:volt|temp|current).*cell\b", q)
    has_cell_general = cell_mention_general is not None and cell_mention is None  # "cell" mentioned but no specific cell number
    
    if ("battery" in q or has_cell_general) and "acu_cell" in s:
        if battery_number:
            # Battery X should match ACU cell X specifically
            if battery_number in s:
                score -= 50.0  # Very strong bonus for exact battery number match
            else:
                score += 30.0  # Penalty if battery number doesn't match
        else:
            # Battery/cell (no number) matches any ACU cell
            score -= 35.0  # Strong bonus for battery/cell -> acu_cell match
    
    # Also check for battery number (battery X = ACU cell X)
    # Note: battery_number is already defined above if "battery" is in query
    has_cell_number = cell_mention is not None or battery_number is not None
    
    # Numeric cues - only match if:
    # 1. Cell number is mentioned AND number matches cell number in signal
    # 2. OR number appears in context that suggests it's NOT a threshold (e.g., "cell 45")
    # Otherwise, don't match numbers - they're likely thresholds (45°C, 300A, etc.)
    nums = re.findall(r"\d+", q)
    for n in nums:
        # Only apply numeric bonus if:
        # - Cell number is explicitly mentioned AND matches
        # - OR number appears right after "cell" keyword
        if has_cell_number and cell_mention.group(1) == n:
            if n in s:
                score -= 30.0
        elif re.search(rf"\bcell\s*{n}\b", q):
            # Number appears in "cell X" context
            if n in s:
                score -= 30.0
        # Otherwise, don't match numbers - they're thresholds, not cell identifiers
    
    # Penalize cell-specific signals when no cell number is mentioned
    # EXCEPTION: If "battery" or "cell" (without number) is mentioned, ACU cell signals are acceptable (battery/cell = all ACU cells)
    # EXCEPTION: If "battery X" is mentioned, ACU cell X is acceptable (battery X = ACU cell X)
    # This helps prefer general temperature/voltage/current signals
    if not has_cell_number and re.search(r"cell\d+", s):
        # Query doesn't mention a cell, but signal is cell-specific
        # Only penalize if query is about general temperature/voltage/current
        # BUT: Don't penalize if "battery" or "cell" (without number) is mentioned (battery/cell = all ACU cells)
        if re.search(r"\b(temp|temperature|volt|voltage|current)\b", q):
            if "battery" not in q and not has_cell_general:
                score += 50.0  # Strong penalty for cell-specific when general is wanted (unless battery/cell mentioned)
            elif battery_number:
                # Battery X mentioned: Check if signal matches that specific cell
                if battery_number in s:
                    score -= 10.0  # Bonus for matching battery number
                else:
                    score += 20.0  # Penalty if battery number doesn't match
            else:
                # Battery/cell mentioned (no number): ACU cell signals are acceptable, but still slight penalty for specificity
                score += 10.0  # Small penalty to prefer general signals if they exist, but allow ACU cells

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
        ("inverter", 10.0), ("magnetometer", 10.0), ("motor", 30.0), ("battery", 30.0),
        ("pack", 25.0), ("draw", 20.0)
    ]
    for kw, bonus in keywords:
        if kw in q and kw in s:
            score -= bonus
            bonuses_hit.append(kw)

    # Check if query mentions a specific cell number
    cell_mention = re.search(r"\bcell\s*(\d{1,3})\b", q)
    has_cell_number = cell_mention is not None
    
    numeric_hits: List[str] = []
    nums = re.findall(r"\d+", q)
    for n in nums:
        if has_cell_number and cell_mention.group(1) == n:
            if n in s:
                score -= 30.0
                numeric_hits.append(n)
        elif re.search(rf"\bcell\s*{n}\b", q):
            if n in s:
                score -= 30.0
                numeric_hits.append(n)
    
    # Penalize cell-specific signals when no cell number is mentioned
    if not has_cell_number and re.search(r"cell\d+", s):
        if re.search(r"\b(temp|temperature|volt|voltage|current)\b", q):
            score += 50.0

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


def best_signals_for_query(
    query: str,
    max_signals: int = 6,
    threshold: float = 100.0,
    pool: Optional[List[str]] = None,
    filter_pred: Optional[Callable[[str], bool]] = None,
    trip_id: Optional[str] = None,
    default_trip_id: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Find best matching signals for a query."""
    if pool is not None:
        signals_source = pool
    elif trip_id is not None:
        signals_source = get_known_signals(trip_id)
    else:
        # Fallback to default trip_id if not provided
        signals_source = get_known_signals(default_trip_id or "4")
    if not signals_source:
        return []
    
    # Filter out metadata fields that are NOT actual sensor signals
    metadata_fields = [
        'trip_id', 'trip', 'tripid',
        'run_id', 'runid', 'run',
        'vehicle_id', 'vehicleid', 'vehicle',
        'produced_at', 'producedat', 'timestamp', 'time',
        'token'
    ]
    signals_source = [s for s in signals_source if s.lower() not in metadata_fields]
    
    if filter_pred:
        signals_source = [s for s in signals_source if filter_pred(s)]
    scored = [(sig, score_signal(query, sig)) for sig in signals_source]
    scored.sort(key=lambda x: x[1])
    return [(sig, sc) for sig, sc in scored[:max_signals] if sc <= threshold]


def infer_specific_signals(message: str, signals_pool: List[str]) -> List[str]:
    """Infer specific signals from message (e.g., cell temperature/voltage)."""
    ml = message.lower()
    # detect cell number
    m = re.search(r"\bcell\s*(\d{1,3})\b", ml)
    cell_num = m.group(1) if m else None
    wants_temp = bool(re.search(r"\btemp(?:erature)?s?\b", ml))
    wants_volt = bool(re.search(r"\bvolt(?:age|ages)?\b", ml))  # Matches "volt", "voltage", "voltages"
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


def is_vehicle_data_query(message: str, trip_id: Optional[str] = None, default_trip_id: Optional[str] = None) -> bool:
    """Determine via fuzzy matching against known signals and query intent terms."""
    # Quick intent words
    intents = ["average", "mean", "median", "min", "max", "top", "bottom", "compare", "correlation", "corr", "speed", "temperature", "voltage", "current"]
    ml = message.lower()
    intent_hit = any(w in ml for w in intents)
    # Fuzzy signal matches - only use cache if trip_id is provided (to avoid wrong cache lookups)
    # If trip_id is None, try to extract it from message, but don't default to TRIP_ID
    if trip_id is None:
        trip_id = extract_trip_id(message)  # Don't use default_trip_id here to avoid wrong cache
    # Only do signal matching if we have a trip_id (to use correct cache)
    # If no trip_id, rely on intent words only
    if trip_id is not None:
        matches = best_signals_for_query(message, max_signals=3, threshold=100.0, trip_id=trip_id, default_trip_id=default_trip_id)
        return intent_hit or len(matches) > 0
    else:
        # No trip_id available - just check intent words (don't use cache with wrong trip_id)
        return intent_hit

