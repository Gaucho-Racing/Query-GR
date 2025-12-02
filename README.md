# Vehicle Data Chatbot

A modern React + TypeScript chatbot that answers telemetry questions by generating and running Pandas scripts on live JSON data from vehicle sensors.

## Purpose

This chatbot provides an intelligent interface for querying vehicle telemetry data. Users can ask natural language questions about vehicle performance metrics, and the system will:

1. Classify whether the query is vehicle-data related
2. Fetch live data from the vehicle API
3. Generate custom Pandas scripts using AI
4. Execute the scripts to compute requested metrics
5. Return formatted results to the user (and plots when requested)

## Tech Stack

- **Frontend**: React + TypeScript + Tailwind CSS + Vite
- **Backend**: FastAPI + Pandas + HTTPX
- **AI Integration**: Gemini (intent detection + code generation)
- **Plotting**: Matplotlib (server-side image generation)
- **Data Source**: Live vehicle telemetry API

## Features

- 🎨 **Modern UI**: Clean, responsive chat interface with Tailwind CSS
- 🌙 **Dark Theme**: App runs in dark mode by default (no toggle)
- 🤖 **AI-Powered**: Uses Gemini for query classification and Pandas script generation
- 📊 **Live Data**: Fetches real-time vehicle telemetry data
- 🔄 **Real-time Processing**: Generates and executes custom data analysis scripts
- 🖼️ **Graphs**: Supports graph/plot queries (e.g., "Graph acu cell 110 temp vs voltage") and returns base64 PNG
- 📋 **Structured Tables**: "Show all" queries display results in scrollable, formatted tables
- 📝 **Error Logging**: Comprehensive error tracking and reporting
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- 🔍 **Intelligent Signal Selection**: Automatically maps user queries to correct vehicle signals (e.g., "battery temperature" → all ACU cell temps, "cell voltages" → all ACU cell voltages)
- 🚀 **Multi-Trip Support**: Query across all trips or specific ranges (e.g., "trips 3-5")

## How to Run

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+ (tested with Python 3.13)
- Gemini API key


### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Backend Setup

1. **Install dependencies:**

```bash
cd backend
pip install -r requirements.txt
```

2. **Configure environment:**

```bash
cp env.example .env
# Edit .env and add your GEMINI API key (and optional settings below)
```

3. **Run the server:**

```bash
python3 -m uvicorn src.main:app --reload
```

The backend API will be available at `http://localhost:8000`

## Workflow

1. **User submits a query** → Frontend sends message to FastAPI backend
2. **Query classification & script gen** → Gemini creates a Pandas/Matplotlib script
3. **If valid:**
   - Script executes securely in Python and fetches JSON itself via an execution helper
   - Computed result and optional base64 PNG image are returned to frontend
4. **If invalid:** Returns polite fallback response

## Error Logging

- Frontend errors are automatically sent to the FastAPI `/log` endpoint
- Backend errors are logged with full stack traces

## Theme

- The UI is dark-only. There is no light/dark toggle.

## Example Usage

### Valid Queries (Vehicle Data)

- "Give me the averages of the mobile speed"
- "What's the maximum inverter temperature recorded?"
- "What is the average temperature of all the accumulator cells?"
- "Graph acu cell 110 temp vs voltage" (returns a chart)
- "Show me all test runs where battery temperature exceeded 25°C AND cell voltages were above 3V" (returns formatted table)
- "Show me all test runs where motor current draw was above 300A" (returns formatted table)
- "Show me battery 22 temperature for all trips" (queries all trips)

**Response:** "The average mobile speed for trip 4 is 45.6 km/h." or formatted table data for "show all" queries.

### Invalid Queries (Non-Vehicle Data)

- "Tell me a joke"
- "What's the weather like?"

**Response:** "Sorry, I can't help you with that."

## API Endpoints

### POST `/llm/query`

Handles user messages and processes vehicle data queries.

**Note:** The endpoint `/query` is also available for backward compatibility and forwards to `/llm/query`.

**Request:**

```json
{
  "message": "Give me the averages of the mobile speed"
}
```

**Response:**

```json
{
  "success": true,
  "message": "The average mobile speed for trip 4 is 45.6 km/h.",
  "data": {
    "script": "import pandas as pd\n# ... generated script",
    "image_base64": "iVBORw0...", // optional when plotting
    "table_data": { // optional for "show all" queries
      "columns": ["trip_id", "acu_cell1_temp", "acu_cell1_voltage"],
      "rows": [
        { "trip_id": "1", "acu_cell1_temp": 25.5, "acu_cell1_voltage": 3.2 },
        { "trip_id": "2", "acu_cell1_temp": 26.1, "acu_cell1_voltage": 3.3 }
      ]
    },
    "signal_scoring": {
      "selected": [["acu_cell16_voltage", 3.5]],
      "top": [{ "signal": "acu_cell16_voltage", "final": 3.5, "ratio": 0.82 }]
    },
    "debug": { "stdout_len": 42, "duration_ms": 120 }
  }
}
```

### POST `/log`

This endpoint allows the frontend (browser) to send error details back to the backend for debugging.
Any time something goes wrong in the user’s browser—such as:

  API failures
  JavaScript errors
  Unexpected UI crashes
  Bad network requests
  Missing data

Accepts JSON (Example):
{
  "level": "error",
  "message": "test log from terminal",
  "error": "optional error field",
  "timestamp": "2025-11-22T17:00:00Z",
  "userAgent": "curl test",
  "url": "/manual-test"
}

Responds with:
{
  "status": "success",
  "message": "Error logged successfully"
}

Use this URL for the logging endpoint for the frontend: http://127.0.0.1:8000/log

### GET/POST `/llm/clear-cache`

Clears both script cache and signal cache. Useful for testing or when signal data has been updated in the database.

### GET `/health`

Health check endpoint for monitoring backend.

Returns:

{
  "status": "healthy",
  "message": "Vehicle Data Chatbot API is running"
}

Use this URL to confirm your backend is online: http://127.0.0.1:8000/health

## Development

### Project Structure

```
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   └── InputBox.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── types/
│   │   │   └── chatbot.ts
│   │   └── App.tsx
│   └── package.json
├── backend/
│   ├── src/
│   │   ├── main.py              # Main FastAPI application
│   │   ├── api/                 # API endpoints (for future expansion)
│   │   │   ├── __init__.py
│   │   │   ├── health.py        # backend health end checkpoint
│   │   │   ├── log.py           # frontend logging endpoint
│   │   ├── llm/                 # LLM integration (separate from api)
│   │   │   ├── __init__.py
│   │   │   ├── router.py        # /llm/query endpoint
│   │   │   ├── agent.py         # Gemini API wrapper
│   │   │   └── tools.py         # Tool schemas + dispatch logic
│   │   └── tools/               # Query execution logic
│   │       ├── __init__.py
│   │       ├── run_queries.py   # Domain-specific querying logic
│   │       └── utils.py         # Signal matching, scoring, DB cache
│   ├── tests/
│   │   ├── test_llm.py          # LLM API tests
│   │   └── test_tools.py        # Tools functionality tests
│   ├── requirements.txt
│   ├── test.sh                  # Custom test runner with progress tracking
│   └── env.example
└── README.md
```

### Key Components

**Frontend:**
- **ChatWindow**: Main chat interface container
- **MessageBubble**: Displays text or base64 graphs from backend
- **InputBox**: Message input with send functionality

**Backend:**
- **src/main.py**: FastAPI application entry point, mounts routers and handles general endpoints
- **src/llm/router.py**: LLM query endpoint handler (`/llm/query`)
- **src/llm/agent.py**: Gemini API integration and wrapper
- **src/llm/tools.py**: Tool schemas and dispatch logic for LLM agent
- **src/tools/run_queries.py**: Query execution logic with Pandas script generation and execution
- **src/tools/utils.py**: Signal matching, scoring, and database caching utilities

### Running Tests

To run the backend tests with custom progress tracking and failure reporting:

```bash
cd backend
chmod +x test.sh  # Make script executable (only needed once)
./test.sh
```

Or run pytest directly:

```bash
cd backend
python3 -m pytest tests/ -v
```

The custom test runner (`test.sh`) will display:
- **Test results**: Clear pass/fail indicators for each test (pytest shows progress percentages naturally)
- **Status codes**: HTTP status codes with meanings (200 = ✅ Healthy, etc.) - shown in test output
- **Failure details**: Shows reasons for failed tests
- **Custom summary**: Formatted summary with total tests, passed, failed, and error counts

## Configuration

### Environment Variables

Create a `.env` in backend:

```env
GEMINI_API_KEY=your_gemini_api_key_here
# Optional tuning (defaults shown)
# GEMINI_MODEL=gemini-1.5-flash
# GEMINI_TIMEOUT=60
# GEMINI_MAX_RETRIES=2
# GEMINI_RETRY_BACKOFF=1.0
# DEBUG_ANALYSIS=false
# GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1
# VEHICLE_DATA_TIMEOUT=120  # Increased for complex multi-trip queries
# SCRIPT_TIMEOUT=120  # Increased for complex data processing
## MySQL connection for signals catalog (required for fuzzy mapping)
DATABASE_HOST=verstappen-ec2.gauchoracing.com
DATABASE_PORT=3306
DATABASE_USER=your_username_here
DATABASE_PASSWORD=your_password_here
DATABASE_NAME=mapache
```

### AI Configuration and Behavior

- The backend uses Gemini to generate a Pandas script tailored to each query.
- The script is required to: fetch the JSON, robustly parse shapes like `data.data`, compute requested metric(s), and both `print(result)` and `set_result(result)`.
- For graph/plot queries, the script produces a Matplotlib PNG and calls `set_image_base64(<base64>)`; the frontend renders it.
- The executor injects helpers:
  - `build_url(signals: list[str])` constructs the exact mapache URL.
  - `http_get(url)` logs the URL and returns a real Response for `.raise_for_status()` and `.json()`.
  - `parse_series(payload, signals)` parses JSON into numeric Series; for single signals it returns a Series, for multiple it returns `{signal: Series}`.
  - `parse_series_df(payload, signals)` returns a DataFrame (safe for `.empty`).
- Transient errors are handled with retries/backoff; timeouts are configurable.

### Advanced Queries Supported

- **Aggregations**: average/mean, min, max, median, percentiles
- **Ranking**: top N / bottom N rows
- **Comparisons**: compute and compare metrics across trips/signals
- **Graphs**: plot temp vs voltage, axes annotated with units (V, C)
- **Multi-Metric Queries**: Filter by multiple conditions (e.g., "battery temperature > 25°C AND cell voltages > 3V")
- **Show All Queries**: Display results in formatted, scrollable tables
- **Multi-Trip Queries**: Query across all trips ("all test runs") or specific ranges ("trips 3-5")

Results include a single-line answer; the executed script, signal scoring, optional debug/plot, and structured table data are returned in `data`.

### Signal Selection & Fuzzy Mapping

- Signals come from MySQL (`SELECT DISTINCT name FROM signal LIMIT 9999`) and are cached in-memory for scoring (no CSV required).
- **Caching**: Signals are cached per `trip_id` to optimize database queries. The SQL query only runs if the requested `trip_id` is not already in the cache. For example:
  - If cache has trip_id 3 and user requests trip_id 3 → uses cache (no SQL query)
  - If cache has trip_id 3 and user requests trip_id 4 → runs SQL query and caches trip_id 4
  - If cache has trip_id 3 and user requests trip_id 3 again → uses cache (no SQL query)
- **Metadata Filtering**: Non-signal fields like `run_id`, `trip_id`, `produced_at`, `vehicle_id`, and `token` are automatically filtered out from signal selection and API requests.
- Queries are mapped to signals using a 0–200 score (0 best). The lowest-scored signal(s) are chosen.
- **Intelligent Signal Mapping**:
  - **Battery/Cell Temperature**: "battery temperature" or "cell temperature" (without number) → selects ALL `acu_cell*_temp` signals (up to 50)
  - **Battery/Cell Voltage**: "cell voltages" or "battery voltage" (without number) → selects ALL `acu_cell*_voltage` signals (up to 50)
  - **Specific Cells**: "battery 22 temperature" → `acu_cell22_temp`, "cell 1 voltage" → `acu_cell1_voltage`
  - **Motor Current**: "motor current draw" → `tcm_power_draw`
  - Exact inference: patterns like "cell 16 temperature" map directly to `acu_cell16_temp`; "cell 16 voltage" maps to `acu_cell16_voltage`.
- **Multi-Metric Queries**: When multiple metrics are requested (e.g., "temperature AND voltage"), the system selects signals for each metric independently and combines them.
- For correlation/"vs" queries, the top two signals are selected.
- If best score > 100, the query is considered unrelated and a polite fallback is returned.
- Response includes `data.signal_scoring` for transparency.

### Clarification Flow

- If a query mentions `cell` and `temperature` or `voltage` but lacks a cell number, backend responds with:
  "Which cell number for temperature/voltage? e.g., 16 or 110" and data `{ intent: "clarify_cell_metric", metric: "temperature|voltage" }`.
- The frontend merges the follow-up (e.g., "cell 16") into the original request to preserve the user's metric intent (e.g., only "max").
- If a query omits the trip/run, backend asks: "Which trip (run) number? e.g., 3, or 'all' for all trips, or 'trips 3-5' for a range" and data `{ intent: "clarify_trip" }`. Frontend merges the run number into the last query.
- **Trip ID Support**: The system accepts:
  - Single trip: "trip 3" or "run 3"
  - All trips: "all test runs" or "all trips"
  - Range: "trips 3-5" or "runs 3-5"
