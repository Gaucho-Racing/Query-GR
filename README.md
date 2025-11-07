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
- 📝 **Error Logging**: Comprehensive error tracking and reporting
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices

## How to Run

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+ (tested with Python 3.13)
- Gemini API key

**Note:** For Python 3.13 users, the project uses `requirements-minimal.txt` to ensure compatibility.

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
pip install -r requirements-minimal.txt
```

2. **Configure environment:**

```bash
cp env.example .env
# Edit .env and add your GEMINI API key (and optional settings below)
```

3. **Run the server:**

```bash
uvicorn main:app --reload
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

**Response:** "The average mobile speed for trip 4 is 45.6 km/h."

### Invalid Queries (Non-Vehicle Data)

- "Tell me a joke"
- "What's the weather like?"

**Response:** "Sorry, I can't help you with that."

## API Endpoints

### POST `/query`

Handles user messages and processes vehicle data queries.

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
    "signal_scoring": {
      "selected": [["acu_cell16_voltage", 3.5]],
      "top": [{ "signal": "acu_cell16_voltage", "final": 3.5, "ratio": 0.82 }]
    },
    "debug": { "stdout_len": 42, "duration_ms": 120 }
  }
}
```

### POST `/log`

Receives and stores frontend error reports.

### GET `/health`

Health check endpoint for monitoring.

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
│   ├── main.py
│   ├── requirements.txt
│   ├── requirements-minimal.txt
│   └── env.example
└── README.md
```

### Key Components

- **ChatWindow**: Main chat interface container
- **MessageBubble**: Displays text or base64 graphs from backend
- **InputBox**: Message input with send functionality

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
# VEHICLE_DATA_TIMEOUT=30
# SCRIPT_TIMEOUT=20
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

- Aggregations: average/mean, min, max, median, percentiles
- Ranking: top N / bottom N rows
- Comparisons: compute and compare metrics across trips/signals
- Graphs: plot temp vs voltage, axes annotated with units (V, C)

Results include a single-line answer; the executed script, signal scoring and optional debug/plot are returned in `data`.

### Signal Selection & Fuzzy Mapping

- Signals come from MySQL (`SELECT DISTINCT name FROM signal LIMIT 9999`) and are cached in-memory for scoring (no CSV required).
- Queries are mapped to signals using a 0–200 score (0 best). The lowest-scored signal(s) are chosen.
- Exact inference: patterns like "cell 16 temperature" map directly to `acu_cell16_temp`; "cell 16 voltage" maps to `acu_cell16_voltage`.
- For correlation/"vs" queries, the top two signals are selected.
- If best score > 100, the query is considered unrelated and a polite fallback is returned.
- Response includes `data.signal_scoring` for transparency.

### Clarification Flow

- If a query mentions `cell` and `temperature` or `voltage` but lacks a cell number, backend responds with:
  "Which cell number for temperature/voltage? e.g., 16 or 110" and data `{ intent: "clarify_cell_metric", metric: "temperature|voltage" }`.
- The frontend merges the follow-up (e.g., "cell 16") into the original request to preserve the user's metric intent (e.g., only "max").
- If a query omits the trip/run, backend asks: "Which trip (run) number? e.g., 3" and data `{ intent: "clarify_trip" }`. Frontend merges the run number into the last query.
