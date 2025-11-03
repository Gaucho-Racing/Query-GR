# Vehicle Data Chatbot

A modern React + TypeScript chatbot that answers telemetry questions by generating and running Pandas scripts on live JSON data from vehicle sensors.

## Purpose

This chatbot provides an intelligent interface for querying vehicle telemetry data. Users can ask natural language questions about vehicle performance metrics, and the system will:

1. Classify whether the query is vehicle-data related
2. Fetch live data from the vehicle API
3. Generate custom Pandas scripts using AI
4. Execute the scripts to compute requested metrics
5. Return formatted results to the user

## Tech Stack

- **Frontend**: React + TypeScript + Tailwind CSS + Vite
- **Backend**: FastAPI + Pandas + HTTPX
- **AI Integration**: Gemini (intent detection + code generation)
- **Data Source**: Live vehicle telemetry API

## Features

- 🎨 **Modern UI**: Clean, responsive chat interface with Tailwind CSS
- 🌙 **Dark Theme**: App runs in dark mode by default (no toggle)
- 🤖 **AI-Powered**: Uses Gemini for query classification and Pandas script generation
- 📊 **Live Data**: Fetches real-time vehicle telemetry data
- 🔄 **Real-time Processing**: Generates and executes custom data analysis scripts
- 📝 **Error Logging**: Comprehensive error tracking and reporting
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices

## How to Run

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+ (tested with Python 3.13)
- Gemini API key

**Note:** For Python 3.13 users, the project uses `requirements-minimal.txt` to ensure compatibility with the latest Python version.

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
2. **Query classification & script gen** → Gemini creates a Pandas script
3. **If valid:**
   - Script executes securely in Python and fetches JSON itself via an execution helper
   - Computed result returned to frontend
4. **If invalid:** Returns polite fallback response

## Error Logging

The system includes comprehensive error logging:

- Frontend errors are automatically sent to the FastAPI `/log` endpoint
- Backend errors are logged with full stack traces
- All errors include timestamp, user agent, and context information

## Theme

- The UI is dark-only. There is no light/dark toggle.
- Tailwind `dark:` utilities are used with class-based dark mode.

## Example Usage

### Valid Queries (Vehicle Data)

- "Give me the averages of the mobile speed"
- "What's the maximum acceleration during the trip?"
- "What was the maximum inverter temperature recorded?"
- "What is the average temperature of all the accumulator cells?"

**Response:** "The average mobile speed for trip 4 is 45.6 km/h."

### Invalid Queries (Non-Vehicle Data)

- "Tell me a joke"
- "What's the weather like?"
- "How do I cook pasta?"

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
    "script": "import pandas as pd\n# ... generated script"
  }
}
```

### POST `/log`

Receives and stores frontend error reports.

**Request:**

```json
{
  "error": "Network request failed",
  "timestamp": "2024-01-15T10:30:00Z",
  "userAgent": "Mozilla/5.0...",
  "url": "http://localhost:5173"
}
```

### GET `/health`

Health check endpoint for monitoring.

## Development

### Project Structure

```
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── ChatWindow.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   └── InputBox.tsx
│   │   ├── contexts/            # (removed)
│   │   ├── services/            # API services
│   │   │   └── api.ts
│   │   ├── types/               # TypeScript types
│   │   │   └── chatbot.ts
│   │   └── App.tsx
│   └── package.json
├── backend/
│   ├── main.py                  # FastAPI application
│   ├── requirements.txt         # Python dependencies
│   └── env.example             # Environment variables template
└── README.md
```

### Key Components

- **ChatWindow**: Main chat interface container
- **MessageBubble**: Individual message display component
- **InputBox**: Message input with send functionality

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
# Optional tuning (with defaults shown)
# GEMINI_MODEL=gemini-1.5-flash
# GEMINI_TIMEOUT=60
# GEMINI_MAX_RETRIES=2
# GEMINI_RETRY_BACKOFF=1.0
# DEBUG_ANALYSIS=false
# GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1
```

### AI Configuration and Behavior

- The backend uses Gemini to generate a Pandas script tailored to each query.
- The script is required to: fetch the JSON, robustly parse shapes like `data.data`, compute the asked metric(s), and both `print(result)` and call `set_result(result)`.
- The executor injects a helper `http_get(url)` that logs the exact URL the script calls and returns a real response object (so scripts can do `resp.raise_for_status()` and `resp.json()`).
- The backend enforces the exact data URL and logs it, enabling auditing of requests.
- Transient errors are handled with retries and exponential backoff (configurable via env above). Timeouts are also adjustable.

### Advanced Queries Supported

- Aggregations: average/mean, min, max, median, percentiles
- Ranking: top N / bottom N mobile_speed rows
- Comparisons: compute and compare metrics across trips/signals

Results are returned as a single line of text to the frontend; the executed script and optional debug info are included in the response data.

### API Configuration

The vehicle data API is configured with:

- **URL**: `https://mapache.gauchoracing.com/api/query/signals`
- **Vehicle ID**: `gr24-main`
- **Trip ID**: `4`
- **Default Signal**: `mobile_speed`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request
