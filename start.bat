@echo off
REM Vehicle Data Chatbot Startup Script for Windows

echo 🚀 Starting Vehicle Data Chatbot...

REM Check if we're in the right directory
if not exist "README.md" (
    echo ❌ Please run this script from the project root directory
    pause
    exit /b 1
)

REM Start backend
echo 🔧 Starting FastAPI backend...
cd backend

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing Python dependencies...
pip install -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  Warning: .env file not found. Please create one from env.example
    echo    You need to add your OpenRouter API key to the .env file
)

REM Start backend in new window
echo 🚀 Starting backend server on http://localhost:8000
start "Backend Server" cmd /k "uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"

cd ..

REM Start frontend
echo 🎨 Starting React frontend...
cd frontend

REM Install dependencies
echo 📥 Installing Node.js dependencies...
npm install

REM Start frontend
echo 🚀 Starting frontend server on http://localhost:5173
start "Frontend Server" cmd /k "npm run dev"

cd ..

echo.
echo ✅ Both services are starting up!
echo    Frontend: http://localhost:5173
echo    Backend:  http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo.
echo Press any key to exit...
pause >nul
