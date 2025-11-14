#!/bin/bash

# Vehicle Data Chatbot Startup Script

echo "🚀 Starting Vehicle Data Chatbot..."

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Function to start backend
start_backend() {
    echo "🔧 Starting FastAPI backend..."
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "📥 Installing Python dependencies..."
    echo "   Installing with minimal requirements for Python 3.13 compatibility..."
    pip install -r requirements-minimal.txt
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        echo "⚠️  Warning: .env file not found. Please create one from env.example"
        echo "   You need to add your OpenRouter API key to the .env file"
    fi
    
    # Start backend
    echo "🚀 Starting backend server on http://localhost:8000"
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    
    cd ..
}

# Function to start frontend
start_frontend() {
    echo "🎨 Starting React frontend..."
    cd frontend
    
    # Install dependencies
    echo "📥 Installing Node.js dependencies..."
    npm install
    
    # Start frontend
    echo "🚀 Starting frontend server on http://localhost:5173"
    npm run dev &
    FRONTEND_PID=$!
    
    cd ..
}

# Start both services
start_backend
start_frontend

echo ""
echo "✅ Both services are starting up!"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for interrupt
trap 'echo ""; echo "🛑 Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' INT

# Keep script running
wait
