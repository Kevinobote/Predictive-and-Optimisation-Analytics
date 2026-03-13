@echo off
echo ==========================================
echo 🎙️  Tubonge - Speech Analytics
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
    echo.
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -q -r requirements.txt
echo ✅ Dependencies installed
echo.

REM Start the server
echo 🚀 Starting Tubonge server...
echo.
echo 📍 Application: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo 🔧 Health Check: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo ==========================================
echo.

python main.py
