@echo off
REM Launch script for Streamlit app on Windows
REM Run this from the project root directory

echo ============================================================
echo NBA Team Selection - Streamlit App Launcher
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if data file exists
if exist "data\nba_players.csv" (
    echo [OK] Data file found at data\nba_players.csv
) else (
    echo [WARNING] Data file not found at data\nba_players.csv
    echo Please ensure your NBA dataset is placed at: data\nba_players.csv
)
echo.

REM Check if Streamlit app exists
if exist "app\streamlit_app.py" (
    echo [OK] Streamlit app found at app\streamlit_app.py
) else (
    echo [ERROR] Streamlit app not found at app\streamlit_app.py
    pause
    exit /b 1
)
echo.

echo Launching Streamlit app...
echo ============================================================
echo.

REM Launch Streamlit from project root
streamlit run app\streamlit_app.py

pause