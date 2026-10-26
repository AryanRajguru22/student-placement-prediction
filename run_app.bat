@echo off
rem Starts the Student Placement Probability Predictor on Windows.
rem First run: creates a local virtual environment (.venv) and installs requirements-app.txt (internet needed once). Later runs just start the app.
rem Stop the app with Ctrl+C in this window.
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
    echo ERROR: Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/ and run this file again.
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
    if errorlevel 1 (
        echo ERROR: Python 3.11 or newer is required. Install it from https://www.python.org/downloads/ and run this file again.
        goto :fail
    )
    %PY% -m venv ".venv"
    if errorlevel 1 ( echo ERROR: could not create the virtual environment. & goto :fail )
)

if not exist ".venv\.requirements-app.installed" (
    echo Installing dependencies from requirements-app.txt ...
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r "requirements-app.txt"
    if errorlevel 1 ( echo ERROR: dependency installation failed. Check your internet connection and the messages above. & goto :fail )
    echo ok> ".venv\.requirements-app.installed"
)

echo Starting the app. Your browser should open http://localhost:8501 in a few seconds (open it yourself if not). Press Ctrl+C in this window to stop.
rem headless = skip Streamlit's first-run e-mail prompt; the browser is opened here instead
start "" /b cmd /c "ping -n 7 127.0.0.1 >nul & start "" http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run "app\app.py" --server.headless true
if errorlevel 1 ( echo ERROR: the app stopped with an error. See app\README.md, section Troubleshooting. & goto :fail )
endlocal
exit /b 0

:fail
echo.
pause
endlocal
exit /b 1
