@echo off
REM Startup script untuk YOLO v12 Eye Disease Detection API (Windows)

setlocal enabledelayedexpansion

echo.
echo ========================================
echo YOLO v12 Eye Disease Detection API
echo Startup Script (Windows)
echo ========================================
echo.

REM Configuration
set MODEL_PATH=runs\detect\train4\weights\best.pt
set PORT=8000
set WORKERS=4
set HOST=0.0.0.0

REM Check model exists
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model not found at: %MODEL_PATH%
    echo Please ensure model is trained and saved at: runs\detect\train4\weights\best.pt
    pause
    exit /b 1
)

echo [OK] Model found: %MODEL_PATH%
echo.

REM Check dependencies
echo Checking dependencies...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)

for %%p in (fastapi uvicorn torch ultralytics cv2 numpy) do (
    python -c "import %%p" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] %%p not installed
        echo Run: pip install -r requirements-api-v2.txt
        pause
        exit /b 1
    )
    echo [OK] %%p installed
)

echo.

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "uploads" mkdir uploads

REM Check GPU
echo Checking GPU availability...
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('CPU only')"

echo.
echo ========================================
echo Starting API server...
echo ========================================
echo Configuration:
echo   - Host: %HOST%
echo   - Port: %PORT%
echo   - Workers: %WORKERS%
echo   - Model: %MODEL_PATH%
echo.
echo Server running at:
echo   - API Documentation: http://localhost:%PORT%/v1/docs
echo   - Alternative Docs: http://localhost:%PORT%/v1/redoc
echo   - Health Check: http://localhost:%PORT%/v1/health
echo.

REM Run server
python -m uvicorn api_v2:app --host %HOST% --port %PORT% --reload --log-level info

pause
