@echo off
echo ========================================
echo ShieldX Gateway Production Test Suite
echo ========================================

echo.
echo Step 1: Installing Python dependencies...
pip install requests flask > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Failed to install Python dependencies
    echo Please install Python and pip first
    pause
    exit /b 1
)

echo Step 2: Starting mock services...
start "Mock Services" python mock_services.py
timeout /t 3 > nul

echo Step 3: Starting ShieldX Gateway...
echo Note: This requires Go to be installed
echo If Go is not available, please install it first
start "ShieldX Gateway" cmd /c "go run main.go"
timeout /t 5 > nul

echo Step 4: Running tests...
echo.
python test_gateway.py

echo.
echo ========================================
echo Test completed!
echo ========================================
echo.
echo To stop services:
echo - Close the "Mock Services" window
echo - Close the "ShieldX Gateway" window
echo.
pause