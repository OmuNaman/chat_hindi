@echo off
echo ================================
echo nano_hindi Environment Setup
echo ================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch --index-url https://download.pytorch.org/whl/cu121

REM Install other requirements
echo Installing other requirements...
pip install -r requirements.txt

echo.
echo ================================
echo Setup complete!
echo ================================
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To download data:
echo   python data/download.py --max_tokens 500000000
echo.
echo To preprocess data:
echo   python data/preprocess.py
echo.
echo To train:
echo   python train.py --config 25m
echo.
echo To generate text:
echo   python generate.py --checkpoint checkpoints/checkpoint_step1000.pt --interactive
echo.
pause
