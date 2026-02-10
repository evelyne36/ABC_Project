@echo off
REM MetaGB vs XGBoost Comparison Experiments
REM Windows batch script to run experiments

echo ========================================
echo MetaGB vs XGBoost Comparison Experiments
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM Parse arguments
set EXP=all
set SEED=42

:parse_args
if "%~1"=="" goto run
if "%~1"=="--exp" (
    set EXP=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--seed" (
    set SEED=%~2
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:run
echo Running experiments: %EXP%
echo Random seed: %SEED%
echo.

cd /d "%~dp0"
python run_all_experiments.py --exp %EXP% --seed %SEED%

echo.
echo ========================================
echo Experiments completed!
echo ========================================
pause
