@echo off
title NHL Daily Predictions Pipeline
echo ============================================================
echo   NHL DAILY PREDICTIONS PIPELINE
echo   %date% %time%
echo ============================================================
echo.

REM ── Step 0: Activate conda environment ──
call C:\Users\shell\miniconda3\Scripts\activate.bat base

REM ── Step 1: Update box scores and features ──
echo [1/3] Updating box scores and features...
echo.
cd /d C:\Users\shell\OneDrive\Documents\NHL_Player\Boxscores
python daily_update.py
echo.

REM ── Step 2: Grade yesterday's player picks ──
echo [2/3] Grading yesterday's player prop results...
echo.
cd /d C:\Users\shell\OneDrive\Documents\NHL_Player
python track_player_results.py
echo.

REM ── Step 3: Run today's predictions ──
echo [3/3] Generating player prop predictions...
echo.
cd /d C:\Users\shell\OneDrive\Documents\NHL_Player
python predict.py
echo.

echo ============================================================
echo   ALL DONE - %date% %time%
echo ============================================================
echo.
echo Press any key to close...
pause >nul