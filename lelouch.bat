@echo off
title Lelouch - Video Harvester
cd /d "%~dp0"

:: Check if venv exists
if exist ".venv\Scripts\activate.bat" goto :activate

echo [Lelouch] No virtual environment found. Creating .venv ...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create venv. Is Python 3.10+ installed?
    pause
    exit /b 1
)

echo [Lelouch] Installing dependencies ...
call .venv\Scripts\activate.bat
pip install -e .
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)
echo [Lelouch] Setup complete!
echo.
goto :run

:activate
call .venv\Scripts\activate.bat

:run
python -m lelouch --simple %*
