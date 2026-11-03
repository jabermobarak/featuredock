@echo off
cd /d "%~dp0"
if not exist .venv python -m venv .venv
if errorlevel 1 exit /b 1
call .venv\Scripts\activate.bat
python -m pip install -e .
if errorlevel 1 exit /b 1
python -m featuredock.cli demo
if errorlevel 1 exit /b 1
python -m featuredock.cli serve
