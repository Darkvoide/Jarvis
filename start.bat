@echo off
title JARVIS Launcher
cd /d "%~dp0"

echo [1/3] Checking Ollama model...
ollama pull qwen2.5:7b

echo [2/3] Installing dependencies...
call .venv\Scripts\pip install -r requirements.txt

echo [3/3] Starting JARVIS (Voice + Text Mode)...
call .venv\Scripts\python.exe run.py --mode all
pause
