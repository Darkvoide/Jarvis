# JARVIS One-Command PowerShell Launcher
param(
    [string]$Mode = "all" # Options: all, voice, text, gesture
)

Set-Location $PSScriptRoot

Write-Host ">>> [1/3] Pulling Ollama model (qwen2.5:7b)..." -ForegroundColor Cyan
ollama pull qwen2.5:7b

Write-Host ">>> [2/3] Verifying/Installing dependencies..." -ForegroundColor Cyan
& .venv\Scripts\pip install -r requirements.txt

Write-Host ">>> [3/3] Launching JARVIS (Mode: $Mode)..." -ForegroundColor Green
& .venv\Scripts\python run.py --mode $Mode
