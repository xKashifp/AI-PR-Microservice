@echo off
set PROJECT=C:\Users\kashi\OneDrive\Desktop\AI-PR Intelligence
set VENV=C:\ai_pr_venv
cd /d "%PROJECT%"
%VENV%\Scripts\python.exe -m pytest tests/ --cov=app --cov-report=term-missing -v 2>&1
