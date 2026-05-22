@echo off
set PROJECT=C:\Users\kashi\OneDrive\Desktop\AI-PR Intelligence
set VENV=C:\ai_pr_venv

echo === Training classifier ===
cd /d "%PROJECT%"
%VENV%\Scripts\python.exe -m ml.train_classifier
if errorlevel 1 (
    echo FAILED training
    exit /b 1
)
echo === TRAINING COMPLETE ===
