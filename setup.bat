@echo off
set PROJECT=C:\Users\kashi\OneDrive\Desktop\AI-PR Intelligence
set VENV=C:\ai_pr_venv

echo === Creating venv at %VENV% ===
C:\Python314\python.exe -m venv %VENV% --clear
if errorlevel 1 goto :err

echo === Upgrading pip ===
%VENV%\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :err

echo === Installing requirements ===
%VENV%\Scripts\python.exe -m pip install -r "%PROJECT%\requirements.txt"
if errorlevel 1 goto :err

echo === Training classifier ===
cd /d "%PROJECT%"
%VENV%\Scripts\python.exe -m ml.train_classifier
if errorlevel 1 goto :err

echo === SETUP COMPLETE ===
goto :eof

:err
echo FAILED with error %errorlevel%
exit /b 1
