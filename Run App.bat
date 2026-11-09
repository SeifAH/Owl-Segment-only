@echo off
setlocal
title OWLv2 + SAM Object Finder
cd /d "%~dp0"

rem ---------------------------------------------------------------------------------------------
rem  Double-click to start. First run: installs uv + Python 3.11 + packages (~1 GB) and downloads
rem  the OWLv2 and SAM models (~1 GB). Later runs start in a few seconds and work offline.
rem ---------------------------------------------------------------------------------------------

call scripts\setup.bat
if defined SETUP_FAILED goto :fail

echo.
echo Starting the app at http://127.0.0.1:7862 - your browser opens automatically.
echo Keep this window open while using the app; close it to stop the app.
echo.
uv run --frozen python app.py
if errorlevel 1 goto :fail
goto :eof

:fail
echo.
echo Something went wrong - see the messages above.
pause
exit /b 1
