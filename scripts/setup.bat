@echo off
rem Shared setup for "Run App.bat" and "Run on Folder.bat" (called with CALL, from the project root).
rem Installs uv if needed, syncs the Python environment, downloads the OWLv2 and SAM models once. Sets SETUP_FAILED=1 on error.

rem Keep the Python environment outside the project folder (OneDrive breaks virtual environments)
set "UV_PROJECT_ENVIRONMENT=%USERPROFILE%\.venvs\owl-sam-segment"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
set "SETUP_FAILED="

where uv >nul 2>nul
if errorlevel 1 (
    echo [setup] Installing uv - the Python package manager ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto :failed
)

echo [setup] Checking the Python environment ...
set "TRIES=0"
:sync
uv sync --frozen
if not errorlevel 1 goto :synced
rem Some wheels unpack read-only folders, and antivirus can briefly lock fresh files; clear flags, wait, retry
set /a TRIES+=1
if %TRIES% geq 3 goto :failed
echo [setup] Retrying %TRIES%/2 after clearing read-only flags in the environment ...
attrib -R "%UV_PROJECT_ENVIRONMENT%\*" /S /D >nul 2>nul
ping -n 4 127.0.0.1 >nul
goto :sync
:synced
if not exist "%UV_PROJECT_ENVIRONMENT%\.models_ok" (
    echo [setup] Downloading the OWLv2 and SAM models, about 1 GB - first run only ...
    uv run --frozen python scripts\download_models.py
    if errorlevel 1 goto :failed
    echo ok> "%UV_PROJECT_ENVIRONMENT%\.models_ok"
)

rem Models are cached locally - no need to contact the internet
set "HF_HUB_OFFLINE=1"
exit /b 0

:failed
set "SETUP_FAILED=1"
exit /b 1
