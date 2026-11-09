@echo off
setlocal
title OWLv2 + SAM Object Finder - folder
rem ---------------------------------------------------------------------------------------------
rem  Process every image in a folder: drag the folder onto this file, or double-click and paste a path.
rem  Results (annotated images + summary.csv + objects.csv) go to outputs\batch_<timestamp>.
rem ---------------------------------------------------------------------------------------------

set "FOLDER=%~1"
cd /d "%~dp0"
if not defined FOLDER (
    echo Drag a folder onto "Run on Folder.bat", or paste the folder path here and press Enter.
    set /p "FOLDER=Folder: "
)
if not defined FOLDER goto :fail
rem strip quotes a pasted path may carry
set "FOLDER=%FOLDER:"=%"
if not exist "%FOLDER%\" (
    echo Folder not found: %FOLDER%
    goto :fail
)

call scripts\setup.bat
if defined SETUP_FAILED goto :fail

echo.
uv run --frozen python scripts\batch.py "%FOLDER%"
if errorlevel 1 goto :fail
if not defined NO_OPEN (
    echo.
    echo Opening the results folder ...
    for /f "delims=" %%D in ('dir /b /ad /o-d "outputs\batch_*" 2^>nul') do (
        start "" "outputs\%%D"
        goto :done
    )
)
:done
pause
goto :eof

:fail
echo.
echo Something went wrong - see the messages above.
pause
exit /b 1
