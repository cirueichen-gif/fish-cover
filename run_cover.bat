@echo off
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "PYTHONPATH=%~dp0extra_packages;%PYTHONPATH%"
rem Separation needs torchvision -> dsh-voice has it, Applio env does not.
set "PY=C:\Users\cirue\dsh-voice\Scripts\python.exe"
if not exist "%PY%" set "PY=C:\Users\cirue\Applio\env\Scripts\python.exe"
title Fish Cover 6970
"%PY%" server.py --open
echo.
echo Stopped. Press any key to close.
pause >nul
