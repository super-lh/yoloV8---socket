@echo off
cd /d "%~dp0"
set ARGS=
:loop
if "%~1"=="" goto run
set ARGS=%ARGS% "%~1"
shift
goto loop
:run
"D:\anaconda\python.exe" detect.py %ARGS%
echo.
echo === finished, press any key to close ===
pause >nul
