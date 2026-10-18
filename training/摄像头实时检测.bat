@echo off
cd /d "%~dp0"
"D:\anaconda\python.exe" detect.py 0 --conf 0.35
echo.
echo === finished, press any key to close ===
pause >nul
