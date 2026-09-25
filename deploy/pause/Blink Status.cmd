@echo off
rem Blink Status: what Blink is doing, changing nothing. It prints `blink status` for every run that
rem trains or waits out a pause, one nvidia-smi line (GPU use, temperature, power), whether the pause
rem flag is up, and the latest strength check (EVAL.md PR-6). "Pause Blink.cmd" and "Resume Blink.cmd"
rem sit beside it. BLINK_HOME is D:\blink and Blink's python the main checkout's venv unless set.
setlocal
if not defined BLINK_HOME set "BLINK_HOME=D:\blink"
if not defined BLINK_PYTHON set "BLINK_PYTHON=C:\dev\blink-chess\.venv\Scripts\python.exe"
if not exist "%BLINK_HOME%\" (
    echo The Blink folder %BLINK_HOME% was not found.
    goto done
)
echo Training:
"%BLINK_PYTHON%" -m blink.cli status --live
echo.
echo GPU use, temperature in C, power:
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo   nvidia-smi was not found, so the GPU cannot be shown.
) else (
    call nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
)
echo.
if exist "%BLINK_HOME%\PAUSE" (
    echo The pause flag is up: Blink training waits until Resume Blink removes %BLINK_HOME%\PAUSE.
) else (
    echo The pause flag is down: a launched Blink run trains.
)
echo.
echo Latest strength check:
"%BLINK_PYTHON%" -m blink.cli eval strength --show --last 1
:done
echo.
pause
exit /b 0
