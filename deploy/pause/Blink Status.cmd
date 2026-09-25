@echo off
rem Blink Status: what Blink is doing, changing nothing. It prints `blink status` for every run that
rem trains or waits out a pause, one nvidia-smi line (GPU use, temperature, power), whether the pause
rem flag is up, and the latest strength check (EVAL.md PR-6). "Pause Blink.cmd" and "Resume Blink.cmd"
rem sit beside it. BLINK_HOME is D:\blink and Blink's python the main checkout's venv unless set.
rem Blink runs from BLINK_REPO, the checkout the flagship trains in (C:\dev\blink-run unless set):
rem pressed from the Desktop, the venv alone would find the main checkout, whose blink may not have
rem `status --live` or `eval strength` yet.
setlocal
if not defined BLINK_HOME set "BLINK_HOME=D:\blink"
if not defined BLINK_PYTHON set "BLINK_PYTHON=C:\dev\blink-chess\.venv\Scripts\python.exe"
if not defined BLINK_REPO set "BLINK_REPO=C:\dev\blink-run"
if not exist "%BLINK_HOME%\" (
    echo The Blink folder %BLINK_HOME% was not found.
    goto done
)
set "in_repo="
if exist "%BLINK_REPO%\blink\cli.py" (
    pushd "%BLINK_REPO%"
    set "in_repo=1"
) else (
    echo The Blink checkout %BLINK_REPO% was not found, so the runs and strength checks are not shown.
    echo.
)
echo Training:
if defined in_repo "%BLINK_PYTHON%" -m blink.cli status --live
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
if defined in_repo "%BLINK_PYTHON%" -m blink.cli eval strength --show --last 1
if defined in_repo popd
:done
echo.
pause
exit /b 0
