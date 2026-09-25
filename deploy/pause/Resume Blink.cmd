@echo off
rem Resume Blink: removes BLINK_HOME\PAUSE (D:\blink\PAUSE unless BLINK_HOME says otherwise), which
rem "Pause Blink.cmd" created. Blink training starts again from its checkpoint within a minute.
rem It never lifts a P7-VAA pause: that gate is cleared by hand.
setlocal
if not defined BLINK_HOME set "BLINK_HOME=D:\blink"
if not exist "%BLINK_HOME%\PAUSE" (
    echo Blink was not paused, so there is nothing to resume.
    goto done
)
del /f /q "%BLINK_HOME%\PAUSE"
if exist "%BLINK_HOME%\PAUSE" (
    echo Could not remove %BLINK_HOME%\PAUSE: close anything that has it open, then try again.
    goto done
)
echo Blink will resume within a minute.
:done
echo.
pause
