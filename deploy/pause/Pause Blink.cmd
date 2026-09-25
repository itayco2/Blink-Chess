@echo off
rem Pause Blink: Blink training saves a checkpoint and lets go of the GPU, so the PC is free for games.
rem It creates BLINK_HOME\PAUSE (D:\blink\PAUSE unless BLINK_HOME says otherwise), then watches
rem nvidia-smi for up to 5 minutes until no Blink process holds the GPU. "Resume Blink.cmd" undoes it.
rem The GPU check is PowerShell, kept at the end of this file: cmd.exe stops at "exit /b" and never
rem reads it, and PowerShell runs only what follows its marker line.
setlocal
if not defined BLINK_HOME set "BLINK_HOME=D:\blink"
if not defined BLINK_PAUSE_WAIT_S set "BLINK_PAUSE_WAIT_S=300"
set "BLINK_PAUSE_SCRIPT=%~f0"
set "code=2"
if not exist "%BLINK_HOME%\" (
    echo The Blink folder %BLINK_HOME% was not found, so nothing was paused.
    goto done
)
> "%BLINK_HOME%\PAUSE" echo Paused by Pause Blink.cmd on %DATE% at %TIME%. Resume Blink.cmd removes this file.
if not exist "%BLINK_HOME%\PAUSE" (
    echo Could not create %BLINK_HOME%\PAUSE, so nothing was paused.
    goto done
)
echo Blink is pausing: it saves a checkpoint, then lets go of the GPU.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$text = [IO.File]::ReadAllText($env:BLINK_PAUSE_SCRIPT); Invoke-Expression $text.Substring($text.LastIndexOf('#BEGIN' + ' POWERSHELL'))"
set "code=%errorlevel%"
:done
echo.
pause
exit /b %code%

#BEGIN POWERSHELL: the GPU check (Pause Blink.cmd runs this part; see the top of the file)
$waitSeconds = [int]$env:BLINK_PAUSE_WAIT_S
$deadline = (Get-Date).AddSeconds($waitSeconds)
$waitText = if ($waitSeconds -ge 60) { "$([math]::Round($waitSeconds / 60, 1)) minutes" } else { "$waitSeconds seconds" }
$blink = 'blink\.cli|blink\.exe'  # how a Blink process's command line reads

function Get-BlinkOnGpu {
    # One line per Blink process that nvidia-smi lists as holding the GPU; none once Blink let go.
    $ids = @(nvidia-smi --query-compute-apps=pid --format=csv,noheader | ForEach-Object { "$_".Trim() })
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi exited with code $LASTEXITCODE" }
    Get-CimInstance Win32_Process | Where-Object {
        $ids -contains [string]$_.ProcessId -and $_.CommandLine -match $blink
    } | ForEach-Object { "pid $($_.ProcessId): $($_.CommandLine.Trim())" }
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host 'nvidia-smi was not found, so the GPU cannot be checked.'
    Write-Host 'Blink still pauses: it lets go of the GPU at its next step, usually within a minute.'
    exit 2
}
Write-Host "Waiting for Blink to let go of the GPU (up to $waitText)..."
try {
    while ($true) {
        $still = @(Get-BlinkOnGpu)
        if ($still.Count -eq 0) {
            Write-Host 'Blink is paused and the GPU is free. Have fun.'
            exit 0
        }
        if ((Get-Date) -ge $deadline) {
            Write-Host "Blink is paused, but after $waitText these Blink processes still hold the GPU:"
            $still | ForEach-Object { Write-Host "  $_" }
            Write-Host 'They may still be saving a checkpoint: run Pause Blink again in a minute to check.'
            exit 1
        }
        Start-Sleep -Seconds 5
    }
} catch {
    Write-Host "The GPU could not be checked: $($_.Exception.Message)"
    Write-Host 'Blink still pauses: it lets go of the GPU at its next step, usually within a minute.'
    exit 2
}
