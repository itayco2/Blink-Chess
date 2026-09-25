@echo off
rem Pause Blink: Blink training saves a checkpoint and lets go of the GPU, so the PC is free for games.
rem It creates BLINK_HOME\PAUSE (D:\blink\PAUSE unless BLINK_HOME says otherwise), then watches
rem nvidia-smi for up to 5 minutes until no Blink process holds the GPU. "Resume Blink.cmd" undoes it.
rem A Blink process is one whose command line, or whose parent's, runs blink.cli or a Blink folder's
rem venv python (so `python -m pytest -m cuda` from C:\dev\blink-chess counts too). Supervised training
rem stops for the flag; anything else Blink runs (a calibration, a test, a benchmark) runs to its end,
rem and while one holds the GPU this says so plainly: do not start a game.
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
# how a Blink command line reads: blink.cli (or blink.exe), or a Blink folder's venv python
$blink = 'blink\.cli|blink\.exe|\\blink[\w.-]*\\\.venv\\Scripts\\python'

function Get-BlinkOnGpu {
    # One line per process nvidia-smi lists as holding the GPU whose own command line or whose
    # parent's is Blink's (the venv's python.exe starts the real interpreter as its child); none once
    # Blink let go.
    $ids = @(nvidia-smi --query-compute-apps=pid --format=csv,noheader | ForEach-Object { "$_".Trim() })
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi exited with code $LASTEXITCODE" }
    $byId = @{}
    Get-CimInstance Win32_Process | ForEach-Object { $byId[[string]$_.ProcessId] = $_ }
    foreach ($id in $ids) {
        $proc = $byId[$id]
        if ($null -eq $proc) { continue }
        $parent = $byId[[string]$proc.ParentProcessId]
        if ($null -ne $parent -and $parent.CreationDate -gt $proc.CreationDate) { $parent = $null }  # a reused ID
        $own = "$($proc.CommandLine)".Trim()
        if ($own -match $blink) {
            "pid ${id}: $own"
        } elseif ($null -ne $parent -and "$($parent.CommandLine)" -match $blink) {
            "pid ${id}: $own (started by $("$($parent.CommandLine)".Trim()))"
        }
    }
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host 'nvidia-smi was not found, so the GPU cannot be checked.'
    Write-Host 'Supervised Blink training still pauses: it lets go of the GPU at its next step, usually within a minute.'
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
            Write-Host 'Blink could NOT free the GPU: do not start a game.'
            Write-Host "After $waitText these Blink processes still hold it:"
            $still | ForEach-Object { Write-Host "  $_" }
            Write-Host 'They do not stop for a pause: a calibration, a test or a benchmark runs to its end, and'
            Write-Host 'supervised training would have let go by now. Blink stays paused. Run Pause Blink again'
            Write-Host 'later, and start a game only once it reports the GPU free.'
            exit 1
        }
        Start-Sleep -Seconds 5
    }
} catch {
    Write-Host "The GPU could not be checked: $($_.Exception.Message)"
    Write-Host 'Supervised Blink training still pauses: it lets go of the GPU at its next step, usually within a minute.'
    exit 2
}
