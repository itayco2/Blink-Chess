@echo off
rem Resume Blink: removes BLINK_HOME\PAUSE (D:\blink\PAUSE unless BLINK_HOME says otherwise), which
rem "Pause Blink.cmd" created. Blink training starts again from its checkpoint within a minute.
rem First it checks the GPU: each paused run whose supervisor still waits says in its heartbeat how much
rem free GPU memory its restart needs (resume_needs_gpu_gb). While nvidia-smi shows less than that free
rem (a game still open, or only minimized), the flag stays and Resume says to close the game first.
rem It never lifts a P7-VAA pause: that gate is cleared by hand.
rem The GPU check is PowerShell, kept at the end of this file: cmd.exe stops at "exit /b" and never
rem reads it, and PowerShell runs only what follows its marker line.
setlocal
if not defined BLINK_HOME set "BLINK_HOME=D:\blink"
set "BLINK_RESUME_SCRIPT=%~f0"
set "code=0"
if not exist "%BLINK_HOME%\PAUSE" (
    echo Blink was not paused, so there is nothing to resume.
    goto done
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$text = [IO.File]::ReadAllText($env:BLINK_RESUME_SCRIPT); Invoke-Expression $text.Substring($text.LastIndexOf('#BEGIN' + ' POWERSHELL'))"
if errorlevel 1 if not errorlevel 2 (
    set "code=1"
    goto done
)
del /f /q "%BLINK_HOME%\PAUSE"
if exist "%BLINK_HOME%\PAUSE" (
    set "code=2"
    echo Could not remove %BLINK_HOME%\PAUSE: close anything that has it open, then try again.
    goto done
)
echo Blink will resume within a minute.
:done
echo.
pause
exit /b %code%

#BEGIN POWERSHELL: the GPU check (Resume Blink.cmd runs this part; see the top of the file)
# exit 1 keeps the flag: a paused run needs more free GPU memory than there is. exit 0 (or 2, when the
# GPU cannot be read) removes it; the supervisor still holds a restart the GPU has no room for.
$freshSeconds = 60  # a paused heartbeat older than this has no supervisor waiting behind it
$invariant = [Globalization.CultureInfo]::InvariantCulture
$now = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
$runs = Join-Path $env:BLINK_HOME 'runs'
$waiting = @(Get-ChildItem -LiteralPath $runs -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        $beat = Get-Content -LiteralPath (Join-Path $_.FullName 'heartbeat.json') -Raw -ErrorAction Stop | ConvertFrom-Json
    } catch {
        return
    }
    if ($beat.state -eq 'paused: user' -and $null -ne $beat.resume_needs_gpu_gb -and ($now - [double]$beat.time) -le $freshSeconds) {
        [pscustomobject]@{ Run = $_.Name; Need = [double]$beat.resume_needs_gpu_gb }
    }
})
if ($waiting.Count -eq 0) { exit 0 }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host 'nvidia-smi was not found, so the GPU was not checked.'
    exit 2
}
try {
    $line = @(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)[0]
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi exited with code $LASTEXITCODE" }
    $free = [double]::Parse("$line".Trim(), $invariant) / 1024
} catch {
    Write-Host "The GPU could not be checked: $($_.Exception.Message)"
    exit 2
}
$short = @($waiting | Where-Object { $_.Need -gt $free })
if ($short.Count -eq 0) { exit 0 }
Write-Host ([string]::Format($invariant, 'Close the game first: the GPU has {0:0.00} GB free, and Blink needs more to resume.', $free))
$short | ForEach-Object { Write-Host ([string]::Format($invariant, '  {0} needs {1:0.00} GB free', $_.Run, $_.Need)) }
Write-Host 'Blink stays paused. Quit the game (minimized, it keeps its memory), then run Resume Blink again.'
exit 1
