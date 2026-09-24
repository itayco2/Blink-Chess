# watch-bot.ps1 (copy this template to D:\blink-bot\watch-bot.ps1).
# Runs `blink lichess watch` from the frozen engine install (RUNBOOK section 7): the stop rule every
# 2 minutes, with no agent session needed.
# Its own Task Scheduler task starts it at logon (RUNBOOK section 8). It never reads token.dpapi and
# must never be started from start-bot.ps1, which holds the token in its environment. Its only action
# on the bot is the pause: stop the lichess-bot process by PID and leave the PAUSED flag.
param([Parameter(Mandatory = $true)][string]$Bot)
$ErrorActionPreference = 'Stop'

if ($Bot -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{1,29}$') {
    Write-Host "Not a Lichess name: $Bot"
    exit 2
}
$env:PYTHONUTF8 = '1'  # this process and the watcher only
New-Item -ItemType Directory -Force 'D:\blink\logs' | Out-Null
cmd.exe /d /c "D:\blink-bot\engine\Scripts\blink.exe lichess watch --bot $Bot >> D:\blink\logs\lichess-watch.log 2>&1"
exit $LASTEXITCODE
