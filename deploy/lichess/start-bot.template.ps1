# start-bot.ps1 (copy this template to D:\blink-bot\start-bot.ps1).
# Itay runs it himself. The build agent never runs it and never reads token.dpapi.
# The token is decrypted into this process and handed to the lichess-bot child process, which holds it
# in its environment for as long as the bot runs, as does every process lichess-bot starts (its game
# workers, and each game's blink-uci until it deletes the variable at startup). See RUNBOOK,
# "What protects the token".
#
# The PAUSED flag only blocks new starts (including a Task Scheduler start). It does not stop a bot
# that is already running; to pause a live bot, its process is stopped by PID and the flag keeps it down.
param([string]$Config = 'D:\blink-bot\config.yml')
$ErrorActionPreference = 'Stop'

$pauseFlag = 'D:\blink\lichess\PAUSED'
if (Test-Path $pauseFlag) {
    Write-Host "Blink bot is paused ($pauseFlag exists). Delete the flag to start it."
    exit 0
}

try {
    $secure = Get-Content 'D:\blink-bot\token.dpapi' | ConvertTo-SecureString
    $env:LICHESS_BOT_TOKEN = [System.Net.NetworkCredential]::new('', $secure).Password
    Set-Location 'D:\blink-bot\lichess-bot'
    & 'D:\blink-bot\venv\Scripts\python.exe' 'lichess-bot.py' --config $Config
}
finally {
    Remove-Item Env:\LICHESS_BOT_TOKEN -ErrorAction SilentlyContinue
    Remove-Variable secure -ErrorAction SilentlyContinue
}
