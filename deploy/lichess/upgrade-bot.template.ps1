# upgrade-bot.ps1 (copy this template to D:\blink-bot\upgrade-bot.ps1).
# Turns a fresh Lichess account into a BOT account. This is IRREVERSIBLE.
# Run once, by Itay, only on an account that has played zero games.
# It calls POST /api/bot/account/upgrade directly, because `lichess-bot.py -u` also starts playing.
#
# On failure this script prints only the HTTP status. Never dump the raw error object
# ($Error[0], Format-List *, -Verbose): it keeps the request, including the Authorization header.
# If the upgrade fails, revoke the token on lichess.org and create a fresh one instead of debugging it.
$ErrorActionPreference = 'Stop'

$answer = Read-Host 'This permanently turns the account into a BOT. Type UPGRADE to continue'
if ($answer -ne 'UPGRADE') {
    Write-Host 'Cancelled.'
    exit 1
}

try {
    $secure = Get-Content 'D:\blink-bot\token.dpapi' | ConvertTo-SecureString
    $token = [System.Net.NetworkCredential]::new('', $secure).Password
    Invoke-RestMethod -Method Post -Uri 'https://lichess.org/api/bot/account/upgrade' `
        -Headers @{ Authorization = "Bearer $token" } | Out-Null
    Write-Host 'Upgraded. The account page should now show the BOT title.'
}
catch {
    $status = $null
    if ($_.Exception.Response) { $status = [int]$_.Exception.Response.StatusCode }
    Write-Host "Upgrade failed (HTTP status: $status). Revoke this token, create a new one, and retry."
    $Error.Clear()
    exit 1
}
finally {
    Remove-Variable token, secure -ErrorAction SilentlyContinue
}
