# Lichess bot runbook

Every step on this page is done by Itay. The build agent prepares the config, watches the public
Lichess API and may stop the bot process by its PID. It never creates accounts, never handles the
token, never reads `D:\blink-bot\token.dpapi`, and never starts the bot.

## 1. Create the account (any time before the bot is needed)

1. Sign up for a fresh Lichess account. Candidate names that were free on 2026-09-24:
   `BlinkBot`, `Blink_BOT`, `OneLookBot`.
2. **Play zero games on it.** Any game, even a casual one, blocks the upgrade for good.

## 2. Create the token and store it encrypted

1. Create a personal API token with only the `bot:play` scope:
   `https://lichess.org/account/oauth/token/create?scopes[]=bot:play&description=Blink`
2. Store it with Windows DPAPI, so only your Windows user can decrypt it:

   ```powershell
   New-Item -ItemType Directory -Force D:\blink-bot | Out-Null
   Read-Host -AsSecureString 'Lichess bot token' | ConvertFrom-SecureString | Set-Content D:\blink-bot\token.dpapi
   ```

The token never goes into `config.yml`, a repo file, a user environment variable or a chat.

## What protects the token, honestly

- **DPAPI protects the file from other Windows users and from a copied disk. It does not protect it
  from programs running as you.** The build agent runs as your Windows user, so "the agent never reads
  the token" is a rule the agent follows, not a wall it cannot cross.
- While the bot runs, the token sits in plain text in the bot process's environment. Any program
  running as you could read it; `Remove-Item Env:` only cleans up after the bot exits.
- A crash dump of the bot or of PowerShell would contain the token. Treat any
  `%LOCALAPPDATA%\CrashDumps\python.exe*.dmp` from a bot run as holding it, and delete it.
- **Why this is acceptable:** the token has only the `bot:play` scope, on an account that is only a bot.
  The worst a leak can do is play games as the bot. You can revoke it in one click on
  lichess.org, and a new one takes a minute.
- **Optional hardening:** run the bot under a separate, standard (non-admin) Windows account. DPAPI
  then becomes a real boundary, because a program running as you cannot decrypt that account's file or
  read its processes' memory without admin rights.
- **Also optional:** restrict the folder to your account (PowerShell):
  `icacls D:\blink-bot /inheritance:r /grant:r "$($env:USERNAME):(OI)(CI)F"`.

## 3. Start-bot script

Copy [start-bot.template.ps1](start-bot.template.ps1) to `D:\blink-bot\start-bot.ps1`. It:

- exits without starting if `D:\blink\lichess\PAUSED` exists (the agent writes that flag before a GPU
  job and when a stop rule fires; delete it when you restart the bot);
- decrypts the token into its own process only;
- starts lichess-bot with `D:\blink-bot\config.yml` (or `-Config <path>` for the casual config).

## 4. Upgrade to a BOT account (irreversible, once)

Copy [upgrade-bot.template.ps1](upgrade-bot.template.ps1) to `D:\blink-bot\upgrade-bot.ps1` and run it.
It calls `POST https://lichess.org/api/bot/account/upgrade` directly. Do not use
`lichess-bot.py -u`: it upgrades and then immediately starts accepting challenges.

After the upgrade, a new BOT starts at rating 3000 with RD 500 in the standard pools, so its rating
will fall steeply over its first games. That is expected.

## 5. Running it

- Casual smoke test (gate G5): `D:\blink-bot\start-bot.ps1 -Config D:\blink-bot\config.casual.yml`,
  then challenge the bot from your own account to 5 casual games (1+1, 3+2, 10+0).
- Rated launch (gate G7): `D:\blink-bot\start-bot.ps1`.
- Optional auto-start (gate G7): a Task Scheduler task that runs `start-bot.ps1` "only when user is
  logged on", with an at-logon trigger only and no restart on failure, so a pause is never undone.
- After the launch post, keep the bot online for at least 7 days, then on the hours listed in its bio.
