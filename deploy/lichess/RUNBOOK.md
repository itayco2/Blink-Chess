# Lichess bot runbook

Every account step, every token step and every bot start on this page is done by Itay. Steps marked
**(agent)** are `blink lichess` commands the build agent runs: it generates the config, reads the
public Lichess API without a token, and may stop the bot process by its PID. It never creates
accounts, never handles the token, never reads `D:\blink-bot\token.dpapi` or the bot's environment,
and never starts the bot.

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

**(agent)** checks the upgrade on the public profile only (no token):

```powershell
uv run blink lichess snapshot --bot <BotName> --no-write
```

It must print `(BOT)` and `blitz 3000, RD 500, N 0`.

## 5. Generate the configs (agent)

The two configs are generated from the tracked templates, never edited by hand, and never hold the
token (lichess-bot reads it from `LICHESS_BOT_TOKEN`, which only `start-bot.ps1` sets):

| file | template | what it is |
|---|---|---|
| `D:\blink-bot\config.yml` | [config.template.yml](config.template.yml) | the rated bot (G7): torch CUDA fp32, the shipped model, sha and mode |
| `D:\blink-bot\config.casual.yml` | [config.casual.yml](config.casual.yml) | the G5 casual smoke: the preview model on CPU, only `itayco2` |

Both switch off every lookup lichess-bot could make for the engine (polyglot book, every
`online_moves` source, `lichess_bot_tbs`, the tablebase resign and draw options, pondering), remove
the default `uci_options` (blink-uci declares none), and set `abort_time: 30` explicitly (the code
default for a missing key is 20).

lichess-bot starts one `blink-uci` per game with the same flags, and the rated bot can run two games
at once, so the decision log path holds `{process}`: each engine writes its own file,
`D:\blink\lichess\decisions\<UTC start>-<PID>.jsonl` (casual: `D:\blink\lichess\decisions-casual\`),
one per game. `check-config` refuses a single shared log file whenever `challenge.concurrency` is
above 1.

```powershell
# G5, during P7: the casual config only, with the preview model
uv run blink lichess config --only casual --model <preview selector> --mode <mode>
uv run blink lichess check-config --config D:\blink-bot\config.casual.yml

# G7, after E8: the rated config, pinned to the shipped weights (sha256 hashed from the file)
uv run blink lichess config --only rated --model ship --mode <shipped mode>
uv run blink lichess check-config --config D:\blink-bot\config.yml
```

`check-config` exits 0 only with `0 problems`. For the rated config it also compares the recorded
sha256 and mode with `shipped` in `results/results.json` and with the weights file itself, so a
config that points at anything but the evaluated model fails. `--sha <sha>` makes the generator
refuse a weights file with a different hash.

## 6. Casual smoke (gate G5, during P7, CPU)

1. **(agent)** generates and checks `config.casual.yml` (section 5).
2. Itay: `D:\blink-bot\start-bot.ps1 -Config D:\blink-bot\config.casual.yml`
3. Itay challenges the bot from `itayco2` to 5 casual games: 1+1, 3+2 and 10+0 (about 40 minutes).
4. **(agent)** `uv run blink lichess check --bot <BotName> --window 5 --pgn-dir D:\blink\lichess\pgn-casual`
5. Pass: 5/5 games completed, 0 aborts, 0 time losses, 0 illegal moves. Then Itay stops the bot
   with Ctrl+C.

## 7. Rated launch (gate G7, after E8 and after training)

1. **(agent)** generates and checks `config.yml` (section 5): rated blitz only; matchmaking bases
   [180, 300] and increments [0, 2, 3], `challenge_timeout: 2`, `opponent_rating_difference: 300`,
   `challenge_filter: fine`; under `challenge:` `concurrency: 2` with `games_reserved_for_humans: 1`
   (so one bot game at a time), `preference: human`, `bullet_requires_increment: true`,
   `max_simultaneous_games_per_user: 1`; resign and draw offers off; PGNs in `D:\blink\lichess\pgn`.
2. Itay: `D:\blink-bot\start-bot.ps1`
3. Optional auto-start: a Task Scheduler task that runs `start-bot.ps1` "only when user is logged
   on", with an at-logon trigger only and no restart on failure, so a pause is never undone. Task
   Scheduler stops a task after 72 hours by default, which would end the bot inside the 7-day
   window, so the time limit is switched off:

   ```powershell
   $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -File D:\blink-bot\start-bot.ps1'
   $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
   $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
   $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero)
   Register-ScheduledTask -TaskName 'BlinkBot' -Action $action -Trigger $trigger -Principal $principal -Settings $settings
   ```

## 8. Watching it and the stop rule (agent)

```powershell
uv run blink lichess check --bot <BotName> --stop     # exits 1 when the rule fires
uv run blink lichess snapshot --bot <BotName> --no-write
```

- The stop rule: **time losses > 2% or aborts > 1% over the last 50 games** (any kind; fewer games
  count as they are). With `--stop` a firing rule pauses the bot exactly as section 9 does and
  records the rule in `D:\blink\lichess\pause.json`; Itay decides when it restarts.
- Lichess's public game export never contains aborted games (lila exports only finished games), so
  `check` also counts the `Termination "Abandoned"` PGNs that lichess-bot saves in
  `D:\blink\lichess\pgn`. The public snapshot's abort rate sees only games that never started
  (`noStart`).
- `snapshot --no-write` prints rating, RD, N, the human share, performance vs humans and vs bots,
  and the time-loss, abort and duplicate-game rates. It writes nothing and commits nothing.

## 9. Pause and restart (GPU windows, stop rule)

During rating accrual, GPU jobs run only while the bot is paused (the D15 window: export,
quantize, film extract).

```powershell
uv run blink lichess pause --bot <BotName> --reason "D15 GPU window"   # (agent)
```

1. Writes `D:\blink\lichess\PAUSED`, so `start-bot.ps1` and the Task Scheduler task will not start
   the bot.
2. Polls the public status every 15 s until the bot is in no game (at most `--timeout`, 30 minutes
   by default; after that the game in progress is cut off and the record says so).
3. Stops the lichess-bot process found by its command line under `D:\blink-bot`, and every process
   below it (game workers, blink-uci engines), by PID.
4. Records the time, reason, wait and PIDs in `D:\blink\lichess\pause.json`.

To restart:

```powershell
uv run blink lichess resume-note     # (agent) only deletes the PAUSED flag
D:\blink-bot\start-bot.ps1           # Itay
```

## 10. Publishing the rating (gate G12)

```powershell
uv run blink lichess snapshot --bot <BotName>     # (agent) writes results/lichess.json with its date
```

The rating is publishable only at N >= 200 rated blitz games and RD < 75; until then the file says
`"publishable": false` and the page shows "rating accruing". No workflow writes this file.

## 11. Staying online

- After the LinkedIn post (G13), keep the bot online for **at least 7 days**: sleep never on AC, the
  at-logon task from section 7, and no GPU pause in those 7 days unless the stop rule fires.
- After the 7 days, run it on the hours listed in its bio.
