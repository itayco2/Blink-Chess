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
- While the bot runs, the token sits in plain text in the environment of `start-bot.ps1`'s
  PowerShell, of the lichess-bot process and of every process lichess-bot starts: its game workers
  and each game's `blink-uci` engine all inherit it (lichess-bot starts the engine with no
  environment of its own). Any program running as you could read it there; `Remove-Item Env:` only
  cleans up after the bot exits. `blink-uci` deletes the variable from its own environment at
  startup, without reading it, so the engine and anything it starts no longer carry it; the
  lichess-bot processes still do.
- A crash dump of the bot, of PowerShell or of an engine would contain the token. Treat any
  `%LOCALAPPDATA%\CrashDumps\python.exe*.dmp` or `blink-uci.exe*.dmp` from a bot run as holding it,
  and delete it.
- **The code that runs with the token is writable by you, and so by the agent.** The engine
  (`D:\blink-bot\engine`, built by the agent), lichess-bot (`D:\blink-bot\lichess-bot` and its venv)
  and `D:\blink-bot\config.yml` (written by `blink lichess config`) all belong to your account.
  Whatever changes them runs next time with the token in its environment. `check-config` catches a
  config that sends the token elsewhere or starts another program (section 5), but not changed code.
- **Why this is acceptable:** the token has only the `bot:play` scope, on an account that is only a bot.
  The worst a leak can do is play games as the bot. You can revoke it in one click on
  lichess.org, and a new one takes a minute.
- **Optional hardening, and what it takes:** a separate, standard (non-admin) Windows account makes
  DPAPI a real boundary only if that account also owns everything that runs with the token and your
  account cannot write to any of it: the frozen engine install, lichess-bot and its venv, the
  scripts and `config.yml`. The agent then writes configs to a staging folder, and the bot account
  copies one in after `check-config` passes. A separate account that still runs code or a config
  your account can change protects nothing, because that code would read the token for it.
- **Also optional:** restrict the folder to your account (PowerShell):
  `icacls D:\blink-bot /inheritance:r /grant:r "$($env:USERNAME):(OI)(CI)F"`. This keeps other
  Windows users out; it does not keep out programs running as you.

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
| `D:\blink-bot\config.yml` | [config.template.yml](config.template.yml) | the rated bot (G7): torch CUDA in the shipped fast mode (fp32 uncompiled by default), the shipped model, sha and mode, run from the frozen engine install `D:\blink-bot\engine` (section 7) |
| `D:\blink-bot\config.casual.yml` | [config.casual.yml](config.casual.yml) | the G5 casual smoke: the preview model on CPU (1 thread, below-normal priority), only `itayco2`, run from the dev venv `C:\dev\blink-chess\.venv` |

Both switch off every lookup lichess-bot could make for the engine (polyglot book, every
`online_moves` source, `lichess_bot_tbs`, the tablebase resign and draw options, pondering), remove
the default `uci_options` (blink-uci declares none), and set `abort_time: 30` explicitly (the code
default for a missing key is 20).

lichess-bot sends the token to `url` and starts `interpreter interpreter_options dir/name` in
`working_dir` with the token in its environment, so `check-config` pins those too: `url` is
`https://lichess.org/`, `engine.dir` is the engine's install folder, `working_dir` is empty,
`interpreter`, `interpreter_options` and `matchmaking.overrides` are absent or empty, and no engine
key outside the template's is allowed.

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

`check-config` exits 0 only with `0 problems`, and it fails closed: anything it cannot verify is a
problem, not a note. The engine exe must exist. For the rated config the weights file must be found
and hash to the recorded sha256 (all 64 hex digits), and `results/results.json` (run it from the
repo root) must exist and name the shipped model, whose sha and mode the config must match; so a
config that points at anything but the evaluated model fails. `--results none` skips only the
results.json comparison, on purpose, and says so. `--sha <sha>` makes the generator refuse a weights
file with a different hash.

In value mode the rated config also plays the R4 tie window E2b chose: `blink lichess config` copies
`results/epsilon.json` (run it from the repo root, or pass `--results-dir`) into `engine_options.epsilon`,
because every rated game behind the published Elo was played with it and blink-uci's own default is 0.
It refuses a value-mode rated config when that file is missing, and `check-config` compares the
epsilon with the one results.json records for the shipped model.

The rated config plays the fast play mode the shipped model was rated in, the same way. When the
evaluation ran with `blink eval all --precision bf16 [--compile]`, every rated game was played in that
mode (under a name ending `-bf16` or `-bf16-compile`), and results.json's shipped record says so
(`shipped.precision`, `shipped.compile`). `blink lichess config` copies them into
`engine_options.precision` and `engine_options.compile` (lichess-bot passes `--precision=bf16
--compile=True` to blink-uci) and into the `blink:` stamp. The default mode, fp32 uncompiled, adds no
key, so a default config is unchanged. `check-config` fails a rated engine whose mode differs from its
stamp or from the shipped record, any engine that asks for bf16 off CUDA (blink-uci would exit 2), and a
compiled CUDA engine whose own install has no triton (section 7: blink-uci would fail at its first
`isready`).

The rated config also passes the sha to every engine (`engine_options.sha`, which lichess-bot hands
to blink-uci as `--sha=<sha>`). Each blink-uci hashes its weights file before the UCI handshake and
exits 2 on any other hash, so an overwritten `ship` file stops the bot at lichess-bot's startup
engine check instead of putting an unevaluated model on the rated account.

## 6. Casual smoke (gate G5, during P7, CPU)

The smoke runs while the long training run is live, so its engine keeps to the plan's P7 budget
for side processes: `blink-uci --threads=1 --priority=below_normal` (torch on one thread, the process
at BELOW_NORMAL priority), set in `config.casual.yml` and required by `check-config`.

1. **(agent)** generates and checks `config.casual.yml` (section 5).
2. Itay: `D:\blink-bot\start-bot.ps1 -Config D:\blink-bot\config.casual.yml`
3. Itay challenges the bot from `itayco2` to 5 casual games: 1+1, 3+2 and 10+0 (about 40 minutes).
4. **(agent)** `uv run blink lichess check --bot <BotName> --window 5 --pgn-dir D:\blink\lichess\pgn-casual`
5. Pass: 5/5 games completed, 0 aborts, 0 time losses, 0 illegal moves. Then Itay stops the bot
   with Ctrl+C.

## 7. Rated launch (gate G7, after E8 and after training)

**Deviation from the plan (P9 setup):** the plan puts the bot's engine in
`C:\dev\blink-chess\.venv\Scripts`. That venv is an editable install of the dev checkout, and every
`uv run` re-syncs it, so the P10-P12 merges and branch checkouts during rating accrual would change
or break the engine that lichess-bot starts for every game. The rated bot therefore runs from a
separate, non-editable install of the shipped tag, `D:\blink-bot\engine`, and `check-config` pins
that folder; the G5 casual smoke keeps the plan's folder. Blink is still evaluated exactly as it
ships: the same tag, sha, runtime, mode and rules.

1. **(agent)** freezes the engine from the shipped tag (the uv cache already holds torch, so nothing
   new is downloaded; set the variable in this shell only, never with `setx`). When results.json's
   shipped record has `compile: true`, the sync also takes `--group compile`. The compiled engine
   needs triton-windows, torch.compile's CUDA backend, and torch's Windows wheel does not include it.
   The dev venv syncs that group by default, so the uv cache already holds it.

   ```powershell
   git -C C:\dev\blink-chess worktree add C:\dev\blink-wt\ship-<tag> <tag>
   $env:UV_PROJECT_ENVIRONMENT = 'D:\blink-bot\engine'
   uv sync --project C:\dev\blink-wt\ship-<tag> --frozen --no-editable --no-default-groups --group train   # add --group compile when shipped.compile is true
   Remove-Item Env:\UV_PROJECT_ENVIRONMENT
   git -C C:\dev\blink-chess worktree remove C:\dev\blink-wt\ship-<tag>
   'uci', 'quit' | D:\blink-bot\engine\Scripts\blink-uci.exe --model ship --sha <shipped sha256>   # uciok, exit 0
   ```

   This check proves only that the engine starts and accepts the weights: `uci` never loads the
   model. The cold-start gate follows in step 2.
2. **(agent)** generates and checks `config.yml` (section 5): rated blitz only; matchmaking bases
   [180, 300] and increments [0, 2, 3], `challenge_timeout: 2`, `opponent_rating_difference: 300`,
   `challenge_filter: fine`; under `challenge:` `concurrency: 2` with `games_reserved_for_humans: 1`
   (so one bot game at a time), `preference: human`, `bullet_requires_increment: true`,
   `max_simultaneous_games_per_user: 1`; resign and draw offers off; PGNs in `D:\blink\lichess\pgn`.

   Then the agent measures the cold-start gate: cold start plus the first move, under 10 s. It runs
   the engine the way lichess-bot starts it, with every `engine_options` entry of `config.yml` except
   `log`. That means `--sha` (every engine hashes its weights first, at about 300 MB/s on this PC
   under load), `--epsilon` in value mode, and `--precision` and `--compile` when the shipped record
   has them. The model loads and warms up at the first `isready`, and a compiled engine compiles
   there too. lichess-bot starts one engine per game, so every game start pays this cost:

   ```powershell
   $flags = '--model=ship', '--mode=<mode>', '--device=cuda', '--sha=<sha256>'   # plus '--epsilon=<e>', '--precision=bf16', '--compile=True' as config.yml has them
   Measure-Command { 'uci', 'isready', 'position startpos', 'go movetime 1000', 'quit' | D:\blink-bot\engine\Scripts\blink-uci.exe @flags | Out-Host }
   ```

   Pass: `uciok`, `readyok` and one `bestmove` are printed, and `TotalSeconds` is under 10.
3. Itay registers and starts the watcher task (section 8) before the bot's first rated game; the
   agent confirms its heartbeat in `D:\blink\lichess\watch.json`.
4. Itay: `D:\blink-bot\start-bot.ps1`
5. Optional auto-start: a Task Scheduler task that runs `start-bot.ps1` "only when user is logged
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

## 8. The stop rule, enforced by the watcher

The stop rule: **time losses > 2% or aborts by the bot > 1% over the last 50 games** (any kind;
fewer games count as they are). An abort counts against the bot only when the bot was the side to
move. lichess-bot aborts a game itself when the opponent never moves, and lila records that as an
abort too; those opponent no-shows are reported beside the rule and never counted.

The bot runs for days and comes back at every logon, so the rule cannot depend on an agent session
being alive (plan section 4: every long job's stop rules are enforced without an agent).
`blink lichess watch` enforces it from a Task Scheduler task of its own:

1. Copy [watch-bot.template.ps1](watch-bot.template.ps1) to `D:\blink-bot\watch-bot.ps1`. It runs
   the watcher from the frozen engine install (section 7), so dev work cannot change it either. It
   never reads the token and is never started from `start-bot.ps1`.
2. Itay registers the task once, at G7, beside the bot's. Unlike the bot's task it restarts on
   failure, because the watcher can only pause the bot, never start it:

   ```powershell
   $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -WindowStyle Hidden -File D:\blink-bot\watch-bot.ps1 -Bot <BotName>'
   $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
   $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
   $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
   Register-ScheduledTask -TaskName 'BlinkBotWatch' -Action $action -Trigger $trigger -Principal $principal -Settings $settings
   Start-ScheduledTask -TaskName 'BlinkBotWatch'
   ```

3. Every 2 minutes the watcher reads the PGNs lichess-bot saves in `D:\blink\lichess\pgn` and the
   public export (no token), and writes its heartbeat to `D:\blink\lichess\watch.json` (time, round,
   verdict, action); its output goes to `D:\blink\logs\lichess-watch.log`.
4. When the rule fires on a game no earlier stop acted on, it pauses the bot at once, as section 9
   does but **without waiting for the live game** (`--timeout 0`): while a pause waits, a broken
   engine loses that game anyway and lichess-bot keeps accepting new ones. The games it acted on go
   to `D:\blink\lichess\stop-rule.json`, so the bot Itay restarts is not stopped again for the same
   games while they are among the last 50; any new time loss or abort by the bot stops it again.
5. While `D:\blink\lichess\PAUSED` exists the watcher only writes its heartbeat. If the public API
   fails, it judges the local PGNs alone and says so in the heartbeat.
6. **(agent)** every status poll reads `watch.json`. A heartbeat older than 5 minutes (while the PC is
   on) means the watcher is down; the agent records it in STATUS and asks Itay to start the task.

How fast: lichess-bot writes a game's PGN when the game ends, or, for a game whose engine failed,
when its retries give up (at most 10 minutes). So a crashing engine is paused within about
12 minutes of its first abort, and one abort by the bot is enough to fire the rule.

The same rule on demand **(agent)**:

```powershell
uv run blink lichess check --bot <BotName>            # exits 1 when the rule fires
uv run blink lichess check --bot <BotName> --stop     # also pauses at once, only for new games
uv run blink lichess snapshot --bot <BotName> --no-write
```

- Lichess's public game export never contains aborted games (lila exports only finished games), so
  `check` and `watch` also read every game in lichess-bot's PGNs (`Termination "Abandoned"` for
  aborts, `"Time forfeit"` with the Result tag for time losses). The public snapshot's abort rate
  sees only games that never started (`noStart`).
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
  at-logon tasks from sections 7 and 8, and no GPU pause in those 7 days unless the stop rule fires.
- After the 7 days, run it on the hours listed in its bio.
