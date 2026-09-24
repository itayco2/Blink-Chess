# PREFLIGHT

Every defect found while building Blink, with the number that exposed it and the number after the fix.
A row is closed only when its number after is measured. Tests that lock a row name it in their docstring.

| # | what was wrong | the number that showed it | fix | the number after |
|---|---|---|---|---|
| PF01 | C: is the fast drive but nearly full | 25 GB free of 464 GB | repo and venv on C:; all data, caches and runs under `BLINK_HOME=D:\blink`; uv cache on D: | pending: C: free after the torch install |
| PF02 | free VRAM is not a constant: desktop apps hold part of the 8 GB | 6.3 GB free on one read, about 4.9 GB on another | doctor measures `torch.cuda.mem_get_info()` and sizes batches from it | pending: doctor on the build machine |
| PF03 | the default PyPI torch wheel for Windows is CPU-only | a plain install gives `+cpu` | explicit cu130 index for win32 in `pyproject.toml` | `uv.lock` pins `download.pytorch.org/whl/cu130` (3 entries); pending: doctor reports `+cu130` |
| PF04 | torch.compile on Windows needs the community triton-windows fork, paired by minor version | torch 2.14 pairs only with Triton 3.8 | pin `triton-windows>=3.8,<3.9`; compile stays optional | pending: triton smoke test |
| PF05 | torch 2.14 on Windows builds no flash attention kernel | `USE_FLASH_ATTENTION` excludes MSVC at v2.14.0 | use the memory-efficient SDPA kernel in bf16 | pending: doctor on the build machine |
| PF38 | a detached job must outlive the agent session that launched it | not yet measured | heartbeat probe launched with `Start-Process`, checked from a later session | pending |
| PF39 | a redirected stdout on Windows uses the ANSI code page, so one Hebrew or sigma character kills a run | the unguarded print exits non-zero (`test_without_the_reconfigure_a_redirected_print_crashes_on_windows`) | `configure_stdio()` forces UTF-8 on stdout and stderr | exit 0 and the exact UTF-8 text (`test_cli_survives_non_ascii_output_when_redirected`) |
| PF40 | the 22.09 GB eval DB over one connection | 1.1 MB/s, 334 min estimated | 4 parallel range workers (each connection is capped at about 1.2 MB/s) | pending: measured aggregate rate |
| PF41 | Lichess answers 429 above about 5 connections, and a naive range appender wrote the 429 HTML page into the data | 2 of 6 segments held a 162-byte nginx page | 64 MiB chunk files, each kept only on HTTP 206 with the pinned ETag and the exact byte count | pending: sha256 of the assembled file |
| PF42 | stopping a background shell left its subshells alive; they restarted curl and kept appending to renamed files | 4 pieces grew by 36.5 to 44.0 MB after "stop" | kill the whole process tree by PID; trim each piece to its planned offset; check that pieces tile the file exactly | pending: tiling check and sha256 |
| PF43 | with torch in a uv extra, any plain `uv run` would uninstall it (uv syncs exactly) | found in review before it happened | torch and triton moved to default dependency groups; the torch-free CI leg opts out | `uv run` keeps torch |
| PF44 | escape sequences written into the punctuation test became the very characters it bans | the test flagged its own file (7 characters) | build the banned characters from code points with `chr()` | the test passes on every tracked file |
