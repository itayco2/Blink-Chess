# EVAL (v0, pre-registered)

This file fixes how Blink will be measured, before any real training run. Version 0 is committed in P1.
Version 1 is frozen when the flagship run launches and tagged `eval-v1-frozen`; after that, only the
rules below decide what ships and what is published. Results never edit this file.

## 1. What "no search" means (NSC-1)

At each decision Blink has the current position P, the real game history H, the weights and the clock.

- **N1, budget.** At most one network call per decision, on one batch of at most L(P)+1 rows, drawn only
  from P and the positions after each legal move of P, with no position repeated.
  One look evaluates only P. One look per move evaluates P plus every child.
- **N2, depth.** No position two or more plies beyond P is ever network input, scored or compared.
  A rule query on a child may enumerate that child's legal replies as a rules check; those replies are
  never evaluated.
- **N3, no outside knowledge.** No engine process, opening book, tablebase, cloud evaluation, opening
  explorer, position cache or network access at play time; the only file read is the weights at start.
- **N4, no iteration.** Compute never grows with thinking time: no rollouts, deepening, sampling, voting,
  ensembles, test-time augmentation, pondering or randomisation. Nothing carries between moves except H.

**Allowed rule checks (closed list).**

| id | rule |
|---|---|
| R1 | legal-move generation and masking the policy to legal moves |
| R2 | mate now: if a child is checkmate, play it (fires before the network call, so 0 calls) |
| R3 | rule draws (stalemate, insufficient material, 50-move rule, third repetition from Blink's own history counter): value mode scores them 0.5; policy mode demotes them when its root value is above 0.5 + 0.10 and prefers the best of them when below 0.5 - 0.10 |
| R4 | value-mode tie-break: among children within epsilon of the best win%, the highest root policy logit |
| R5 | clock guard: under low time, switch to one look (only ever fewer evaluations) |

**Proof.** Every decision is counted. The engine reports the rows it evaluated as UCI nodes, match PGNs
keep that count per move, and `blink audit no-search` rebuilds the histogram from the PGNs alone.
Published invariant: at most 1 call and at most L+1 rows on 100% of decisions; 0 calls only when R2 fired.

## 2. How the shipped mode is chosen

- SPRT of value mode against policy mode for the chosen size: elo0 = 0, elo1 = 20, alpha = beta = 0.05,
  logistic model, development opening slice, cap 6,000 games.
- H1 accepted: value mode ships. H0 accepted: the reverse SPRT runs with the same bounds; H1 there means
  policy ships; H0 again means "statistically tied" and policy ships, as the stricter claim.
- Cap reached without a decision: the higher point estimate ships, and its interval is published.
- The shipped weights are the final EMA weights, unless the raw weights beat them by more than 2 sigma on
  the value-agreement metric. Checkpoints are never chosen on test puzzles or final-slice games.

## 3. What is measured, and on what

| number | set | method |
|---|---|---|
| Elo against Stockfish 19 anchors | final opening slice (8moves_v3 openings 10,001-34,700) | Ordo with fixed UCI_Elo anchors, 95% interval; stated as CCRL-Blitz-anchored engine Elo, not FIDE |
| Stockfish node crossover | final slice | node ladder 2^k, k = 4, 6, ..., 16 |
| DeepMind 10K puzzles | puzzles.csv | DeepMind's own scorer (whole solution line), Wilson 95% interval, by rating band; clean subset reported separately |
| Lichess BOT blitz rating | live games | published only at N >= 200 rated games and RD < 75, with the date |
| static metrics | held-out test positions and 10K game positions labelled by Stockfish | policy top-1/3/5, value-mode agreement, Brier, calibration error, Kendall tau on scores |

Match rules: per-engine time controls (Blink `st=1 timemargin=500`, Stockfish `st=0.1 timemargin=100`),
no resignation, no win adjudication, draws by rule plus one fastchess draw adjudication at 600 engine
plies (`-maxmoves 300`). Blink time forfeits must be 0.

Games fastchess does not run (every block except Stockfish's self-check, the anchor gauntlets and
DeepMind 9M's anchor gauntlet) are played one at a time in the harness process under the same clocks:
a Blink or DeepMind 9M move over 1.5 s, or a Stockfish move over 0.2 s at `st=0.1`, loses on time, and
Stockfish at a fixed node count has no clock. There the time is measured around the move choice rather
than over UCI, and each clocked player first makes one untimed warm-up move, as fastchess's `isready`
lets an engine start before its first move.

## 4. Pre-registered predictions (low confidence)

FINDINGS will report whether each held.

1. Value mode beats policy mode.
2. DeepMind puzzles land at 75-85%.
3. The Lichess BOT blitz rating lands at 1800-2200.
4. The chosen size lands at 20-30M parameters (a sweep outcome, not a scaling law).

Itay's own predictions (optional, added before the flagship run):

1.
2.
3.
4.

## 5. Amendments pre-registered on 2026-09-25, before the data they gate

A schedule review (read-only analysts, each checked by a skeptic) proposed these. The commit time of this
section is the evidence that each was written before its data existed.

**PR-1 (P5, arm a10, adopted).** a10 trains the same true wall-clock as D's arms. Its planning rate in D's
units is 10,690.7 x Rt(s-muon) / Rt(s), where Rt is the true rate, the samples over the wall-clock seconds
between consecutive metrics rows from step 300 to 600, from two 600-step real-trainer runs made back to
back in the 2026-09-25 GPU gap: configs/s.toml, then configs/s-muon.toml. No --rate reaches a10; a15, if
it combines a10, plans at the same rate. a10's true training seconds are reported beside D's. The adopt
rule is unchanged; games10k_top1 is reported and does not gate adoption. Read in advance: a null at S keeps
AdamW and FINDINGS says no compute-equivalent gain of about 1.19x or more was detected at S; a pass at S is
expected to shrink at M (arXiv 2509.02046), and P7 has no guard against Muon doing worse than AdamW at M.

**PR-4 (E8 and E2b endgames, adopted).** endgames.epd is screened front to back with the unchanged rule
(SF19, Threads=1, at least +5.00 or a mate at 1M nodes, the same side still at least +5.00 at 10M nodes).
Looks at lines 1,000, 5,000 and 20,000, then every 20,000 lines. The file is declared unable to supply 700
positions if and only if the one-sided 95% Poisson upper bound on the projected file total is below 700
(at the first three looks: kept <= 0, <= 14, <= 73), or the file ends with fewer than 700 kept. A fallback
source is used only with Itay's OK, given before any conversion game: dev (200, E2b) from val-split
endgames with at most 6 queens, rooks, bishops and knights in total, outside test_grouped's groups; final
(500, E8) from test_grouped endgames; both prefiltered on a Lichess label of at least +5.00 or a mate, then
the unchanged SF19 rule. endgames.json records the source and its sha256, the harness commit, the count at
each look and the branch taken. Nothing about the source, thresholds or counts changes after any Blink
conversion game. The screen runs at below-normal priority with at most 4 Stockfish processes before P7 and
3 during P7, and stops during every throughput, calibration, latency or parity measurement.

**PR-5 (P7 length, adopted).** T_long is 120 training hours, fixed, with no adaptive shortening.
configs/long.toml steps = floor(120 x 3600 x R_true / 1024), where R_true is the samples over the wall-clock
seconds between consecutive metrics rows whose later row is in the train phase, leaving out the first 500
steps, measured by a 2,000-step calibration of the final configs/long.toml just before launch (or by N*'s
P6 run after its first 30 minutes). It is never a bench row and never the window samples_per_s. The
flagship evaluates every 4,000 steps (eval_every = 4000); the 5/25/30/50/100% checks, the 30% preview,
30-minute checkpoints, metrics every 50 steps and every stop rule are unchanged. The actual training hours
are reported.

**PR-3 (play latency, adopted: ratified by Itay on 2026-09-25 at 13:25).** The 100 ms p99 rule of section 3 cannot be
met in fp32 by any size (measured 2026-09-24: M 583 ms at 5 games at once, 219 rows), so N* could not be
chosen. Proposed: latency never changes N* (p99 is reported, not gating). Policy mode ships in fp32. Value
mode ships as bf16 + torch.compile if and only if, on the P7 30% preview's weights over the full valprobe,
value-mode move agreement with fp32 is at least 99% and |dVAA| <= 2 sigma_EMA; otherwise fp32. Concurrency:
in the ship mode, at 219 rows, on an otherwise idle machine, soak at least 5,000 moves at each of 5, 4, 3
and 2 games at once; a level passes if the maximum is at most 1,000 ms and no move is over 1,500 ms; P8's
Blink blocks run at the largest passing level and the bot at 2 (or 1). Results go to results/play_mode.json.
The 2026-09-25 gap's fast-play rows are exploratory and gate nothing. It replaces section 3's 100 ms rule.

**PR-2 (P6, adopted: approved by Itay on 2026-09-25 at 13:25, before any P6 data).** N* = M (22,550,272 parameters) by the
research prior (the 120 h optimum was expected at 20-30M parameters; an equal-6 h comparison favours smaller
sizes, since S sees 42.5 samples per parameter to M's 2.7), overridden only by the epoch floor and VRAM. S,
M12 and L are not run at 6 h, and prediction 4 is withdrawn untested. M's 6 h rung is a cooldown branched
from the flagship at step 47,301 (Blink-M, 6.2 GPU-h: 4.96 shared with the flagship plus 1.24 branched), and
a guard replaces the 5% check: if its final EMA VAA is more than 2 sigma below the mean of a01-a03, the
flagship pauses. T_long stays 120 training hours (PR-5); the plan's clause to extend to 132 h for an early
launch is not used.

**PR-4 fallback (authorized by Itay on 2026-09-25 at 13:25).** The screen of endgames.epd stopped after about
9,500 of 157,846 lines with 2 positions passing the 1M-node screen, so PR-4's declaration holds at the 5,000-line
look (kept <= 14). E2b and E8 use PR-4's fallback source, screened with the unchanged SF19 rule, before any
conversion game is played.

**PR-6 (P7 length, adopted by Itay on 2026-09-25 at 14:20; replaces PR-5's fixed 120 h).** The PC is Itay's
during the day, so the flagship trains whenever he does not need the GPU (always at night) and pauses when he
asks. Its length is decided by Itay during the run: at each pause, or at least once a day, the latest EMA
checkpoint is scored on the first 2,000 DeepMind puzzles (value and policy mode, on the CPU) and the result
is shown to him beside DeepMind 9M's 86.6% on the same puzzles and the earlier checks. When he decides the
level is good enough, the run is re-planned so its final 20% 1-sqrt cooldown starts at the current step, and
the cooled model is the flagship. configs/long.toml keeps PR-5's 120-hour step count as the upper bound. The
flagship's actual training hours, the full check history and the step at which he stopped are published;
no claim describes the length as fixed in advance. PR-3's parity and soak tests run on the first check after
24 training hours instead of the 30% preview. Everything else in sections 3-5 stands.
