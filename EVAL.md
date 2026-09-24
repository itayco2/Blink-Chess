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
