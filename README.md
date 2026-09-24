# Blink

A chess AI that picks its move without searching: a transformer trained by supervised learning on the
Lichess evaluation database (409,710,113 positions scored by Stockfish, CC0), on one home RTX 3070.

**Work in progress.** The build is public as it happens. Every number below is measured, and the
headline numbers (Elo against Stockfish anchors, a public Lichess rating, DeepMind's 10K puzzles) arrive
only when the flagship model is trained and evaluated. How they will be measured is fixed in advance in
[EVAL.md](EVAL.md).

## The idea

Chess engines win by calculating millions of moves ahead. Blink does not calculate. It plays in one of
two ways, and the measurements decide which one ships:

- **One look:** one pass through the network, straight to a move.
- **One look per move:** the network scores the position after each legal move once, in one batch, and
  plays the best. It never looks at the opponent's reply.

The exact rule, what counts as search and how every published game proves compliance, is in
[EVAL.md](EVAL.md).

## Where it stands

- [x] P0 environment, doctor and the frozen board contract (1880-move vocabulary, 64 square codes)
- [x] P1 walking skeleton, end to end on a 0.34M-parameter model trained for 2.5 minutes
- [ ] P2 the full 409.7M-position pack, with a 636,245-position leakage blocklist
- [ ] P3-P6 baselines, recipe experiments and a size sweep
- [ ] P7 the flagship run
- [ ] P8-P13 evaluation, the Lichess bot, the browser page, the learning film

The walking skeleton, to show the pipeline works (not Blink's strength):

| check | result |
|---|---|
| initial losses | 7.539 and 4.852 (ln 1880 and ln 128: a network that knows nothing) |
| after 3,000 steps | policy loss 2.94, value loss 3.25, top-1 on held-out positions 28.3% |
| DeepMind puzzles, first 1,000 | 25.8% one look, 29.7% one look per move, 0 illegal moves |
| 100 games against a random mover | 99.5% one look |
| no-search audit | 0 violations in 6,166 decisions |
| browser (ONNX, one thread, WASM) | 10.9 ms per move, identical to Python on 50/50 test positions |

Every defect found on the way is in [PREFLIGHT.md](PREFLIGHT.md), with the number that exposed it.

## Run the tests

```
uv sync
uv run pytest
```

Code: [MIT](LICENSE). python-chess is a GPL-3.0-or-later runtime dependency (not vendored).
Built with an AI coding agent. I chose the design and the gates, and ran and checked every experiment.
