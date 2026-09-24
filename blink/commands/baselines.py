"""`blink baselines train --kind linear|mlp --positions N --data DIR`: ladder rungs 2 and 3 (plan P3).

Writes BLINK_HOME/runs/baseline-<kind>/model.pt and metrics.json. Torch is imported only when the
command runs, so `blink --help` stays fast and works on the torch-free CI leg.
"""

import argparse
import sys
from pathlib import Path

from blink import paths
from blink.train.status import valid_run_name

EXIT_REFUSED = 2
KINDS = ("linear", "mlp")
LADDER_POSITIONS = 10_000_000


def _device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_train(args: argparse.Namespace) -> int:
    from blink.baselines import train

    try:
        cfg = train.BaselineConfig(
            kind=args.kind,
            positions=args.positions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            val_positions=args.val_positions,
            device=_device(args.device),
        )
        run = args.run or f"baseline-{args.kind}"
        if not valid_run_name(run):
            raise ValueError(f"bad run name {run!r} (letters, digits, _ - . only)")
        out = paths.home() / "runs" / run
        metrics = train.train_baseline(cfg, Path(args.data), out, log=lambda line: print(line, flush=True))
    except (FileNotFoundError, ValueError) as exc:
        print(f"blink baselines train: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"{out.name}: val win% MAE {metrics['val_mae']:.4f} after epoch {metrics['best_epoch']}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    baselines = sub.add_parser("baselines", help="the learned baseline rungs (linear, MLP)")
    actions = baselines.add_subparsers(dest="baselines_command", required=True)
    train = actions.add_parser("train", help="train one rung on the ladder's fixed training set")
    train.add_argument("--kind", choices=KINDS, required=True)
    train.add_argument("--positions", type=int, default=LADDER_POSITIONS, help="the first N train roots")
    train.add_argument("--data", required=True, help="a packed shard directory (v1 or skeleton layout)")
    train.add_argument("--epochs", type=int, default=5, help="at most 5; early stop on val")
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--lr", type=float, default=None, help="default: 3e-3 linear, 1e-3 MLP")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--val-positions", type=int, default=500_000, help="val roots read (front of file)")
    train.add_argument("--device", choices=("cuda", "cpu"), help="default: cuda when available")
    train.add_argument("--run", help="run name under BLINK_HOME/runs (default baseline-<kind>)")
    train.set_defaults(func=cmd_train)
