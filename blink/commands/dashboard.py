"""`blink dashboard`: the live training dashboard on http://127.0.0.1:8767 (loopback only, by design)."""

import argparse

from blink import paths
from blink.dashboard import server


def cmd_dashboard(args: argparse.Namespace) -> int:
    server.serve(paths.home() / "runs", args.port)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("dashboard", help="live training dashboard on 127.0.0.1 (never the LAN)")
    parser.add_argument("--port", type=int, default=server.DEFAULT_PORT)
    parser.set_defaults(func=cmd_dashboard)
