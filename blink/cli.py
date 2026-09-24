"""The `blink` command: one task runner for Windows and Linux (it replaces make)."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from blink import checks


def configure_stdio() -> None:
    """Force UTF-8 on stdout and stderr.

    A redirected stream on Windows uses the ANSI code page with strict errors, so printing
    sigma, an arrow or Hebrew into a log file kills a detached run (PF39).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def _cmd_doctor(args: argparse.Namespace) -> int:
    from blink import doctor, paths

    if args.create_layout:
        created = paths.ensure_layout()
        print(f"layout ready under {paths.home()} ({len(created)} directories)")
    results = doctor.run(disk_c=12 * doctor.GB, disk_d=240 * doctor.GB)
    print(checks.format_results(results))
    return checks.exit_code(results)


def _cmd_gate(args: argparse.Namespace) -> int:
    from blink import gate

    results = gate.run()
    print(checks.format_results(results))
    return checks.exit_code(results)


def _cmd_heartbeat_probe(args: argparse.Namespace) -> int:
    from blink import heartbeat

    heartbeat.probe(Path(args.out), minutes=args.minutes, interval=args.interval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blink", description="Blink: a searchless chess transformer.")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="report this machine's facts and check them")
    doctor.add_argument("--create-layout", action="store_true", help="create the BLINK_HOME subdirectories")
    doctor.set_defaults(func=_cmd_doctor)

    gate = sub.add_parser("gate", help="the under-a-minute check run before every phase")
    gate.set_defaults(func=_cmd_gate)

    probe = sub.add_parser(
        "heartbeat-probe", help="write a heartbeat for N minutes (proves detached jobs live)"
    )
    probe.add_argument("--out", required=True)
    probe.add_argument("--minutes", type=float, default=20.0)
    probe.add_argument("--interval", type=float, default=10.0)
    probe.set_defaults(func=_cmd_heartbeat_probe)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_stdio()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
