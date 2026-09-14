import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="CPU-only AI Trader: research, backtest and paper trade"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("download", "Download and verify bulk public historical data"),
        ("train", "Train on downloaded candles and evaluate untouched test data"),
        ("run", "Download -> train -> evaluate -> HTML report"),
    ]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--config", default="configs/default.json")
        cmd.add_argument("--threads", type=int, help="CPU threads; 0 uses all available")
        cmd.add_argument("--start", help="Override historical start date, YYYY-MM-DD")
        cmd.add_argument("--end", help="Override exclusive end date, YYYY-MM-DD")
        cmd.add_argument("--refresh", action="store_true", help="Re-download cached archives")
    paper = sub.add_parser("paper", help="Forward paper trading using fresh public bid/ask quotes")
    paper.add_argument("--run", required=True, type=Path, help="Saved model run directory")
    paper.add_argument("--state", type=Path, default=Path("artifacts/paper.sqlite"))
    paper.add_argument("--poll-seconds", type=float, default=10)
    paper.add_argument("--max-signal-age", type=float, default=60)
    paper.add_argument(
        "--once", action="store_true", help="Observe one cycle and save state; no startup entries"
    )
    status_cmd = sub.add_parser(
        "status", help="Inspect saved paper balance, positions and recent trades"
    )
    status_cmd.add_argument("--state", type=Path, default=Path("artifacts/paper.sqlite"))
    report = sub.add_parser("report", help="Print key metrics from a completed training run")
    report.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command in {"download", "train", "run"}:
            cfg = load_config(args.config)
            if args.threads is not None:
                cfg.model.threads = args.threads
            if args.start:
                cfg.data.start = args.start
            if args.end:
                cfg.data.end = args.end
            if args.refresh:
                cfg.data.refresh = True
            cfg.validate()
            if args.command in {"download", "run"}:
                from .data import download

                manifest = download(cfg)
                print(json.dumps(manifest["symbols"], indent=2), flush=True)
            if args.command in {"train", "run"}:
                from .training import train

                run_dir = train(cfg)
                print(f"\nCompleted: {run_dir}\nOpen {run_dir / 'report.html'}", flush=True)
                print(f"Paper: ai-trader paper --run {run_dir}", flush=True)
        elif args.command == "paper":
            from .paper import run_paper

            run_paper(args.run, args.state, args.poll_seconds, args.once, args.max_signal_age)
        elif args.command == "status":
            from .paper import status

            print(json.dumps(status(args.state), indent=2, allow_nan=False))
        elif args.command == "report":
            data = json.loads((args.run / "training_report.json").read_text())
            print(
                json.dumps(
                    {
                        key: data[key]
                        for key in [
                            "training_completed",
                            "device",
                            "threads",
                            "elapsed_seconds",
                            "chosen_trees",
                            "threshold_bps",
                            "selection_reason",
                            "test",
                        ]
                    },
                    indent=2,
                    allow_nan=False,
                )
            )
    except KeyboardInterrupt:
        print("\nStopped. Download caches and committed paper state are retained.", file=sys.stderr)
        raise SystemExit(130) from None
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from None
