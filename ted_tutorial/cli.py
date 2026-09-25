"""Command-line interface for local training and Snellius submission."""

import argparse
import contextlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

from . import ENVIRONMENTS
from .config import read_config, resolve_config


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=ENVIRONMENTS)
    parser.add_argument("config", help="JSON training configuration")
    parser.add_argument("--output", help="Artifact directory (overrides config.output)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print a plan without training or SSH"
    )
    parser.add_argument("--no-wait", action="store_true", help="Return after submitting the Slurm job")
    parser.add_argument("--fetch", action="store_true", help="Download an already submitted remote job")
    parser.add_argument("--in-allocation", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mpi", action="store_true", help="Run distributed collection under an MPI launcher")
    args = parser.parse_args()
    try:
        config = read_config(args.config, args.environment)
        output = Path(args.output or config.get("output", f"runs/{args.environment}"))
        remote = config.get("workstation", {}).get("target", "local") == "snellius" and not args.in_allocation
        if args.dry_run:
            if remote:
                from .remote import job_script, remote_settings

                settings = remote_settings(config)
                print(
                    job_script(settings, args.environment, "/REMOTE/release", "/REMOTE/venv", "/REMOTE/job")
                )
            else:
                if args.mpi:
                    raise ValueError("Inspect MPI through a Snellius config --dry-run before launching ranks")
                print(json.dumps(resolve_config(config), indent=2))
            return
        if args.mpi:
            from .distributed import run_mpi

            run_mpi(args.environment, config, output)
            return
        if remote:
            from .remote import run_remote

            run_remote(args.environment, config, output, no_wait=args.no_wait, fetch_only=args.fetch)
            return
        if args.no_wait or args.fetch:
            raise ValueError("--no-wait and --fetch apply only to Snellius configurations")
        resolved = resolve_config(config)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "training.log").open("a") as log:
            with (
                contextlib.redirect_stdout(Tee(sys.stdout, log)),
                contextlib.redirect_stderr(Tee(sys.stderr, log)),
            ):
                from .training import run_local

                try:
                    run_local(args.environment, resolved, output)
                except BaseException:
                    traceback.print_exc()
                    raise
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
