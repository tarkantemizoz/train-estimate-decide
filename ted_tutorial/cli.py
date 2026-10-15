"""The common training command, with optional Snellius submission."""

import argparse
import json
import subprocess
from pathlib import Path

from . import ENVIRONMENTS
from .config import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=ENVIRONMENTS)
    parser.add_argument("config", help="JSON training configuration")
    parser.add_argument("--output", help="Artifact directory (overrides config.output)")
    parser.add_argument("--dry-run", action="store_true", help="Show DynaPlex's plan without training or SSH")
    parser.add_argument("--no-wait", action="store_true", help="Return after submitting the Slurm job")
    parser.add_argument("--fetch", action="store_true", help="Download an already submitted remote job")
    parser.add_argument("--in-allocation", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mpi", action="store_true", help="Collect across MPI ranks, one per node")
    args = parser.parse_args()
    try:
        config = read_config(args.config, args.environment)
        output = args.output or config.get("output")
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
                from .training import preview

                print(json.dumps(preview(args.environment, config), indent=2))
            return
        if remote or args.mpi:
            if output is None:
                raise ValueError("Snellius/MPI training requires output or --output")
            if args.mpi:
                from .distributed import run_mpi

                run_mpi(args.environment, config, Path(output))
            else:
                from .remote import run_remote

                run_remote(
                    args.environment, config, Path(output), no_wait=args.no_wait, fetch_only=args.fetch
                )
            return
        if args.no_wait or args.fetch:
            raise ValueError("--no-wait and --fetch apply only to Snellius configurations")
        from .training import run_local

        run_local(args.environment, config, output)
    except (ValueError, TypeError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
