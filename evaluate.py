"""Evaluate heuristics, and optionally a trained agent, with the evaluator of the environment's family.

    python evaluate.py lost_sales/zero_shot_stationary --agent reference/demo
    python evaluate.py lost_sales/zero_shot_stationary --help

An environment <family>/<name> is evaluated by envs/<family>/evaluate.py; for lost_sales, see docs/evaluation.md.
A family of your own is evaluated the same way once it has an evaluate.py with a main().
"""

import importlib
import sys

import envs


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in envs.names():
        print(__doc__ + "\nenvironments: " + ", ".join(envs.names()))
        sys.exit(0 if sys.argv[1:] in (["-h"], ["--help"]) else 2)
    family = sys.argv[1].split("/")[0]
    importlib.import_module(f"envs.{family}.evaluate").main()


if __name__ == "__main__":
    main()
