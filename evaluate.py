"""Evaluate heuristics, and optionally a trained agent, with the evaluator of the environment's family.

    python evaluate.py zero_shot_lost_sales_stationary --agent reference/demo
    python evaluate.py zero_shot_lost_sales_stationary --help

The lost-sales environments are evaluated by envs/lost_sales/evaluate.py; see docs/evaluation.md.
A family of your own is evaluated the same way once it has an evaluate.py with a main().
"""

import importlib
import sys

import envs


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in envs.names():
        print(__doc__ + "\nenvironments: " + ", ".join(envs.names()))
        sys.exit(0 if sys.argv[1:] in (["-h"], ["--help"]) else 2)
    family = envs.find(sys.argv[1]).__name__.rsplit(".", 1)[0]
    importlib.import_module(f"{family}.evaluate").main()


if __name__ == "__main__":
    main()
