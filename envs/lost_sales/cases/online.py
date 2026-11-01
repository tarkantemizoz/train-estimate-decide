"""The 16 online-learning instances of TED Table 2, with the best published online scores.

online_reference.json lists each instance and the profit of the best online-learning method of Lyu et al.
evaluate.py counts how often a trained network beats it.
"""

import json
from pathlib import Path

from .instance import Instance

REFERENCE = Path(__file__).with_name("online_reference.json")


def instances() -> list[Instance]:
    rows = json.loads(REFERENCE.read_text())["instances"]
    return [
        Instance(
            row["instance_id"],
            {
                "p": row["p"],
                "demand_cycles": [0],
                "mean_demand": [row["mean_demand"]],
                "std_demand": [row["std_demand"]],
                "leadtime": row["leadtime"],
            },
            ("known", "demand"),
        )
        for row in sorted(rows, key=lambda row: row["instance_id"])
    ]


def published_scores() -> dict[str, dict[str, float]]:
    """The best published online profit per instance and horizon T."""
    return {
        row["instance_id"]: row["online_profit"] for row in json.loads(REFERENCE.read_text())["instances"]
    }
