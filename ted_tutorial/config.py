"""Read tutorial JSON without choosing or filling DynaPlex training options."""

import json
import os
from pathlib import Path

from . import ENVIRONMENTS


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def read_config(path, environment):
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment {environment!r}")
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("The training configuration must be a JSON object")
    if "execution" in config:
        raise ValueError(
            "The custom execution section was removed. Use dcl.machine for a native plan, "
            "and optional dcl.gen0 / dcl.trained objects for explicit Shape overrides."
        )
    allowed = {"model", "dcl", "network", "train", "generations", "workstation", "output"}
    if extra := set(config) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(extra)}")
    for field in ("model", "dcl", "network", "train", "workstation"):
        if field in config and not isinstance(config[field], dict):
            raise ValueError(f"{field} must be an object")
    positive_int(config.get("generations"), "generations")
    if config.get("model", {}).get("evaluate", False):
        raise ValueError("Training requires model.evaluate=false")
    if config.get("workstation", {}).get("target", "local") not in ("local", "snellius"):
        raise ValueError("workstation.target must be local or snellius")
    return config


def available_cpus():
    """CPU allocation for the independent evaluation and benchmark commands."""
    count = os.process_cpu_count() or os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    if allocated := os.environ.get("SLURM_CPUS_PER_TASK"):
        count = min(count, positive_int(int(allocated), "SLURM_CPUS_PER_TASK"))
    return count
