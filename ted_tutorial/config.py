"""Configuration validation and allocation-aware execution planning."""

import copy
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
    allowed = {"model", "dcl", "network", "train", "execution", "generations", "workstation", "output"}
    if extra := set(config) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(extra)}")
    for field in ("model", "dcl", "network", "train"):
        if not isinstance(config.get(field), dict):
            raise ValueError(f"{field} must be an object")
    positive_int(config.get("generations"), "generations")
    for name in ("n", "m", "h"):
        positive_int(config["dcl"].get(name), f"dcl.{name}")
    if config["dcl"]["n"] < 2:
        raise ValueError("dcl.n must be at least 2 for training and validation")
    if config["model"].get("evaluate", False):
        raise ValueError("Training requires model.evaluate=false")
    target = config.get("workstation", {}).get("target", "local")
    if target not in ("local", "snellius"):
        raise ValueError("workstation.target must be local or snellius")
    return config


def available_cpus():
    """Use all allowed logical CPUs, respecting affinity and Slurm allocations."""
    count = os.process_cpu_count() or os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        count = min(count, len(os.sched_getaffinity(0)))
    if allocated := os.environ.get("SLURM_CPUS_PER_TASK"):
        count = min(count, positive_int(int(allocated), "SLURM_CPUS_PER_TASK"))
    return count


def choose_device(requested):
    import torch

    devices = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if requested == "auto":
        return next(device for device in ("cuda", "mps", "cpu") if devices[device])
    if requested not in devices:
        raise ValueError("execution.device must be auto, cpu, cuda, or mps")
    if not devices[requested]:
        raise ValueError(f"Requested {requested} device is unavailable in this Python environment")
    return requested


def resolve_config(config, *, cpus=None, world_size=1):
    """Resolve auto fields without changing the supplied research configuration.

    Uniform ranks use the same shape, including for an uneven final sample quota.
    One engine per process avoids continued collection by already-completed engines.
    Native DynaPlex's runtime seed guard remains authoritative.
    """
    from .capacity import resolve_execution

    resolved = copy.deepcopy(config)
    cpus = positive_int(cpus if cpus is not None else available_cpus(), "available CPUs")
    positive_int(world_size, "world_size")
    execution = resolved.setdefault("execution", {})
    allowed = {"processes", "engines", "workers", "slots", "device", "torch_threads", "fit_threads"}
    if extra := set(execution) - allowed:
        raise ValueError(f"Unknown execution keys: {sorted(extra)}")
    execution["device"] = choose_device(execution.get("device", "auto"))
    engines = positive_int(execution.get("engines", 1), "execution.engines")
    if engines != 1:
        raise ValueError("The portable trainer uses one engine per process")
    execution["engines"] = engines
    execution["slots"] = positive_int(execution.get("slots", 512), "execution.slots")
    largest_quota = (resolved["dcl"]["n"] + world_size - 1) // world_size
    smallest_quota = resolved["dcl"]["n"] // world_size
    if smallest_quota < 2:
        raise ValueError("Each MPI rank needs at least two samples")
    requested_processes = execution.get("processes", "auto")
    requested_workers = execution.get("workers", "auto")
    # A divisor uses the full allocation without giving each child an all-core pool.
    candidates = [p for p in range(1, cpus + 1) if cpus % p == 0]
    if cpus >= 32 and execution["device"] == "cpu":
        # Avoid one enormous neural inference pool on a many-core server.
        # The pinned Genoa preset also uses groups of about six CPU cores.
        candidates = [p for p in candidates if cpus // p <= 6]
    if requested_processes != "auto":
        candidates = [positive_int(requested_processes, "execution.processes")]
    for processes in candidates:
        workers = (
            cpus // processes
            if requested_workers == "auto"
            else positive_int(requested_workers, "execution.workers")
        )
        if workers < 1 or processes * workers > cpus or processes > smallest_quota:
            continue
        trial = copy.deepcopy(resolved)
        trial["dcl"]["n"] = largest_quota
        trial["generations"] *= world_size
        trial["execution"] = {
            "processes": processes,
            "engines": 1,
            "workers": workers,
            "slots": execution["slots"],
            "device": execution["device"],
            "torch_threads": 1,
        }
        try:
            resolve_execution(trial)
        except ValueError:
            continue
        execution.update(processes=processes, workers=workers)
        break
    else:
        raise ValueError(
            "Cannot fit this sample budget and execution shape within the CPU/seed allocation; "
            "reduce N or use more CPUs/nodes (or use auto processes/workers)"
        )
    for key, default in (("torch_threads", max(1, cpus // execution["processes"])), ("fit_threads", cpus)):
        value = execution.get(key, "auto")
        execution[key] = default if value == "auto" else positive_int(value, f"execution.{key}")
        if execution[key] > cpus:
            raise ValueError(f"execution.{key} exceeds the allocated CPUs")
    if execution["processes"] * execution["torch_threads"] > cpus:
        raise ValueError("processes * torch_threads exceeds the allocated CPUs")
    return resolved
