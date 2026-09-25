"""Controller and random-stream capacity checks for pinned DynaPlex 1.14."""

from dynaplex.dcl.machine import _ARM_STRIDE, _SEED_SAMPLE_MAX

_EXECUTION_DEFAULTS = {"workers": 1, "slots": 512, "device": "cpu", "torch_threads": 1}


def resolve_execution(config: dict) -> dict:
    """Fill optional execution settings and size controllers for the sample budget."""
    execution = _EXECUTION_DEFAULTS | config.get("execution", {})
    values = {
        "engines": execution.get("engines", 1),
        "workers": execution["workers"],
        "slots": execution["slots"],
        "torch_threads": execution["torch_threads"],
        "n": config["dcl"]["n"],
        "m": config["dcl"]["m"],
        "generations": config["generations"],
    }
    if "processes" in execution:
        values["processes"] = execution["processes"]
    for name, value in values.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    workers, slots = values["workers"], values["slots"]
    n, m = values["n"], values["m"]

    # collect_samples derives this pool bound. Each completed label rearms a root.
    # Reserve one pool for initial arms and another for the final round's overshoot.
    # With one engine per process, this bounds our infinite-horizon collection
    # (no FINAL retries, and completed engines are not stepped again).
    # With multiple engines it checks nominal quotas only: DynaPlex may keep a
    # completed engine stepping until its peers finish. Its runtime seed guard
    # remains necessary, and saved arms_per_controller_max records the actual use.
    root_pool = (4 * workers * m + 3 * slots * workers + m - 1) // m
    safe_quota = _ARM_STRIDE - 2 * root_pool
    if safe_quota < 1:
        raise ValueError(
            "The root pool leaves no seed capacity; reduce workers/slots per process or increase dcl.m"
        )
    engines = values["engines"]
    minimum_controllers = (n + safe_quota - 1) // safe_quota
    minimum = (minimum_controllers + engines - 1) // engines
    processes = execution.setdefault("processes", minimum)
    controllers = processes * engines
    if controllers > n:
        raise ValueError("processes * engines must not exceed dcl.n (each controller needs samples)")
    if processes < minimum:
        raise ValueError(
            f"DynaPlex's {_ARM_STRIDE:,}-arm seed band cannot fit the nominal {n:,}-sample budget with "
            f"{controllers} controller(s). Use at least {minimum} processes "
            f"(execution.processes or --processes); workers are per controller and do not add seed bands. "
            "Use a new work directory when changing the collection shape."
        )
    max_bands = _SEED_SAMPLE_MAX // _ARM_STRIDE
    if controllers * values["generations"] > max_bands:
        raise ValueError(f"processes * engines * generations must not exceed {max_bands} DynaPlex seed bands")
    return execution
