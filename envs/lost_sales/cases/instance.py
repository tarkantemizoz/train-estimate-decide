"""A fixed problem instance in the legacy format, and the paper's demand moments."""

import math
from dataclasses import dataclass

DEMAND_PATTERNS = {
    "low": ("poisson", "binomial", "poisson", "binomial", "poisson", "binomial", "poisson"),
    "high": (
        "geometric",
        "negative_binomial",
        "geometric",
        "negative_binomial",
        "geometric",
        "negative_binomial",
        "geometric",
    ),
    "mix": (
        "poisson",
        "geometric",
        "negative_binomial",
        "binomial",
        "poisson",
        "geometric",
        "negative_binomial",
    ),
}


def demand_std(mean: float, family: str) -> float:
    """Reproduce the executable's moments, including C++ positive rounding."""
    p_dummy = 0.3
    if family == "binomial":
        n = math.floor(mean / p_dummy + 0.5)
        prob = mean / n
        return math.sqrt(n * prob * (1 - prob))
    if family == "poisson":
        return math.sqrt(mean)
    if family == "negative_binomial":
        r = max(2, math.ceil(mean * p_dummy / (1 - p_dummy)))
        prob = r / (mean + r)
        return math.sqrt(mean / prob)
    if family == "geometric":
        prob = 1 / (1 + mean)
        return math.sqrt((1 - prob) / (prob * prob))
    raise ValueError(f"Unknown demand family: {family}")


@dataclass(frozen=True)
class Instance:
    """An instance and the information modes the paper evaluates it in.

    parameters: p, demand_cycles, mean_demand and std_demand (one entry per period of the demand cycle),
    and either a fixed leadtime or stochastic_leadtime with a leadtime_distribution and order_crossover.
    Modes: "known" parameters, or estimated from observations: "demand", "leadtime" or "both".
    """

    id: str
    parameters: dict
    modes: tuple[str, ...]


def instance(case: str, index: int, means, families, penalty: float, modes, **arrival) -> Instance:
    """One demand period per (mean, family) pair, with the executable's moment construction."""
    parameters = {
        "p": penalty,
        "demand_cycles": list(range(len(means))),
        "mean_demand": list(means),
        "std_demand": [demand_std(m, f) for m, f in zip(means, families, strict=True)],
    } | arrival
    return Instance(f"{case}-{index:03d}", parameters, modes)
