"""A fixed problem instance in the legacy format, the paper's demand moments, and their distribution."""

import math
from dataclasses import dataclass

from dynaplex.modelling import DiscreteDist

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


def fit_aer(mean: float, std: float) -> DiscreteDist:
    """DynaPlex's native AER fit, except where legacy Mix clamps an endpoint.

    Two legacy Case 2 moment pairs give q=1.0000000000000002.
    Legacy Mix clamps q within 1e-6 of [0, 1]; DynaPlex 1.14 rejects it.
    Build the selected endpoint component with the same legacy parameter p.
    The requested moments, feature values, and all ordinary fits stay unchanged.
    """
    if mean > 0.0 and std >= 0.0:
        cv = std / mean
        a = cv * cv - 1.0 / mean
        if abs(a) >= 1e-6 and a < 0.0 and abs(a + 1.0) >= 1e-10:
            k = int(math.floor(1.0 / -a))
            radicand = -a * k * (1.0 + k) - k
            if radicand >= 0.0:
                q = (1.0 + a * (1.0 + k) + math.sqrt(radicand)) / (1.0 + a)
                p = mean / (k + 1.0 - q)
                if q > 1.0 and q < 1.0 + 1e-6:
                    return DiscreteDist.binomial(k, p)
                if q < 0.0 and q > -1e-6:
                    return DiscreteDist.binomial(k + 1, p)
        elif a >= 1e-6 and a < 1.0:
            k = int(math.floor(1.0 / a))
            radicand = (1.0 + k) * (1.0 - a * k)
            if radicand >= 0.0:
                q = ((1.0 + k) * a - math.sqrt(radicand)) / (1.0 + a)
                p = 1.0 - mean / (k + 1.0 - q + mean)
                if q > 1.0 and q < 1.0 + 1e-6:
                    return DiscreteDist.negative_binomial(float(k), p)
                if q < 0.0 and q > -1e-6:
                    return DiscreteDist.negative_binomial(float(k + 1), p)
    return DiscreteDist.adan_eenige_resing(mean, std)


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
