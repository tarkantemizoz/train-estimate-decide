"""Experiment grids from generally_capable_lost_sales_agent.cpp.

Instance order matches the nested loops in the legacy executable.
Distribution names describe its moment construction; the MDP uses the AER fit.
"""

import math
from dataclasses import dataclass

CASE_NAMES = ("case1", "case2", "case3", "paper")
MODEL_DEFAULTS = {
    "max_p": 100.0,
    "max_demand": 12.0,
    "max_leadtime": 10,
    "max_num_cycles": 7,
    "train_stochastic_leadtimes": True,
    "train_cyclic_demand": True,
}
CYCLIC_MEANS = (
    ((2.5, 4.5, 3.0), (9.0, 11.0, 9.5), (3.0, 6.0, 10.0)),
    ((4.0, 2.5, 3.0, 4.5, 3.0), (7.5, 11.5, 10.0, 9.5, 8.5), (11.0, 3.0, 5.0, 8.0, 6.0)),
    (
        (3.0, 4.0, 2.0, 3.5, 4.5, 2.5, 4.0),
        (11.0, 10.0, 9.0, 10.0, 11.0, 8.5, 9.5),
        (3.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0),
    ),
)
STOCHASTIC_MEANS = (
    (5.0,),
    (8.0, 10.0, 6.0),
    (11.0, 3.0, 5.0, 8.0, 6.0),
    (3.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0),
)
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
LEADTIME_DISTRIBUTIONS = (
    (0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    (0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.1, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2),
    (0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3),
    (0.05, 0.05, 0.1, 0.1, 0.15, 0.25, 0.3, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.1, 0.15, 0.25, 0.25, 0.15, 0.1, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5),
    (0.0, 0.0, 0.0, 0.3, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0),
)


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
    id: str
    case: str
    parameters: dict
    labels: dict

    def model_config(self, dimensions: dict, mode: str = "known") -> dict:
        if mode not in modes_for_case(self.case):
            raise ValueError(f"Mode {mode!r} is not applicable to {self.case}")
        config = MODEL_DEFAULTS | dimensions | self.parameters
        config.update(
            evaluate=True,
            censored_demand=mode in ("demand", "both"),
            censored_leadtime=mode in ("leadtime", "both"),
            maximize_rewards=mode in ("demand", "both"),
        )
        if "leadtime_distribution" in config:
            probs = list(config["leadtime_distribution"])
            length = config["max_leadtime"] + 1
            if len(probs) > length and any(probs[length:]):
                raise ValueError(f"{self.id} needs a larger max_leadtime")
            config["leadtime_distribution"] = (probs + [0.0] * max(0, length - len(probs)))[:length]
        return config


def modes_for_case(case: str) -> tuple[str, ...]:
    if case not in CASE_NAMES:
        raise ValueError(f"Unknown case {case!r}; choose from {CASE_NAMES}")
    return ("known", "demand", "leadtime", "both") if case == "case3" else ("known", "demand")


def instances(case: str) -> list[Instance]:
    modes_for_case(case)
    result = []

    def append(means, families, penalty, labels, **arrival):
        parameters = {
            "p": penalty,
            "demand_cycles": list(range(len(means))),
            "mean_demand": list(means),
            "std_demand": [demand_std(m, f) for m, f in zip(means, families, strict=True)],
        } | arrival
        result.append(
            Instance(f"{case}-{len(result):03d}", case, parameters, labels | {"families": list(families)})
        )

    if case in ("case1", "paper"):
        penalties = (5.0, 10.0) if case == "paper" else (9.0, 39.0, 69.0, 99.0)
        means = (10.0,) if case == "paper" else (3.0, 5.0, 7.0, 10.0)
        families = (
            ("poisson", "geometric")
            if case == "paper"
            else ("binomial", "poisson", "negative_binomial", "geometric")
        )
        leadtimes = (1, 3, 5, 7) if case == "paper" else (2, 4, 6, 8, 10)
        for penalty in penalties:
            for mean in means:
                for family in families:
                    for leadtime in leadtimes:
                        append((mean,), (family,), penalty, {"family": family}, leadtime=leadtime)
    elif case == "case2":
        for mean_group in CYCLIC_MEANS:
            for profile, means in enumerate(mean_group):
                for pattern, families in DEMAND_PATTERNS.items():
                    for penalty in (9.0, 39.0, 69.0):
                        for leadtime in (3, 6, 9):
                            append(
                                means,
                                families[: len(means)],
                                penalty,
                                {"mean_profile": profile, "variability": pattern},
                                leadtime=leadtime,
                            )
    else:
        for crossover in (False, True):
            for dist_id, probs in enumerate(LEADTIME_DISTRIBUTIONS):
                for means in STOCHASTIC_MEANS:
                    for penalty in (9.0, 39.0, 69.0):
                        append(
                            means,
                            DEMAND_PATTERNS["mix"][: len(means)],
                            penalty,
                            {"leadtime_distribution_index": dist_id, "variability": "mix"},
                            stochastic_leadtime=True,
                            order_crossover=crossover,
                            leadtime_distribution=list(probs),
                        )
    return result
