"""Tune the paper's base-stock (BS) and capped base-stock (CBS) benchmarks with known parameters.

    python -m envs.lost_sales.cases.tune --cases case2 case3

Parameters are stored in benchmark_parameters.csv; evaluate.py reads them and tunes a missing instance on first use.
The policies run on zero_shot_large with the paper's dimensions, which represent every case instance; an instance
of your own that lies outside them gets wider ones.
The search is the legacy first-nonimprovement procedure, so it does not certify a global optimum.
"""

import argparse
import csv
from pathlib import Path

import dynaplex as dp
from dynaplex.modelling import DiscreteDist

import envs

from ..distributions import fit_aer
from ..zero_shot_large.model import DIMENSIONS, ZeroShotLostSalesMDP, instance_config
from ..zero_shot_large.policies import BaseStockPolicy, CappedBaseStockPolicy
from . import CASES

TABLE = Path(__file__).with_name("benchmark_parameters.csv")
FIELDS = ("instance", "base_stock_level", "capped_base_stock_level", "cap")
# The legacy tuning protocol: known parameters, 100 warmup periods, then 5,000 scored periods.
SETTINGS = {"number_of_trajectories": 1000, "horizon": 5000, "warmup_time": 100, "seed": 1122}


def search_bounds(config: dict) -> tuple[int, int]:
    """Legacy bounds for the cap and the CBS level: newsvendor fractiles at and through arrival."""
    probs = config.get("leadtime_distribution")
    if probs is None:
        probs = [0.0] * (config["max_leadtime"] + 1)
        probs[config["leadtime"]] = 1.0
    high, cumulative = 0, 0.0
    for high, prob in enumerate(probs):
        cumulative += prob
        if abs(cumulative - 1.0) < 1e-8:
            break
    distributions = [
        fit_aer(mean, std) for mean, std in zip(config["mean_demand"], config["std_demand"], strict=True)
    ]
    order_bound, inventory_bound = 0, 0
    ratio = config["p"] / (config["p"] + 1.0)
    for period in range(len(distributions)):
        total = DiscreteDist.constant(0)
        at_arrival = DiscreteDist.constant(0)
        through_arrival = DiscreteDist.constant(0)
        weight = 0.0
        for leadtime in range(high + 1):
            dist = distributions[(period + leadtime) % len(distributions)]
            total = total.add(dist)
            prob = probs[leadtime]
            if prob > 0:
                if weight == 0:
                    at_arrival, through_arrival = dist, total
                else:
                    at_arrival = at_arrival.mix(dist, prob / (weight + prob))
                    through_arrival = through_arrival.mix(total, prob / (weight + prob))
                weight += prob
        order_bound = max(order_bound, at_arrival.fractile(ratio))
        inventory_bound = max(inventory_bound, through_arrival.fractile(ratio))
    return order_bound, inventory_bound


def search(assess_bs, assess_cbs, max_action: int, max_inventory: int, max_cap: int) -> dict:
    """Increase each level until the cost stops improving, as the legacy executable does."""
    best_bs, best_cost = 1, float("inf")
    for level in range(1, max_action + 1):
        cost = assess_bs(level)
        if cost >= best_cost:
            break
        best_bs, best_cost = level, cost
    else:
        raise ValueError("BS search reached the oracle action boundary; increase the model bounds")
    if best_bs > max_inventory:
        raise ValueError("Legacy CBS search interval is empty; inspect the tuning estimates and bounds")
    if max_cap > max_action:
        raise ValueError("CBS search cap exceeds the oracle action domain")
    best_level, best_cap, best_cbs_cost = max_inventory, max_cap, float("inf")
    for level in range(best_bs, max_inventory + 1):
        inner_cost, inner_cap = float("inf"), 0
        for cap in range(max_cap + 1):
            cost = assess_cbs(level, cap)
            if cost >= inner_cost:
                break
            inner_cost, inner_cap = cost, cap
        if inner_cost < best_cbs_cost:
            best_level, best_cap, best_cbs_cost = level, inner_cap, inner_cost
    return {"base_stock_level": best_bs, "capped_base_stock_level": best_level, "cap": best_cap}


def dimensions(instance) -> dict:
    """The paper's dimensions, widened to hold an instance outside them; every case instance lies within."""
    parameters = instance.parameters
    probs = parameters.get("leadtime_distribution")
    longest = max(i for i, prob in enumerate(probs) if prob > 0) if probs else parameters["leadtime"]
    return DIMENSIONS | {
        "max_p": max(DIMENSIONS["max_p"], parameters["p"]),
        # The dimensions bound demand by a fit with twice max_demand as its standard deviation.
        "max_demand": max(
            DIMENSIONS["max_demand"], *parameters["mean_demand"], *(s / 2 for s in parameters["std_demand"])
        ),
        "max_leadtime": max(DIMENSIONS["max_leadtime"], longest),
        "max_num_cycles": max(DIMENSIONS["max_num_cycles"], len(parameters["mean_demand"])),
    }


def oracle_config(instance, mode: str) -> dict:
    """The zero_shot_large model of the instance, allowing the benchmarks' unclipped orders."""
    return instance_config(instance, mode, dimensions(instance)) | {"action_mode": "legacy_oracle"}


def tune(instance) -> dict:
    config = oracle_config(instance, "known")
    model = ZeroShotLostSalesMDP(**config)
    comparer = dp.PolicyComparer(model, **SETTINGS, jit=True)
    max_cap, max_inventory = search_bounds(config)
    with envs.sweeping_at_safe_points():
        return search(
            lambda level: comparer.assess(BaseStockPolicy(model, level)).mean,
            lambda level, cap: comparer.assess(CappedBaseStockPolicy(model, level, cap)).mean,
            model.num_actions - 1,
            max_inventory,
            max_cap,
        )


def table(path: Path = TABLE) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        return {
            row.pop("instance"): {key: int(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        }


def parameters(instance, path: Path = TABLE) -> dict:
    """The parameters of an instance stored in the table at path; tuned and stored first if missing."""
    known = table(path)
    if instance.id not in known:
        print(f"Tuning BS and CBS for {instance.id}", flush=True)
        known[instance.id] = tune(instance)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows({"instance": key} | known[key] for key in sorted(known))
    return known[instance.id]


def benchmarks(instance, mode: str, path: Path = TABLE):
    """BS and CBS for evaluate.py, tuned with known parameters and then run in the given mode."""
    rule = parameters(instance, path)
    model = ZeroShotLostSalesMDP(**oracle_config(instance, mode))
    return model, {
        "BS": BaseStockPolicy(model, rule["base_stock_level"]),
        "CBS": CappedBaseStockPolicy(model, rule["capped_base_stock_level"], rule["cap"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    for case in args.cases:
        for instance in CASES[case]():
            print(instance.id, parameters(instance), flush=True)


if __name__ == "__main__":
    main()
