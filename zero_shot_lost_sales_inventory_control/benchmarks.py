"""Optional legacy BS/CBS tuning, performed on known-parameter instances."""

import dynaplex as dp
from dynaplex.modelling import DiscreteDist

from .cases import Instance
from .distributions import fit_aer
from .model import BaseStockPolicy, CappedBaseStockPolicy, ZeroShotLostSalesMDP


def legacy_search_bounds(config: dict) -> tuple[int, int]:
    """ReturnBounds uses the supplied raw PMF, including for non-crossing cases."""
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


def search_parameters(assess_bs, assess_cbs, max_action: int, max_inventory: int, max_cap: int) -> dict:
    """Legacy first-nonimprovement searches; not a claim of global optimality."""
    best_bs, best_cost, evaluations = 1, float("inf"), 0
    for level in range(1, max_action + 1):
        cost = assess_bs(level)
        evaluations += 1
        if cost >= best_cost:
            break
        best_bs, best_cost = level, cost
    else:
        raise ValueError("BS search reached the oracle action boundary; increase the model bounds")
    best_s, best_cap, best_cbs_cost = max_inventory, max_cap, float("inf")
    if best_bs > max_inventory:
        raise ValueError("Legacy CBS search interval is empty; inspect the tuning estimates and bounds")
    if max_cap > max_action:
        raise ValueError("CBS search cap exceeds the oracle action domain")
    for level in range(best_bs, max_inventory + 1):
        inner_cost, inner_cap = float("inf"), 0
        # Legacy FindCOLevel always returns zero: its maximum starts at zero,
        # then uses a reversed comparison for strictly positive demand means.
        for cap in range(max_cap + 1):
            cost = assess_cbs(level, cap)
            evaluations += 1
            if cost >= inner_cost:
                break
            inner_cost, inner_cap = cost, cap
        if inner_cost < best_cbs_cost:
            best_s, best_cap, best_cbs_cost = level, inner_cap, inner_cost
    return {
        "base_stock_level": best_bs,
        "capped_base_stock_level": best_s,
        "cap": best_cap,
        "constant_order_lower_bound": 0,
        "max_inventory": max_inventory,
        "max_cap": max_cap,
        "bs_tuning_cost": best_cost,
        "cbs_tuning_cost": best_cbs_cost,
        "policy_evaluations": evaluations,
    }


def tune_benchmarks(instance: Instance, dimensions: dict, settings: dict, workers: int) -> dict:
    config = instance.model_config(dimensions, "known")
    model = ZeroShotLostSalesMDP(**(config | {"action_mode": "legacy_oracle"}))
    comparer = dp.PolicyComparer(model, **settings, workers=workers, jit=True)
    max_cap, max_inventory = legacy_search_bounds(config)
    return search_parameters(
        lambda level: comparer.assess(BaseStockPolicy(model, level)).mean,
        lambda level, cap: comparer.assess(CappedBaseStockPolicy(model, level, cap)).mean,
        model.num_actions - 1,
        max_inventory,
        max_cap,
    )
