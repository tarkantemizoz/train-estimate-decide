"""Convert supported catalogue instances into stationary-model configurations."""


def stationary_config(source: dict) -> dict:
    """Convert only a fixed, single-period, deterministic-lead-time instance."""
    if source.get("evaluate") is not True:
        raise ValueError("Only fixed evaluation instances can be exported")
    if source.get("demand_cycles") != [0]:
        raise ValueError("Cyclic demand cannot be exported to the stationary environment")
    for name in ("mean_demand", "std_demand"):
        if not isinstance(source.get(name), list) or len(source[name]) != 1:
            raise ValueError(f"{name} must contain exactly one value")
    if (
        source.get("stochastic_leadtime", False)
        or source.get("order_crossover", False)
        or source.get("censored_leadtime", False)
        or source.get("leadtime_distribution") is not None
    ):
        raise ValueError("Only known deterministic lead times apply to the stationary environment")
    censored = source.get("censored_demand", False)
    if source.get("maximize_rewards", False) != censored:
        raise ValueError("The stationary objective is determined by its demand-censoring flag")
    return {
        "evaluate": True,
        "max_p": source["max_p"],
        "max_demand": source["max_demand"],
        "max_leadtime": source["max_leadtime"],
        "p": source["p"],
        "leadtime": source["leadtime"],
        "mean_demand": source["mean_demand"][0],
        "std_demand": source["std_demand"][0],
        "censored_demand": censored,
        "action_mode": source.get("action_mode", "masked"),
        "discount_factor": source.get("discount_factor", 1.0),
    }
