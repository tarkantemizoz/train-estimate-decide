"""Evaluate a checkpoint or heuristic on a fixed, optionally censored instance."""

import argparse
import json
from pathlib import Path

import dynaplex as dp
import numpy as np

from .model import BaseStockPolicy, GreedyCappedBaseStockPolicy, StationaryFeaturizer, StationaryLostSalesMDP


def summarize(assessment: dp.PolicyAssessment) -> dict:
    stats = assessment.stats
    true_returns = stats.true_cost / stats.time_elapsed
    fills = np.divide(
        stats.units_sold, stats.total_demand, out=np.ones_like(stats.true_cost), where=stats.total_demand > 0
    )
    result = assessment.to_dict()
    result.update(
        true_cost_mean=float(np.mean(true_returns)),
        true_cost_standard_error=float(np.std(true_returns, ddof=1) / np.sqrt(len(true_returns))),
        mean_trajectory_fill_rate=float(np.mean(fills)),
        aggregate_fill_rate=float(np.sum(stats.units_sold) / np.sum(stats.total_demand))
        if np.sum(stats.total_demand)
        else 1.0,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluate.json")
    parser.add_argument("--agent", help="DCL checkpoint directory, e.g. runs/smoke/agent_gen2")
    parser.add_argument("--censored", action="store_true")
    parser.add_argument("--trajectories", type=int, default=1000)
    parser.add_argument("--horizon", type=int, help="Default: 5000 uncensored, 200 censored")
    parser.add_argument("--warmup", type=int, help="Default: 100 uncensored, 0 censored")
    parser.add_argument("--seed", type=int, default=1122)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-stock-level", type=int, help="Also evaluate an unclipped oracle benchmark")
    parser.add_argument("--output", help="Optional JSON result file")
    args = parser.parse_args()
    if args.trajectories < 2:
        parser.error("Use at least 2 trajectories to estimate standard errors")
    config = json.loads(Path(args.config).read_text())
    if args.censored:
        config["censored_demand"] = True
    model = StationaryLostSalesMDP(**config)
    if not model.evaluate or model.oracle_actions:
        parser.error("The evaluation config must use evaluate=true and masked actions")
    censored = model.censored_demand
    settings = dict(
        number_of_trajectories=args.trajectories,
        horizon=args.horizon if args.horizon is not None else (200 if censored else 5000),
        warmup_time=args.warmup if args.warmup is not None else (0 if censored else 100),
        workers=args.workers,
        seed=args.seed,
        jit=True,
    )
    policies = {"greedy_capped_base_stock": GreedyCappedBaseStockPolicy(model)}
    if args.agent:
        agent = dp.NNAgent.load(args.agent)
        if agent.num_actions != model.num_actions:
            parser.error("Agent action count differs: use the same max_p and max_demand as training")
        expected_featurizer = f"{StationaryFeaturizer.__module__}.{StationaryFeaturizer.__qualname__}"
        if agent.featurizer != expected_featurizer:
            parser.error("The checkpoint uses a different model's featurizer")
        if tuple(agent.spec["v"].sample_shape) != (model.pipeline_width + 4,):
            parser.error("Agent feature count differs: use the same max_leadtime as training")
        policies["trained"] = agent
    comparer = dp.PolicyComparer(model, **settings)
    if len(policies) == 1:
        name, policy = next(iter(policies.items()))
        assessments = [comparer.assess(policy, name=name)]
    else:
        assessments = list(comparer.compare(policies))
    if args.base_stock_level is not None:
        oracle = StationaryLostSalesMDP(**(config | {"action_mode": "legacy_oracle"}))
        assessment = dp.PolicyComparer(oracle, **settings).assess(
            BaseStockPolicy(oracle, args.base_stock_level)
        )
        assessments.append(assessment)
    results = {
        "model": config,
        "evaluation": settings,
        "objective": "holding_minus_sales_revenue" if censored else "true_inventory_cost",
        "policies": [summarize(a) for a in assessments],
    }
    text = json.dumps(results, indent=2, allow_nan=False) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
