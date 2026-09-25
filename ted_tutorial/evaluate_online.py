"""Evaluate a stationary policy on the paper's 16 online-learning instances."""

import argparse
import csv
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import dynaplex as dp
import numpy as np
import torch

from lost_sales_stationary.evaluate import summarize
from lost_sales_stationary.instances import stationary_config
from lost_sales_stationary.model import GreedyCappedBaseStockPolicy, StationaryLostSalesMDP
from zero_shot_lost_sales_inventory_control.cases import MODEL_DEFAULTS, instances

from .config import available_cpus


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paper_instances():
    """Resolve the fixed online grid directly from the shared case catalogue."""
    return [
        {
            "id": item.id,
            "labels": item.labels,
            "config": stationary_config(item.model_config(MODEL_DEFAULTS, "demand")),
        }
        for item in instances("paper")
    ]


def evaluate_instance(task):
    item, settings = task
    torch.set_num_threads(1)
    config = item["config"]
    model = StationaryLostSalesMDP(**config)
    agent = dp.NNAgent.load(settings["agent"])
    agent.to("cpu")
    records = []
    for horizon in settings["horizons"]:
        output = Path(settings["output"]) / f"{item['id']}__{horizon}.json"
        if output.exists():
            records.append(json.loads(output.read_text()))
            continue
        started = time.perf_counter()
        comparer = dp.PolicyComparer(
            model,
            number_of_trajectories=settings["trajectories"],
            horizon=horizon,
            warmup_time=0,
            workers=settings["workers"],
            seed=settings["seed"],
            jit=True,
            nn_batch=1024,
        )
        policies = {"greedy": GreedyCappedBaseStockPolicy(model), "trained": agent}
        assessments = list(comparer.compare(policies))
        record = {
            "id": item["id"],
            "family": item["labels"]["family"],
            "model": config,
            "horizon": horizon,
            "trajectories": settings["trajectories"],
            "seed": settings["seed"],
            "warmup": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "policies": {},
        }
        raw = {}
        for assessment in assessments:
            metrics = summarize(assessment)
            profits = -assessment.returns
            profit = float(profits.mean())
            se = float(profits.std(ddof=1) / np.sqrt(len(profits)))
            metrics.update(
                profit_mean=profit,
                profit_standard_error=se,
                profit_ci95=[profit - 1.96 * se, profit + 1.96 * se],
            )
            record["policies"][assessment.name] = metrics
            raw[f"{assessment.name}_profit"] = profits
            raw[f"{assessment.name}_demand"] = assessment.stats.total_demand
        if not np.array_equal(raw["greedy_demand"], raw["trained_demand"]):
            raise RuntimeError("Common exogenous demands differ between policies")
        delta = raw["trained_profit"] - raw["greedy_profit"]
        record["paired_profit_gain_vs_greedy"] = {
            "mean": float(delta.mean()),
            "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        }
        np.savez_compressed(output.with_suffix(".npz"), **raw)
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        records.append(record)
        print(
            f"{item['id']} T={horizon}: profit={record['policies']['trained']['profit_mean']:.4f}", flush=True
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectories", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13021984)
    parser.add_argument(
        "--processes", type=int, help="Default: an allocation-aware process count, up to five"
    )
    parser.add_argument(
        "--workers", type=int, help="Simulation workers per process; defaults to remaining CPUs"
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[200, 500, 1000, 2000])
    parser.add_argument("--instances", nargs="+")
    args = parser.parse_args()
    if args.trajectories < 2:
        parser.error("At least two trajectories are required")
    cpus = available_cpus()
    if args.processes is None:
        args.processes = max(p for p in range(1, min(cpus, 5) + 1) if cpus % p == 0)
    if args.workers is None:
        args.workers = max(1, cpus // args.processes)
    if min(args.processes, args.workers) < 1 or args.processes * args.workers > cpus:
        parser.error("processes * workers must fit the available CPU allocation")
    if (
        not args.horizons
        or any(h < 1 for h in args.horizons)
        or len(set(args.horizons)) != len(args.horizons)
    ):
        parser.error("horizons must contain distinct positive integers")
    items = paper_instances()
    if args.instances:
        if set(args.instances) - {item["id"] for item in items}:
            parser.error("Unknown paper instance ID")
        items = [item for item in items if item["id"] in args.instances]
    settings = vars(args)
    identity = settings | {
        "checkpoint_sha256": {name: digest(Path(args.agent) / name) for name in ["agent.json", "weights.pt"]},
        "source_sha256": {
            name: digest(Path(__file__).resolve().parents[1] / name)
            for name in [
                "ted_tutorial/evaluate_online.py",
                "lost_sales_stationary/model.py",
                "lost_sales_stationary/evaluate.py",
                "lost_sales_stationary/instances.py",
                "zero_shot_lost_sales_inventory_control/cases.py",
            ]
        },
        "versions": {name: version(name) for name in ["dynaplex", "numpy", "torch"]},
        "instances": {item["id"]: item["config"] for item in items},
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "experiment.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Evaluation identity differs; use a fresh output folder")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.processes, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(evaluate_instance, (item, settings)) for item in items]
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: (row["id"], row["horizon"]))
    fields = [
        "id",
        "family",
        "p",
        "leadtime",
        "horizon",
        "profit",
        "se",
        "ci95_low",
        "ci95_high",
        "greedy_profit",
        "paired_gain",
        "paired_se",
    ]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["policies"]["trained"]
            writer.writerow(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "p": row["model"]["p"],
                    "leadtime": row["model"]["leadtime"],
                    "horizon": row["horizon"],
                    "profit": metrics["profit_mean"],
                    "se": metrics["profit_standard_error"],
                    "ci95_low": metrics["profit_ci95"][0],
                    "ci95_high": metrics["profit_ci95"][1],
                    "greedy_profit": row["policies"]["greedy"]["profit_mean"],
                    "paired_gain": row["paired_profit_gain_vs_greedy"]["mean"],
                    "paired_se": row["paired_profit_gain_vs_greedy"]["standard_error"],
                }
            )
    (output / "timing.json").write_text(
        json.dumps({"elapsed_seconds": time.perf_counter() - started}, indent=2)
    )
    print(f"Completed {len(rows)} windows: {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
