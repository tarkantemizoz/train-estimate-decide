"""Tune reusable BS/CBS tables and cache independent known-parameter evaluations."""

import argparse
import csv
import json
import multiprocessing
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from zero_shot_lost_sales_inventory_control.benchmark_cache import (
    BenchmarkCache,
    atomic_json,
    fingerprint,
    provenance,
    tuning_dynamics,
)
from zero_shot_lost_sales_inventory_control.benchmarks import legacy_search_bounds
from zero_shot_lost_sales_inventory_control.cases import MODEL_DEFAULTS, instances

from .config import available_cpus


def run_group(task):
    group, settings, phase = task
    cache = BenchmarkCache(settings["cache"])
    started = time.perf_counter()
    records = []
    for instance in group:
        tuning, hit = cache.tune(
            instance, MODEL_DEFAULTS, settings["tuning"], settings["workers"], cache_only=phase == "costs"
        )
        if phase == "tuning":
            record = {"id": instance.id, "hit": hit, "tuning": tuning}
        else:
            costs = cache.evaluate(
                instance.model_config(MODEL_DEFAULTS, "known"),
                tuning["parameters"],
                settings["evaluation"],
                settings["workers"],
            )
            record = {"id": instance.id, "tuning_key": tuning["key"], "policies": costs}
        atomic_json(Path(settings["output"]) / instance.case / phase / f"{instance.id}.json", record)
        records.append(record)
        print(f"{phase}: {instance.id}" + (" (parameter cache hit)" if hit else ""), flush=True)
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
        "simulation_hits": cache.hits,
        "new_simulations": cache.misses,
    }


def execute_phase(groups, settings, phase, case, preparation_seconds=0.0):
    started = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    timing_path = Path(settings["output"]) / case / "timings" / f"{stamp}_{phase}.json"
    results = []

    def snapshot(status, error=None):
        result = {
            "status": status,
            "started_utc": started_utc,
            "updated_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": time.perf_counter() - started + preparation_seconds,
            "preparation_seconds": preparation_seconds,
            "processes": settings["processes"],
            "workers_per_process": settings["workers"],
            "groups": len(groups),
            "completed_groups": len(results),
            "instances": sum(len(group) for group in groups),
            "phase": phase,
            "new_simulations": sum(r["new_simulations"] for r in results),
            "simulation_hits": sum(r["simulation_hits"] for r in results),
            "sum_group_seconds": sum(r["elapsed_seconds"] for r in results),
        }
        if error is not None:
            result["error"] = repr(error)
        if phase == "tuning":
            cold = [row["tuning"] for r in results for row in r["records"] if not row["hit"]]
            result["new_parameter_searches"] = len(cold)
            result["policies"] = {
                name: {
                    field: sum(row["phases"][name][field] for row in cold)
                    for field in ("seconds", "new_simulations", "cache_hits")
                }
                for name in ("BS", "CBS")
            }
        atomic_json(timing_path, result)
        return result

    snapshot("running")
    try:
        with ProcessPoolExecutor(
            max_workers=settings["processes"], mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            pending = {pool.submit(run_group, (group, settings, phase)) for group in groups}
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    results.append(future.result())
                    print(f"{case} {phase}: {len(results)}/{len(groups)} dynamics groups", flush=True)
                snapshot("running")
    except BaseException as error:
        snapshot("failed", error)
        raise
    return snapshot("complete")


def export_table(case, selected, settings):
    folder = Path(settings["output"]) / case
    rows = []
    for instance in selected:
        path = folder / "tuning" / f"{instance.id}.json"
        if not path.exists():
            continue
        tuning = json.loads(path.read_text())["tuning"]
        row = {
            "id": instance.id,
            "p": instance.parameters["p"],
            "information": "known",
            "objective": "true_inventory_cost",
            "parameters_json": json.dumps(instance.parameters, sort_keys=True),
            "tuning_key": tuning["key"],
        } | tuning["parameters"]
        row.update(
            {
                f"{phase}_{key}": value
                for phase in ("tuning", "evaluation")
                for key, value in settings[phase].items()
            }
        )
        cost_path = folder / "costs" / f"{instance.id}.json"
        if cost_path.exists():
            costs = json.loads(cost_path.read_text())
            for name, result in costs["policies"].items():
                row[f"{name}_cost"] = result["summary"]["true_cost_mean"]
                row[f"{name}_standard_error"] = result["summary"]["true_cost_standard_error"]
                row[f"{name}_cache_key"] = result["cache_key"]
        rows.append(row)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (folder / "table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", choices=["case1", "case2", "case3"], default=["case1", "case2", "case3"]
    )
    parser.add_argument("--cache", default="runs/benchmark_cache")
    parser.add_argument("--output", default="runs/benchmark_tables")
    parser.add_argument("--processes", type=int, help="Defaults to the available CPUs divided by workers")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--trajectories", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--tuning-seed", type=int, default=1122)
    parser.add_argument("--evaluation-seed", type=int, default=13021984)
    parser.add_argument("--phase", choices=["all", "tuning", "costs"], default="all")
    parser.add_argument("--instances", nargs="+")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.processes is None:
        args.processes = max(1, available_cpus() // args.workers)
    if min(args.processes, args.workers) < 1:
        parser.error("processes and workers must be positive")
    if args.processes * args.workers > available_cpus():
        parser.error("processes * workers must fit the available CPU allocation")
    budget = {
        "number_of_trajectories": args.trajectories,
        "horizon": args.horizon,
        "warmup_time": args.warmup,
    }
    settings = vars(args) | {
        "tuning": budget | {"seed": args.tuning_seed},
        "evaluation": budget | {"seed": args.evaluation_seed},
    }
    catalogue = [item for case in args.cases for item in instances(case)]
    if args.instances and set(args.instances) - {item.id for item in catalogue}:
        parser.error("Unknown instance IDs for the selected cases")
    identity = {
        "tuning": settings["tuning"],
        "evaluation": settings["evaluation"],
        "implementation": provenance(),
    }
    manifest = Path(args.output) / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        parser.error("Output belongs to another protocol or implementation; choose a fresh --output")
    atomic_json(manifest, identity)
    for case in args.cases:
        preparation_start = time.perf_counter()
        selected = [
            item
            for item in catalogue
            if item.case == case and (not args.instances or item.id in args.instances)
        ]
        if not selected:
            continue
        groups = defaultdict(list)
        for instance in selected:
            groups[fingerprint(tuning_dynamics(instance.model_config(MODEL_DEFAULTS)))].append(instance)

        def estimated_work(group):
            largest_penalty = max(group, key=lambda item: item.parameters["p"])
            cap, inventory = legacy_search_bounds(largest_penalty.model_config(MODEL_DEFAULTS))
            return cap * inventory

        ordered = sorted(groups.values(), key=estimated_work, reverse=True)
        preparation_seconds = time.perf_counter() - preparation_start
        for phase in ["tuning", "costs"] if args.phase == "all" else [args.phase]:
            result = execute_phase(ordered, settings, phase, case, preparation_seconds)
            preparation_seconds = 0.0
            export_table(case, selected, settings)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
