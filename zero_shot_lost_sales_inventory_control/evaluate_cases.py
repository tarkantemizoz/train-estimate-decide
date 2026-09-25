"""Run GC-LSN on the legacy Case 1/2/3 and special paper instance grids."""

import argparse
import copy
import csv
import hashlib
import json
import platform
from collections import defaultdict
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import dynaplex as dp
import numpy as np
import torch

from .benchmark_cache import BenchmarkCache
from .cases import CASE_NAMES, MODEL_DEFAULTS, Instance, instances, modes_for_case
from .evaluate import summarize
from .model import ZeroShotFeaturizer, ZeroShotLostSalesMDP

DEFAULT_CONFIG = {
    "cases": ["case1"],
    "agent": "runs/legacy_gc_lsn/agent_gen5",
    "output": "runs/gc_lsn_evaluation",
    "model": MODEL_DEFAULTS,
    "modes": ["legacy"],
    "instance_ids": [],
    "limit_per_case": None,
    "include_known_reference": True,
    "evaluation": {
        "number_of_trajectories": 1000,
        "known_horizon": 5000,
        "known_warmup": 100,
        "censored_horizons": [200, 500, 1000, 2000],
        "seed": 13021984,
    },
    "execution": {"workers": 2, "torch_threads": 2, "nn_batch": 1024},
    "benchmarks": {
        "enabled": False,
        "cache_directory": "runs/benchmark_cache",
        "number_of_trajectories": 1000,
        "warmup_time": 100,
        "horizon": 5000,
        "seed": 1122,
    },
}


def read_config(path: str) -> dict:
    supplied = json.loads(Path(path).read_text())
    if not isinstance(supplied, dict):
        raise ValueError("Evaluation config must be a JSON object")
    config = copy.deepcopy(DEFAULT_CONFIG)
    unknown = supplied.keys() - config.keys()
    if unknown:
        raise ValueError(f"Unknown evaluation settings: {sorted(unknown)}")
    for key, value in supplied.items():
        if isinstance(config[key], dict):
            if not isinstance(value, dict) or value.keys() - config[key].keys():
                raise ValueError(f"Invalid {key} settings")
            config[key].update(value)
        else:
            config[key] = value
    return config


def require_int(value, minimum: int, name: str) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def select_instances(config: dict) -> list[Instance]:
    cases = config["cases"]
    if not isinstance(cases, list) or not cases or any(case not in CASE_NAMES for case in cases):
        raise ValueError(f"cases must be a nonempty list chosen from {CASE_NAMES}")
    if len(set(cases)) != len(cases):
        raise ValueError("cases must not contain duplicates")
    modes = config["modes"]
    if not isinstance(modes, list) or not modes or any(not isinstance(m, str) for m in modes):
        raise ValueError("modes must be a nonempty list")
    if modes != ["legacy"]:
        if len(set(modes)) != len(modes):
            raise ValueError("modes must not contain duplicates")
        for case in cases:
            if set(modes) - set(modes_for_case(case)):
                raise ValueError(
                    f"Requested modes are not applicable to {case}; use ['legacy'] for mixed cases"
                )
    if (
        type(config["include_known_reference"]) is not bool
        or type(config["benchmarks"]["enabled"]) is not bool
    ):
        raise ValueError("include_known_reference and benchmarks.enabled must be booleans")
    for name in ("agent", "output"):
        if not isinstance(config[name], str) or not config[name]:
            raise ValueError(f"{name} must be a nonempty path string")
    if (
        not isinstance(config["benchmarks"]["cache_directory"], str)
        or not config["benchmarks"]["cache_directory"]
    ):
        raise ValueError("benchmarks.cache_directory must be a nonempty path string")
    for key, minimum in (
        ("number_of_trajectories", 2),
        ("known_horizon", 1),
        ("known_warmup", 0),
        ("seed", 0),
    ):
        require_int(config["evaluation"][key], minimum, f"evaluation.{key}")
    horizons = config["evaluation"]["censored_horizons"]
    if not isinstance(horizons, list) or not horizons:
        raise ValueError("censored_horizons must be a nonempty list")
    for horizon in horizons:
        require_int(horizon, 1, "censored horizon")
    if len(set(horizons)) != len(horizons):
        raise ValueError("censored_horizons must not contain duplicates")
    for key in ("workers", "torch_threads", "nn_batch"):
        require_int(config["execution"][key], 1, f"execution.{key}")
    for key, minimum in (("number_of_trajectories", 2), ("horizon", 1), ("warmup_time", 0), ("seed", 0)):
        require_int(config["benchmarks"][key], minimum, f"benchmarks.{key}")
    ids = config["instance_ids"]
    if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("instance_ids must be a list of distinct instance IDs")
    limit = config["limit_per_case"]
    if limit is not None:
        require_int(limit, 1, "limit_per_case")
    catalogue = [item for case in cases for item in instances(case)]
    if set(ids) - {item.id for item in catalogue}:
        raise ValueError("instance_ids contains IDs outside the selected cases")
    selected = []
    for case in cases:
        items = [x for x in catalogue if x.case == case and (not ids or x.id in ids)]
        selected.extend(items[:limit] if limit is not None else items)
    if not selected:
        raise ValueError("No instances selected")
    # Preflight every selected configuration before creating result directories.
    for item in selected:
        ZeroShotLostSalesMDP(**item.model_config(config["model"]))
    return selected


def windows(config: dict, case: str) -> list[tuple[str, int, int]]:
    modes = modes_for_case(case) if config["modes"] == ["legacy"] else config["modes"]
    evaluation = config["evaluation"]
    return [
        (mode, horizon, evaluation["known_warmup"] if mode == "known" else 0)
        for mode in modes
        for horizon in ([evaluation["known_horizon"]] if mode == "known" else evaluation["censored_horizons"])
    ]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_agent(config: dict) -> dp.NNAgent:
    path = Path(config["agent"])
    if not (path / "agent.json").is_file() or not (path / "weights.pt").is_file():
        raise ValueError(
            f"Checkpoint not found: {path}; set --agent to a saved full-environment checkpoint. "
            "Train one with: python train.py zero_shot_lost_sales_inventory_control CONFIG.json"
        )
    agent = dp.NNAgent.load(str(path))
    model = ZeroShotLostSalesMDP(**(MODEL_DEFAULTS | config["model"]))
    expected = f"{ZeroShotFeaturizer.__module__}.{ZeroShotFeaturizer.__qualname__}"
    if agent.featurizer != expected or agent.num_actions != model.num_actions:
        raise ValueError("Checkpoint featurizer/action domain does not match this super-MDP")
    if tuple(agent.spec["v"].sample_shape) != (model.num_features,):
        raise ValueError("Checkpoint feature count does not match the configured model dimensions")
    if "model_config" in agent.info:
        for key, value in agent.info["model_config"].items():
            if getattr(model, key) != value:
                raise ValueError(f"Model setting {key} differs from the imported checkpoint's contract")
    provenance = path.parent / "run_config.json"
    if provenance.exists():
        trained = MODEL_DEFAULTS | json.loads(provenance.read_text())["config"]["model"]
        for key, value in (MODEL_DEFAULTS | config["model"]).items():
            if value != trained[key]:
                raise ValueError(f"Model setting {key} differs from the checkpoint's training configuration")
    return agent


def experiment_identity(config: dict, case: str) -> dict:
    package = Path(__file__).parent
    return {
        "format": "gc-lsn-cases-v1",
        "case": case,
        "model": config["model"],
        "evaluation": config["evaluation"],
        "execution": config["execution"],
        "benchmarks": config["benchmarks"],
        "include_known_reference": config["include_known_reference"],
        "checkpoint_sha256": {
            name: file_hash(Path(config["agent"]) / name) for name in ("agent.json", "weights.pt")
        },
        "source_sha256": {
            name: file_hash(package / name)
            for name in (
                "model.py",
                "distributions.py",
                "cases.py",
                "benchmarks.py",
                "benchmark_cache.py",
                "evaluate.py",
                "evaluate_cases.py",
            )
        },
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("dynaplex", "numpy", "torch")},
    }


def ensure_manifest(folder: Path, identity: dict) -> None:
    path = folder / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(
                f"{folder} belongs to different settings, code, or weights; choose another output folder"
            )
    else:
        atomic_json(path, identity)


def gap(candidate: float, benchmark: float, revenue: bool = False) -> float | None:
    if benchmark == 0:
        return None
    return 100 * (benchmark - candidate if revenue else candidate - benchmark) / benchmark


def evaluate_window(
    instance: Instance,
    mode: str,
    horizon: int,
    warmup: int,
    config: dict,
    agent: dp.NNAgent,
    parameters: dict | None,
) -> dict:
    model_config = instance.model_config(config["model"], mode)
    model = ZeroShotLostSalesMDP(**model_config)
    settings = {
        "number_of_trajectories": config["evaluation"]["number_of_trajectories"],
        "seed": config["evaluation"]["seed"],
        "horizon": horizon,
        "warmup_time": warmup,
        "workers": config["execution"]["workers"],
        "nn_batch": config["execution"]["nn_batch"],
        "jit": True,
    }
    assessments = [dp.PolicyComparer(model, **settings).assess(agent, name="GC-LSN")]
    cached_baselines = {}
    if parameters is not None:
        # Oracle policies are fixed rules, so their estimators cannot affect
        # actions; retaining legacy censor flags preserves event-stream padding.
        cache = BenchmarkCache(config["benchmarks"]["cache_directory"])
        cached_baselines = cache.evaluate(model_config, parameters, settings, config["execution"]["workers"])
    if mode != "known" and config["include_known_reference"]:
        known_config = instance.model_config(config["model"], "known")
        known_config["maximize_rewards"] = model.maximize_rewards
        known = ZeroShotLostSalesMDP(**known_config)
        assessments.append(dp.PolicyComparer(known, **settings).assess(agent, name="GC-LSN-known"))
    policies = {assessment.name: summarize(assessment) for assessment in assessments}
    policies.update({name: row["summary"] for name, row in cached_baselines.items()})
    primary = policies["GC-LSN"]
    for name, result in policies.items():
        if name != "GC-LSN":
            result["gc_lsn_true_cost_gap_percent"] = gap(primary["true_cost_mean"], result["true_cost_mean"])
            result["legacy_gc_lsn_objective_gap_percent"] = gap(
                primary["mean"], result["mean"], model.maximize_rewards
            )
    if "BS" in policies:
        policies["CBS"]["legacy_cbs_vs_bs_objective_gap_percent"] = gap(
            policies["CBS"]["mean"], policies["BS"]["mean"], model.maximize_rewards
        )
    return {
        "instance": asdict(instance),
        "mode": mode,
        "model": model_config,
        "evaluation": settings,
        "objective": "holding_minus_sales_revenue" if model.maximize_rewards else "true_inventory_cost",
        "benchmark_parameters": parameters,
        "benchmark_cache": {
            name: {k: v for k, v in row.items() if k != "summary"} for name, row in cached_baselines.items()
        },
        "policies": policies,
    }


def summarize_case(folder: Path) -> dict:
    rows = [json.loads(path.read_text()) for path in sorted((folder / "results").glob("*.json"))]
    groups = defaultdict(list)
    csv_rows = []
    for row in rows:
        parameters = row["instance"]["parameters"]
        common = {
            "case": row["instance"]["case"],
            "instance_id": row["instance"]["id"],
            "mode": row["mode"],
            "horizon": row["evaluation"]["horizon"],
            "warmup": row["evaluation"]["warmup_time"],
            "p": parameters["p"],
            "cycle_length": len(parameters["mean_demand"]),
            "leadtime": parameters.get("leadtime"),
            "order_crossover": parameters.get("order_crossover", False),
            "objective": row["objective"],
        }
        for name, result in row["policies"].items():
            flat = common | {"policy": name} | result
            csv_rows.append(flat)
            buckets = [("all", "all")]
            if common["case"] in ("case1", "paper"):
                buckets.extend([("penalty", common["p"]), ("mean_demand", parameters["mean_demand"][0])])
            elif common["case"] == "case2":
                buckets.append(("cycle_length", common["cycle_length"]))
            else:
                buckets.append(("order_crossover", common["order_crossover"]))
            for group, value in buckets:
                groups[(group, value, row["mode"], common["horizon"], name)].append(result)
    summary = []
    metrics = (
        "mean",
        "true_cost_mean",
        "mean_trajectory_fill_rate",
        "aggregate_fill_rate",
        "gc_lsn_true_cost_gap_percent",
        "legacy_gc_lsn_objective_gap_percent",
        "legacy_cbs_vs_bs_objective_gap_percent",
    )
    for (group, value, mode, horizon, policy), records in groups.items():
        item = {
            "group": group,
            "value": value,
            "mode": mode,
            "horizon": horizon,
            "policy": policy,
            "instances": len(records),
        }
        for metric in metrics:
            values = [record[metric] for record in records if record.get(metric) is not None]
            if values:
                item[f"average_{metric}"] = float(np.mean(values))
        summary.append(item)
    result = {
        "completed_windows": len(rows),
        "aggregation": "equal weight per instance within each group",
        "groups": summary,
    }
    atomic_json(folder / "summary.json", result)
    if csv_rows:
        fields = list(dict.fromkeys(key for row in csv_rows for key in row))
        temporary = folder / "results.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        temporary.replace(folder / "results.csv")
    return result


def run(config: dict, selected: list[Instance]) -> None:
    torch.set_num_threads(config["execution"]["torch_threads"])
    agent = load_agent(config)
    output = Path(config["output"])
    active_cases = list(dict.fromkeys(instance.case for instance in selected))
    for case in active_cases:
        ensure_manifest(output / case, experiment_identity(config, case))
    total = sum(len(windows(config, instance.case)) for instance in selected)
    completed = 0
    try:
        for instance in selected:
            folder = output / instance.case
            parameters = None
            if config["benchmarks"]["enabled"]:
                path = folder / "tuning" / f"{instance.id}.json"
                if path.exists():
                    parameters = json.loads(path.read_text())
                else:
                    settings = {
                        key: value
                        for key, value in config["benchmarks"].items()
                        if key not in ("enabled", "cache_directory")
                    }
                    print(f"Load or tune BS/CBS: {instance.id}", flush=True)
                    tuning, hit = BenchmarkCache(config["benchmarks"]["cache_directory"]).tune(
                        instance, config["model"], settings, config["execution"]["workers"]
                    )
                    parameters = tuning["parameters"]
                    atomic_json(path, parameters)
            for mode, horizon, warmup in windows(config, instance.case):
                completed += 1
                path = folder / "results" / f"{instance.id}__{mode}__{horizon}.json"
                if path.exists():
                    print(f"[{completed}/{total}] Resume: {path.stem}", flush=True)
                    continue
                print(f"[{completed}/{total}] Evaluate: {path.stem}", flush=True)
                result = evaluate_window(instance, mode, horizon, warmup, config, agent, parameters)
                atomic_json(path, result)
    finally:
        for case in active_cases:
            summary = summarize_case(output / case)
            print(f"{case}: {summary['completed_windows']} saved windows; {output / case}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/zero_shot/evaluate_cases.json")
    parser.add_argument("--case", nargs="+", choices=(*CASE_NAMES, "all"), help="Override selected cases")
    parser.add_argument("--agent", help="New DynaPlex checkpoint folder")
    parser.add_argument("--output", help="Parent folder for per-case results")
    parser.add_argument("--instances", nargs="+", help="Select exact IDs, e.g. case3-000 case3-120")
    parser.add_argument("--limit", type=int, help="Limit instances per case, in legacy loop order")
    parser.add_argument(
        "--benchmarks", action="store_true", help="Enable legacy BS/CBS tuning and comparison"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print selected instances and budgets without loading a checkpoint",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One instance per case, 4 trajectories, 32 periods; output in a smoke subfolder",
    )
    args = parser.parse_args()
    try:
        config = read_config(args.config)
        if args.case:
            if "all" in args.case and args.case != ["all"]:
                raise ValueError("Use --case all alone")
            config["cases"] = list(CASE_NAMES) if args.case == ["all"] else args.case
        for name in ("agent", "output"):
            if getattr(args, name) is not None:
                config[name] = getattr(args, name)
        if args.instances:
            config["instance_ids"] = args.instances
        if args.limit is not None:
            config["limit_per_case"] = args.limit
        if args.benchmarks:
            config["benchmarks"]["enabled"] = True
        if args.smoke:
            config["limit_per_case"] = 1
            config["output"] = str(Path(config["output"]) / "smoke")
            config["evaluation"].update(
                number_of_trajectories=4, known_horizon=32, known_warmup=5, censored_horizons=[32]
            )
            config["benchmarks"].update(number_of_trajectories=4, warmup_time=5, horizon=32)
        selected = select_instances(config)
        if args.list:
            print(
                json.dumps(
                    {
                        "config": config,
                        "instance_count": len(selected),
                        "evaluation_windows": sum(len(windows(config, item.case)) for item in selected),
                        "instances": [asdict(item) for item in selected],
                    },
                    indent=2,
                )
            )
            return
        run(config, selected)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
