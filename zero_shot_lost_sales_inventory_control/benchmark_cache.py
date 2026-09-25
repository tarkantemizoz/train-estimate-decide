"""Persistent, checkpoint-independent BS/CBS tuning and simulation records."""

import hashlib
import json
import os
import platform
import tempfile
import time
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import dynaplex as dp
import numpy as np

from .benchmarks import legacy_search_bounds, search_parameters
from .evaluate import summarize
from .model import BaseStockPolicy, CappedBaseStockPolicy, ZeroShotLostSalesMDP


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


@lru_cache(maxsize=1)
def provenance():
    folder = Path(__file__).parent
    package_code = {}
    for name, module in (("dynaplex", dp), ("numpy", np)):
        package_root = Path(module.__file__).resolve().parent
        hashes = {
            str(path.relative_to(package_root)): file_hash(path)
            for path in sorted(package_root.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".so", ".dylib", ".pyd", ".dll")
        }
        package_code[name] = {"sha256": fingerprint(hashes), "files": len(hashes)}
    return {
        "schema": "bs-cbs-cache-v1",
        "python": platform.python_version(),
        "platform": [platform.system(), platform.machine()],
        "packages": {name: version(name) for name in ("dynaplex", "numpy")},
        "installed_package_code": package_code,
        "source_sha256": {
            name: file_hash(folder / name)
            for name in ("model.py", "distributions.py", "benchmarks.py", "benchmark_cache.py", "evaluate.py")
        },
    }


def simulation_settings(settings):
    """Worker counts do not change PolicyComparer's indexed trajectory seeds."""
    keys = ("number_of_trajectories", "horizon", "warmup_time", "seed")
    result = {key: settings[key] for key in keys}
    for key, minimum in zip(keys, (2, 1, 0, 0), strict=True):
        if type(result[key]) is not int or result[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    return result


def canonical_policy(policy):
    """Merge identical fixed order rules, without changing their trajectories."""
    level = policy["level"]
    if type(level) is not int or level < 0:
        raise ValueError("Stock level must be a nonnegative integer")
    if policy["kind"] == "BS":
        return {"kind": "BS", "level": level}
    if policy["kind"] != "CBS" or type(policy["cap"]) is not int or policy["cap"] < 0:
        raise ValueError("Expected BS or CBS with a nonnegative integer cap")
    if policy["cap"] == 0 or level == 0:
        return {"kind": "BS", "level": 0}
    if policy["cap"] >= level:
        return {"kind": "BS", "level": level}
    return {"kind": "CBS", "level": level, "cap": policy["cap"]}


def tuning_dynamics(config):
    """Canonical penalty only where p cannot alter actions or RNG consumption."""
    result = dict(config)
    # Crossing arrivals pad draws to a p-dependent bound. Keep their p intact.
    if not config.get("order_crossover", False):
        result["p"] = 2.0
    return result


class BenchmarkCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.comparers = {}
        self.hits = 0
        self.misses = 0

    def _identity(self, kind, config, settings, **extra):
        return {
            "kind": kind,
            "model": config | {"action_mode": "legacy_oracle"},
            "evaluation": simulation_settings(settings),
            "implementation": provenance(),
        } | extra

    def assess(self, config, policy, settings, workers=1):
        """Return a summary and raw trajectories, calculating only on a cache miss."""
        policy = canonical_policy(policy)
        identity = self._identity("assessment", config, settings, policy=policy)
        key = fingerprint(identity)
        path = self.directory / "assessments" / key[:2] / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            raw_path = path.with_suffix(".npz")
            if record["identity"] != identity or file_hash(raw_path) != record["raw_sha256"]:
                raise ValueError(f"Benchmark cache integrity check failed: {path}")
            with np.load(raw_path, allow_pickle=False) as archive:
                raw = {name: archive[name].copy() for name in archive.files}
            self.hits += 1
            return record, raw, True
        started = time.perf_counter()
        comparer_key = fingerprint(
            {"model": identity["model"], "settings": identity["evaluation"], "workers": workers}
        )
        if comparer_key not in self.comparers:
            model = ZeroShotLostSalesMDP(**identity["model"])
            self.comparers[comparer_key] = (
                model,
                dp.PolicyComparer(model, **identity["evaluation"], workers=workers, jit=True),
            )
        model, comparer = self.comparers[comparer_key]
        rule = (
            BaseStockPolicy(model, policy["level"])
            if policy["kind"] == "BS"
            else CappedBaseStockPolicy(model, policy["level"], policy["cap"])
        )
        assessment = comparer.assess(rule, name=policy["kind"])
        stats = assessment.stats
        raw = {"returns": assessment.returns.copy()}
        for name in ("true_cost", "lost_units", "total_demand", "units_sold", "time_elapsed"):
            raw[name] = np.asarray(getattr(stats, name)).copy()
        raw["holding"] = raw["true_cost"] - config["p"] * raw["lost_units"]
        if not config.get("maximize_rewards", False):
            reconstructed = (raw["holding"] + config["p"] * raw["lost_units"]) / raw["time_elapsed"]
            if not np.array_equal(reconstructed, raw["returns"]):
                raise ValueError("Cost components do not exactly reconstruct assessment returns")
        record = {
            "key": key,
            "identity": identity,
            "summary": summarize(assessment),
            "simulation_seconds": time.perf_counter() - started,
            "workers": workers,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **raw)
        record["raw_sha256"] = file_hash(temporary)
        os.replace(temporary, path.with_suffix(".npz"))
        atomic_json(path, record)
        self.misses += 1
        return record, raw, False

    def tune(self, instance, dimensions, settings, workers=1, *, cache_only=False):
        config = instance.model_config(dimensions, "known")
        identity = self._identity("tuning", config, settings, search="legacy-first-nonimprovement-v1")
        key = fingerprint(identity)
        path = self.directory / "parameters" / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["identity"] != identity:
                raise ValueError(f"Benchmark parameter identity mismatch: {path}")
            return record, True
        if cache_only:
            raise ValueError(f"No cached parameters for {instance.id}. Run the tuning phase first.")
        started = time.perf_counter()
        canonical = tuning_dynamics(config)
        max_cap, max_inventory = legacy_search_bounds(config)
        model = ZeroShotLostSalesMDP(**(config | {"action_mode": "legacy_oracle"}))
        timing = {name: {"seconds": 0.0, "new_simulations": 0, "cache_hits": 0} for name in ("BS", "CBS")}
        candidate_keys = set()

        def assess(policy):
            before = time.perf_counter()
            result, raw, hit = self.assess(canonical, policy, settings, workers)
            candidate_keys.add(result["key"])
            for name in ("holding", "lost_units"):
                values = raw[name]
                if not np.all(np.isfinite(values) & (values == np.floor(values)) & (np.abs(values) < 2**53)):
                    raise ValueError("Penalty reuse requires exactly represented integer cost components")
            phase = timing[policy["kind"]]
            phase["seconds"] += time.perf_counter() - before
            phase["cache_hits" if hit else "new_simulations"] += 1
            # Retain per-trajectory arithmetic and mean reduction, including ties.
            returns = (raw["holding"] + config["p"] * raw["lost_units"]) / raw["time_elapsed"]
            return float(returns.mean())

        parameters = search_parameters(
            lambda level: assess({"kind": "BS", "level": level}),
            lambda level, cap: assess({"kind": "CBS", "level": level, "cap": cap}),
            model.num_actions - 1,
            max_inventory,
            max_cap,
        )
        record = {
            "key": key,
            "instance_id": instance.id,
            "identity": identity,
            "parameters": parameters,
            "elapsed_seconds": time.perf_counter() - started,
            "phases": timing,
            "candidate_keys": sorted(candidate_keys),
        }
        atomic_json(path, record)
        return record, False

    def evaluate(self, config, parameters, settings, workers=1):
        result = {}
        returns = {}
        for name, policy in {
            "BS": {"kind": "BS", "level": parameters["base_stock_level"]},
            "CBS": {"kind": "CBS", "level": parameters["capped_base_stock_level"], "cap": parameters["cap"]},
        }.items():
            record, raw, hit = self.assess(config, policy, settings, workers)
            returns[name] = raw["returns"]
            result[name] = {
                "summary": record["summary"] | {"name": name},
                "cache_key": record["key"],
                "cache_hit": hit,
            }
        for name in ("BS", "CBS"):
            difference = returns[name] - returns["BS"]
            result[name]["summary"].update(
                delta_mean=float(difference.mean()),
                delta_error=float(difference.std(ddof=1) / np.sqrt(len(difference))),
                is_benchmark=name == "BS",
            )
        return result
