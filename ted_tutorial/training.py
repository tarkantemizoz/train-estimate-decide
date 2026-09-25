"""Shared model construction, provenance, and a resumable DCL generation loop."""

import hashlib
import importlib
import json
import os
import platform
import shutil
import socket
import time
from importlib.metadata import version
from pathlib import Path

from . import DYNAPLEX_VERSION


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def record_timing(output, result):
    """Keep the first completion timing when a later invocation only resumes."""
    write_json(output / "latest_invocation.json", result)
    if not (output / "timing.json").exists():
        write_json(output / "timing.json", result)


def source_hash():
    root = Path(__file__).resolve().parents[1]
    files = sorted(
        p
        for package in ("ted_tutorial", "lost_sales_stationary", "zero_shot_lost_sales_inventory_control")
        for p in (root / package).glob("*.py")
    )
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def model_parts(environment, config):
    module = importlib.import_module(f"{environment}.model")
    if environment == "lost_sales_stationary":
        model = module.StationaryLostSalesMDP(**config["model"])
        features = module.StationaryFeaturizer
    else:
        model = module.ZeroShotLostSalesMDP(**config["model"])
        features = module.ZeroShotFeaturizer
    if model.evaluate:
        raise ValueError("Training requires a super-MDP with evaluate=False")
    return model, module.GreedyCappedBaseStockPolicy(model), features


def prepare_run(environment, config, output, world_size=1):
    if version("dynaplex") != DYNAPLEX_VERSION:
        raise ValueError(f"Install the pinned dynaplex=={DYNAPLEX_VERSION} from requirements.txt")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    experiment = {
        key: config[key] for key in ("model", "dcl", "network", "train", "execution", "generations")
    }
    record = {
        "environment": environment,
        "config": experiment,
        "world_size": world_size,
        "source_sha256": source_hash(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("dynaplex", "numpy", "torch")},
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }
    identity = {
        key: record[key] for key in ("environment", "config", "world_size", "source_sha256", "packages")
    }
    record["identity"] = digest(identity)
    path = output / "run_config.json"
    if path.exists():
        if json.loads(path.read_text()).get("identity") != record["identity"]:
            raise ValueError("This directory records a different run; select a new --output directory")
    else:
        if any(output.glob("agent_gen*")) or (output / "dcl.json").exists():
            raise ValueError("Existing research artifacts lack this trainer's provenance; use a new output")
        write_json(path, record)
    return output


def saved_agent(path):
    import dynaplex as dp

    path = Path(path)
    if not path.exists():
        return None
    if not all((path / name).is_file() for name in ("agent.json", "weights.pt")):
        raise ValueError(f"Incomplete checkpoint {path}; preserve it and select a fresh output directory")
    return dp.NNAgent.load(str(path))


def fit_and_save(samples, config, path):
    import dynaplex as dp
    import torch

    execution = config["execution"]
    torch.set_num_threads(execution["fit_threads"])
    kwargs = {"seed": config["dcl"].get("seed", 0), "verbose": True, "device": execution["device"]} | config[
        "train"
    ]
    agent = dp.train_network(samples, network=dp.MLP(**config["network"]), **kwargs)
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    agent.save(str(temp))
    saved_agent(temp)
    temp.rename(path)
    return agent


def run_local(environment, config, output):
    import dynaplex as dp
    import torch

    started = time.perf_counter()
    output = prepare_run(environment, config, output)
    model, policy, features = model_parts(environment, config)
    execution = config["execution"]
    shape = dp.Shape(**{key: execution[key] for key in ("processes", "engines", "workers", "slots")})
    experiment = dp.DCL(
        model,
        policy,
        features=features,
        **config["dcl"],
        network=dp.MLP(**config["network"]),
        train=config["train"],
        gen0=shape,
        trained=shape,
        machine=None,
        device=execution["device"],
        torch_threads=execution["torch_threads"],
        workdir=str(output),
        sample_info=True,
        jit=True,
    )
    print(f"Collection: {shape.describe()}; device={execution['device']}", flush=True)
    print(f"Artifacts: {output}", flush=True)
    stages = []
    for generation in range(config["generations"]):
        path = output / f"agent_gen{generation + 1}"
        if saved_agent(path) is not None:
            print(f"{path.name} exists; skipping", flush=True)
            stages.append({"generation": generation + 1, "resumed": True})
            continue
        samples_path = output / f"samples_gen{generation}"
        collected_at = time.perf_counter()
        if samples_path.exists():
            samples = dp.SampleSet.load(str(samples_path))
        else:
            torch.set_num_threads(execution["torch_threads"])
            samples = experiment.collect(generation)
            samples.save(str(samples_path))
        collected_seconds = time.perf_counter() - collected_at
        fitting_at = time.perf_counter()
        fit_and_save(samples, config, path)
        stages.append(
            {
                "generation": generation + 1,
                "collect_or_load_seconds": collected_seconds,
                "fit_and_save_seconds": time.perf_counter() - fitting_at,
            }
        )
    result = {
        "elapsed_seconds": time.perf_counter() - started,
        "stages": stages,
        "checkpoint": f"agent_gen{config['generations']}",
        "checkpoint_saved": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    record_timing(output, result)
    print(f"Ready: {output / result['checkpoint']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    return result
