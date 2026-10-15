"""Translate JSON options to native DynaPlex objects and call DCL.run()."""

import contextlib
import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def record_timing(output, result):
    write_json(output / "latest_invocation.json", result)
    if not (output / "timing.json").exists():
        write_json(output / "timing.json", result)


def model_parts(environment, config):
    module = importlib.import_module(f"{environment}.model")
    if environment == "lost_sales_stationary":
        model = module.StationaryLostSalesMDP(**config.get("model", {}))
        features = module.StationaryFeaturizer
    else:
        model = module.ZeroShotLostSalesMDP(**config.get("model", {}))
        features = module.ZeroShotFeaturizer
    if model.evaluate:
        raise ValueError("Training requires a super-MDP with evaluate=False")
    return model, module.GreedyCappedBaseStockPolicy(model), features


def build_experiment(environment, config, output=None):
    import dynaplex as dp

    model, policy, features = model_parts(environment, config)
    options = dict(config.get("dcl", {}))
    for name in ("gen0", "trained"):
        if name in options and options[name] is not None:
            options[name] = dp.Shape(**options[name])
    for name in ("network", "train"):
        if name in config:
            if name in options:
                raise ValueError(f"Specify {name} once, at the top level")
            options[name] = dp.MLP(**config[name]) if name == "network" else config[name]
    if output is not None:
        options["workdir"] = str(output)
    return dp.DCL(model, policy, features=features, **options)


def describe_experiment(experiment):
    return {
        "plan": experiment.plan.describe() if experiment.plan is not None else None,
        "gen0": dict(zip(("processes", "engines", "workers", "slots"), experiment.gen0.as_tuple())),
        "trained": dict(zip(("processes", "engines", "workers", "slots"), experiment.trained.as_tuple())),
        "device": experiment.device,
        "training_device": experiment.train.get("device", experiment.train_device),
        "torch_threads": experiment.torch_threads,
        "dcl": json.loads((Path(experiment.workdir) / "dcl.json").read_text()),
    }


def preview(environment, config):
    # DCL constructs the plan and resolves defaults; discard its temporary run record.
    with tempfile.TemporaryDirectory(prefix="ted-plan-") as directory:
        return describe_experiment(build_experiment(environment, config, directory))


def saved_agent(path):
    import dynaplex as dp

    return dp.NNAgent.load(str(path)) if Path(path).exists() else None


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


def run_local(environment, config, output=None):
    started = time.perf_counter()
    experiment = build_experiment(environment, config, output)
    output = Path(experiment.workdir)
    with (output / "training.log").open("a") as log:
        with (
            contextlib.redirect_stdout(Tee(sys.stdout, log)),
            contextlib.redirect_stderr(Tee(sys.stderr, log)),
        ):
            print(f"Artifacts: {output}", flush=True)
            write_json(output / "requested_config.json", config)
            write_json(output / "execution_plan.json", describe_experiment(experiment))
            try:
                agents = experiment.run(generations=config["generations"])
            except BaseException:
                traceback.print_exc()
                raise
            result = {
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint": f"agent_gen{config['generations']}",
                "checkpoint_saved": len(agents) >= config["generations"],
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
            record_timing(output, result)
            print(f"Ready: {output / result['checkpoint']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    return result
