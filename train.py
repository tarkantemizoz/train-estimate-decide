"""Train a policy with DynaPlex's deep controlled learning (DCL), for any environment in envs/.

    python train.py lost_sales/zero_shot_stationary demo
    python train.py lost_sales/single_instance demo --instance online-003

DynaPlex recognizes the machine: a laptop, or a Snellius Genoa node or H100 unit requested with machines/*.sbatch.
Outside Slurm, a Linux machine DynaPlex does not recognize uses its laptop plan, derived from this machine.
The run is saved in envs/<environment>/train_runs/<machine>/<config>/, one agent_gen<g> folder per generation.
The config's "mdp" section holds the keyword arguments of the MDP: the training ranges of a super-MDP,
or the parameters of a single instance. --instance replaces them with those of a named instance, for an
environment that exports mdp_config(instance_id), and saves the run in <config>-<instance>/.
Running the same command again resumes it; DynaPlex refuses to reuse the folder for a changed experiment.
"""

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

import dynaplex as dp

import envs


def read(path: Path) -> tuple[str, dict]:
    """A JSON file's name for the run folder, without .json or .local.json, and its contents."""
    return path.name.removesuffix(".json").removesuffix(".local"), json.loads(path.read_text())


def plan_name(machine: str) -> str:
    """The DynaPlex plan of a machine name; "auto" falls back to the laptop plan outside a Slurm job."""
    try:
        return dp.machine.plan(machine).name
    except dp.machine.MachineMismatchError:
        # DynaPlex recognizes laptops on macOS and Windows only; a Slurm allocation must match a preset.
        if machine != "auto" or "SLURM_JOB_ID" in os.environ:
            raise
        return dp.machine.plan("laptop").name


class Tee:
    """Copy console output to the run's train.log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("environment", choices=envs.names())
    parser.add_argument("config", help="name in <environment>/configs (e.g. demo), or a JSON path")
    parser.add_argument(
        "--machine",
        default="auto",
        help="a DynaPlex machine: auto (default), laptop, genoa or gpu_h100_quarter; "
        "or a JSON file of your own settings, see docs/training.md",
    )
    parser.add_argument(
        "--instance", help="train on this instance, such as online-003, instead of the config's"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show DynaPlex's execution plan without training"
    )
    args = parser.parse_args()
    environment = envs.find(args.environment)
    folder = Path(environment.__file__).parent
    config_name, config = read(
        Path(args.config) if args.config.endswith(".json") else folder / "configs" / f"{args.config}.json"
    )
    if args.machine.endswith(".json"):
        machine_name, machine = read(Path(args.machine))
    else:
        try:
            machine_name = plan_name(args.machine)
        except (dp.machine.MachineMismatchError, ValueError) as error:
            parser.error(str(error))
        machine = {"machine": machine_name}
    if "mdp" not in config:
        parser.error(f'{args.config} has no "mdp" section with the keyword arguments of the MDP')
    if args.instance:
        if not hasattr(environment, "mdp_config"):
            parser.error(f"{args.environment} has no named instances; --instance does not apply")
        try:
            config = config | {"mdp": environment.mdp_config(args.instance)}
        except KeyError as error:
            parser.error(error.args[0])
        config_name = f"{config_name}-{args.instance}"
    mdp = environment.MDP(**config["mdp"])
    options = config["dcl"] | {
        key: dp.Shape(**value) if key in ("gen0", "trained") else value for key, value in machine.items()
    }

    def experiment(workdir) -> dp.DCL:
        return dp.DCL(
            mdp,
            environment.BasePolicy(mdp),
            features=environment.Featurizer,
            network=dp.MLP(**config["network"]),
            train=config["train"],
            workdir=str(workdir),
            **options,
        )

    if args.dry_run:
        with tempfile.TemporaryDirectory() as workdir:
            dcl = experiment(workdir)
        print(
            dcl.plan.describe()
            if dcl.plan is not None
            else "No machine plan: DynaPlex's simple CPU defaults."
        )
        fitting = dcl.train.get("device", dcl.train_device)
        print(
            f"generation 0: {dcl.gen0}; later generations: {dcl.trained} on {dcl.device}; fitting on {fitting}"
        )
        return
    workdir = folder / "train_runs" / machine_name / config_name
    dcl = experiment(workdir)
    for name, value in (("config.json", config), ("machine.json", machine)):
        (workdir / name).write_text(json.dumps(value, indent=2) + "\n")
    with (workdir / "train.log").open("a") as log:
        with (
            contextlib.redirect_stdout(Tee(sys.stdout, log)),
            contextlib.redirect_stderr(Tee(sys.stderr, log)),
        ):
            dcl.run(generations=config["generations"])


if __name__ == "__main__":
    main()
