"""Environments, grouped by problem family: envs/<family>/<environment>/.

An environment is a package with a configs/ folder that exports MDP, Featurizer and BasePolicy.
train.py and evaluate.py find it by its name, which is unique across families.
"""

import contextlib
from importlib import import_module
from pathlib import Path
from types import ModuleType

import dynaplex as dp

FOLDER = Path(__file__).resolve().parent


@contextlib.contextmanager
def sweeping_at_safe_points():
    """Free DynaPlex's garbage at every GC safe point while comparing policies.

    DynaPlex 1.14 does not count the garbage of model code towards its sweeps, so PolicyComparer never freed
    it. The lost-sales estimates allocate every period, and memory grew by 4 to 7 MiB per trajectory of 1000
    periods, past 10 GiB for the paper's protocol; with stochastic lead times, the initial state leaves about
    190 KiB per trajectory, and tuning BS and CBS grew by 190 MiB per assessment. PolicyComparer sets the GC
    threshold of every engine it builds, and here each engine then also sweeps at every safe point: before
    each trajectory, each batched step of a network, and each decision of a heuristic that calls gc(), as the
    lost-sales SweepingBasePolicy does. Results are bit-identical; run times are unchanged.
    """
    original = dp.Engine.set_gc_threshold

    def set_gc_threshold(engine, num_bytes: int) -> None:
        original(engine, num_bytes)
        engine.set_gc_force_at_safe_points(True)

    dp.Engine.set_gc_threshold = set_gc_threshold
    try:
        yield
    finally:
        dp.Engine.set_gc_threshold = original


def names(family: str = "*") -> list[str]:
    """The environments of a family, or of all families."""
    return sorted(path.parent.name for path in FOLDER.glob(f"{family}/*/configs"))


def find(name: str) -> ModuleType:
    """The environment package with this name."""
    folders = [path.parent for path in FOLDER.glob(f"*/{name}/configs")]
    if len(folders) != 1:
        raise KeyError(f"unknown environment {name!r}; choose from {', '.join(names())}")
    return import_module(f"{__name__}.{folders[0].parent.name}.{name}")
