"""Run a small generation-0 DCL collection, for the memory test in tests/test_models.py.

    python tests/peak_memory.py tests/collect.py <environment> <samples>

The MDP is that of the environment's smoke config.
"""

import json
import sys
import tempfile
from pathlib import Path

import dynaplex as dp

import envs

name, samples = sys.argv[1], int(sys.argv[2])
environment = envs.find(name)
config = json.loads((Path(environment.__file__).parent / "configs" / "smoke.json").read_text())
mdp = environment.MDP(**config["mdp"])
with tempfile.TemporaryDirectory() as workdir:
    dcl = dp.DCL(
        mdp,
        environment.BasePolicy(mdp),
        features=environment.Featurizer,
        network=dp.MLP(hidden=[16]),
        workdir=workdir,
        machine=None,
        gen0=dp.Shape(processes=1, engines=1, workers=1, slots=16),
        n=samples,
        m=500,
        h=21,
        warmup_time=100,
        restart_after=100,
        seed=0,
        verbose=False,
    )
    dcl.collect(0)
