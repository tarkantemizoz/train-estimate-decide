"""Run a small generation-0 DCL collection, for the memory test in tests/test_models.py.

    python tests/peak_memory.py tests/collect.py <environment> <samples> <instance>

The instance applies to the single-instance environment; the super-MDPs use the paper's dimensions.
"""

import sys
import tempfile

import dynaplex as dp

import envs

name, samples, instance = sys.argv[1], int(sys.argv[2]), sys.argv[3]
environment = envs.find(name)
if name == "lost_sales":
    mdp = environment.MDP(**environment.mdp_config(instance))
else:
    mdp = environment.MDP(max_p=100.0, max_demand=12.0, max_leadtime=10)
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
