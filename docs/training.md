# Training

This guide installs the software, trains each environment with deep controlled learning (DCL), and runs training on a laptop or on the Snellius supercomputer.
[Environments](environments.md) describes what is trained, and [Evaluation](evaluation.md) how to test the result.

## Installation

Install Python 3.13, then, from the repository root:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m dynaplex.selftest
```

On Windows, use `py -3.13 -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell.

`requirements.txt` pins the tested versions of DynaPlex, NumPy and PyTorch.
On Linux, `pip` installs the CUDA build of PyTorch by default, a download of several gigabytes.
Without an NVIDIA GPU, install the CPU build first, then the requirements:

```sh
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Training an environment

`train.py` takes an environment and a config, a file in the environment's `configs/` folder:

```sh
python train.py zero_shot_lost_sales_stationary demo
python train.py lost_sales demo --instance online-012
sbatch machines/genoa.sbatch train.py zero_shot_lost_sales_large gc_lsn
```

| Environment | Configs | What a run produces |
| --- | --- | --- |
| `zero_shot_lost_sales_stationary` | `demo`: 100,000 samples, one generation | A network for every stationary instance in the training ranges |
| `zero_shot_lost_sales_large` | `gc_lsn`: the paper's 5 million samples per generation, five generations | GC-LSN, for every instance of the paper |
| `lost_sales` | `demo`, with `--instance <ID>` | An agent for that one instance |

Every environment also has a `smoke` config, a few seconds of training that the tests use.
A config can also be a path: `python train.py zero_shot_lost_sales_stationary my_configs/wide.json`.

Measured on an Apple M5 Pro with the default plan, which runs 3 workers there, the `demo` config of `zero_shot_lost_sales_stationary` took 236 seconds, 224 of them to collect the samples, and the single-instance `demo` for `online-012` took 139 seconds.
With 12 workers, as the shipped `reference/demo` was trained, they took 80 and 45 seconds; see [Machines](#machines).
`gc_lsn` is 250 times the samples of `demo`, which is why it runs on Snellius.

## Configs

A config is a JSON file with these sections:

```json
{
  "mdp": {"max_p": 100.0, "max_demand": 12.0, "max_leadtime": 10},
  "dcl": {"n": 100000, "m": 500, "h": 21, "warmup_time": 100, "restart_after": 100, "seed": 0},
  "network": {"hidden": [256, 128, 128, 128]},
  "train": {"epochs": 100, "batch_size": 1024, "lr": 0.001, "patience": 15, "val_every": 1},
  "generations": 1
}
```

- `mdp`: the MDP's keyword arguments.
  For a super-MDP they are the training ranges: the largest penalty, mean demand and lead time an instance can have, and, for `zero_shot_lost_sales_large`, the longest demand cycle and whether lead times can be stochastic and demand cyclic.
  The ranges are part of the network: an instance outside them asks it to extrapolate, and `evaluate.py` builds every instance with the ranges of the agent's config.
  The single-instance `lost_sales` has no `mdp` section; `--instance` defines its model.
- `dcl`: the sampling budget of each generation, passed to DynaPlex's DCL.
  `n` decision states are labeled per generation.
  To label a state, DCL takes every admissible order once and then follows the current policy for `h` periods, averaged over up to `m` rollouts, and labels the state with the cheapest order.
  Orders that are clearly worse are dropped after fewer rollouts.
  Before labeling, each trajectory runs `warmup_time` periods under the current policy, and after `restart_after` labels it restarts from a fresh instance.
- `network`: the hidden layers of the multilayer perceptron.
- `train`: fitting the network to the labels with cross-entropy: at most `epochs` passes, stopping early once the validation loss has not improved for more than `patience` epochs.
- `generations`: generation 0 rolls out the environment's `BasePolicy`, a greedy capped base-stock policy; each later generation rolls out the previous network.

The paper's GC-LSN used `n` = 5 million, `m` = 500, `h` = 21 and five generations: the `gc_lsn` config.
The [DynaPlex DCL guide](https://github.com/DynaPlex/DynaPlex/blob/main/docs/training/dcl.md) and the [DCL paper](https://www.sciencedirect.com/science/article/pii/S0377221725000463) explain the algorithm.

## Runs and resuming

A run is saved in `envs/<family>/<environment>/train_runs/<machine>/<config>/`, and a single-instance run in `.../<config>/<instance>/`:

```text
agent_gen1/, agent_gen2/, ...   the trained networks: weights.pt and agent.json
samples_gen0/, ...              the labeled samples of each generation
config.json, machine.json       the settings the run used
dcl.json                        DynaPlex's record of the experiment
train.log                       the console output
```

`evaluate.py --agent <machine>/<config>` evaluates the last generation of a run, such as `--agent laptop/demo`.

Running the same command again resumes an interrupted run, or adds generations after you increase `generations`.
DynaPlex refuses to continue a run whose experiment changed, such as the MDP, the budget or the network, with a `WorkdirMismatchError`.
To change the experiment, copy the config under a new name.
Runs stay out of git; to share a trained network, add its `agent_gen<g>/` and `config.json` with `git add -f`.

## Machines

DynaPlex chooses how to run DCL with a *machine plan*: how many processes, engines and worker threads collect samples, and where the network runs.
`--machine` selects it, and the run folder is named after it:

- `auto`, the default, recognizes the machine: the `laptop` plan on macOS and Windows, and on Snellius the plan of the allocation, such as `genoa` for a Genoa node.
  On another Linux machine, outside a Slurm job, `train.py` uses the `laptop` plan, which DynaPlex derives from the machine's cores and accelerator.
- A plan name: `laptop`, `genoa`, `gpu_h100_quarter`, `gpu_h100`, and the other [DynaPlex plans](https://github.com/DynaPlex/DynaPlex/blob/main/docs/training/machines.md).
- A JSON file with your own settings, to override parts of a plan.

`python train.py <environment> <config> --dry-run` prints the plan without training.

A settings file overrides the plan's shapes for generation 0 (`gen0`) and for the trained network (`trained`), with any of `processes`, `engines`, `workers` and `slots`, or other DCL options such as `device`.
For example, DynaPlex's `laptop` plan counts only 5 of the 15 cores of an Apple M5 Pro and runs 3 workers, as [DynaPlex issue 11](https://github.com/DynaPlex/DynaPlex/issues/11) reports.
To use 12 workers, as for the reference network, create `machines/laptop_m5pro.local.json`:

```json
{
  "machine": "auto",
  "gen0": {"processes": 3, "workers": 4}
}
```

```sh
python train.py zero_shot_lost_sales_stationary demo --machine machines/laptop_m5pro.local.json
```

The run folder takes the file's name without `.json` and `.local`: `laptop_m5pro`.
Files ending in `.local.json` stay out of git, since they describe your machine.
A shape is part of DynaPlex's experiment record, so runs with different settings are different experiments, with slightly different samples.

## Snellius

On the Snellius supercomputer, run the one-time setup on a login node, from the repository root:

```sh
module load 2025 Python/3.13.5-GCCcore-14.3.0
python -m venv .venv && .venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -r requirements.txt
```

Then submit any repository command with one of the two batch scripts:

```sh
sbatch machines/genoa.sbatch train.py zero_shot_lost_sales_large gc_lsn
sbatch machines/gpu.sbatch train.py zero_shot_lost_sales_large gc_lsn
sbatch machines/genoa.sbatch evaluate.py zero_shot_lost_sales_large --agent genoa/gc_lsn
```

`genoa.sbatch` requests a whole 192-core Genoa node and `gpu.sbatch` a quarter H100 node: one GPU and 16 cores.
DynaPlex recognizes either allocation, so no `--machine` is needed.
Both request 12 hours; change `--time` in the script for longer runs, and resubmit the same command to resume.
The output goes to `slurm-<job>.out`, and the run's `train.log`.

## Memory

DCL collection and evaluation should use about the same memory whatever their size.
The tests check this for every environment, because DynaPlex 1.14 does not free every kind of garbage on its own; see [Adding your own environment](environments.md#adding-your-own-environment).
