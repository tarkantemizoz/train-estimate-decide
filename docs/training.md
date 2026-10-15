# Training with DynaPlex

Run training from the project directory with one configuration file:

```sh
python train.py lost_sales_stationary configs/training/laptop.json
```

Edit that same `laptop.json` to change the experiment.
The selected tutorial budget is 100,000 samples, one generation, M=500, and H=21.
The command constructs the environment and calls DynaPlex's `DCL(...).run(...)`.
DynaPlex selects the hardware plan, collects samples, fits networks, saves policies, and resumes completed generations.

## Installation

Use Python 3.13 and the pinned dependencies:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m dynaplex.selftest
```

On Windows, use `py -3.13 -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell.
DynaPlex is pinned to 1.14.0, NumPy to 2.5.3, and PyTorch to 2.14.0.
For Linux CPU installations, install the CPU PyTorch wheel first with `python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`.
The CUDA wheel index for this PyTorch version is `https://download.pytorch.org/whl/cu130` and requires a compatible NVIDIA driver.

## One configuration, native defaults

| JSON section | Passed to |
| --- | --- |
| `model` | The selected environment constructor |
| `dcl` | `dynaplex.DCL(...)` keyword arguments |
| `network` | `dynaplex.MLP(...)`, passed as DCL's `network` |
| `train` | DCL's `train` argument |
| `generations` | `DCL.run(generations=...)`; required |
| `output` | DCL's `workdir`; optional, and overridden by `--output` |
| `workstation` | Optional Snellius connection and job submission settings |

Omitted DCL, network, and fitting options retain DynaPlex's defaults.
The tutorial does not select worker counts, devices, or thread counts itself.
There is no custom `execution` section.
The example keeps the chosen research hyperparameters explicit so that changing hardware does not change the training budget.

Inside `dcl`, the standard laptop setting is:

```json
"machine": "auto"
```

Omitting `machine` has the same effect, because `"auto"` is DynaPlex's default.
DynaPlex 1.14 recognizes laptops automatically on macOS and Windows.
On a Linux desktop, set `"machine": "laptop"` explicitly; native `auto` recognizes cluster allocations on Linux.
`"machine": null` disables the plan and uses DynaPlex's simple CPU defaults, as in the small smoke example.

For an explicit generation-0 override, add this inside `dcl` in the same file:

```json
"gen0": {"processes": 3, "workers": 4}
```

Only those two fields override the native plan.
Unspecified fields retain the plan's values.
Remove `gen0` to return to the fully automatic plan.
Use `trained` for collection with the learned policy in later generations.
Both objects accept DynaPlex's `processes`, `engines`, `workers`, and `slots` fields.
Other native overrides include `dcl.device`, `dcl.torch_threads`, and `train.device`.
There is no separate tutorial `fit_threads` default.

Inspect the exact resolved plan and native experiment record without training:

```sh
python train.py lost_sales_stationary configs/training/laptop.json --dry-run
```

This calls DynaPlex to resolve the configuration; it does not implement another hardware planner.
See the official [DCL guide](https://github.com/DynaPlex/DynaPlex/blob/main/docs/training/dcl.md) and [machine-plan guide](https://github.com/DynaPlex/DynaPlex/blob/main/docs/training/machines.md).
The installed pinned version is authoritative where the current documentation differs.

## Saved runs

With the supplied laptop configuration, the policy is saved in `runs/tutorial_laptop/agent_gen1`.
DynaPlex writes `dcl.json`, samples, and numbered agent directories, and handles resuming.
The tutorial additionally records `training.log`, the requested JSON, the resolved execution plan, and total elapsed time in `timing.json`.
The timing covers training through saved and reloaded policies; installation and evaluation are excluded.
A subsequent resume records its duration in `latest_invocation.json` without replacing the first completion time.

DynaPlex rejects an explicitly named work directory containing an incompatible experiment.
Use `--output runs/my_experiment` for a fresh run.
If both `output` and `dcl.workdir` are omitted, DynaPlex creates its normal experiment-specific directory.
Changing a collection shape can change the sample realization and learned policy even with the same seed.
Earlier five-minute and 62/64 reference results describe the original training configuration, not a guarantee for the native automatic plan.

## Larger experiments

| Example | Environment | Samples per generation | Generations | Native plan |
| --- | --- | ---: | ---: | --- |
| `laptop.json` | Stationary | 100,000 | 1 | `auto` |
| `smoke.json` | Either | 128 | 2 | No plan, small CPU check |
| `power.json` | Stationary | 5,000,000 | 5 | `genoa` with explicit shapes |
| `gc_lsn_power.json` | Full GC-LSN | 5,000,000 | 5 | `genoa` with explicit shapes |
| `snellius_genoa.json` | Full GC-LSN | 5,000,000 | 5 | `genoa` with explicit shapes |
| `snellius_mpi.json` | Full GC-LSN | 5,000,000 total across nodes | 5 | `genoa` on each node |
| `snellius_gpu.json` | Full GC-LSN | 1,000,000 | 5 | `gpu_h100_quarter` with explicit shapes |

All example files are in `configs/training`.
Use `zero_shot_lost_sales_inventory_control` for the full-model configurations.
The large single-node examples explicitly distribute samples across more processes because DynaPlex 1.14 limits each controller's random-stream capacity.
Those overrides are visible configuration values; the tutorial does not calculate a replacement plan.
DynaPlex's native runtime checks remain authoritative.
These profiles are starting configurations, not measured scaling results or reproductions of the paper's trained GC-LSN agent.

## Snellius

Set `workstation.target` to `"snellius"`, `workstation.host` to your SSH account, and select profile `genoa`, `mpi`, or `gpu`.
The supplied remote examples contain those fields.
SSH authentication must already work for `ssh YOUR_USERNAME@snellius.surf.nl`.
The command uploads the tutorial code, installs the pinned requirements, submits a Slurm job, and retrieves policies and logs.
Hardware planning inside each node is still handled by DynaPlex from `dcl.machine` and any explicit overrides.

| Submission profile | Default allocation |
| --- | --- |
| `genoa` | One Genoa node, 192 CPUs, 336G memory |
| `mpi` | Two Genoa nodes, 192 CPUs and 336G memory per node |
| `gpu` | One H100 GPU, 16 CPUs, 64G memory |

Override host, account, partition, nodes, CPUs, memory, and wall time in `workstation` as needed.
Keep the native machine plan consistent with the requested allocation.
The module defaults are `2025`, `Python/3.13.5-GCCcore-14.3.0`, and `OpenMPI/5.0.8-GCC-14.3.0`.
Queue waiting, installation, and transfers are outside the training timer.
A remote `--dry-run` prints the submission script without SSH.
Use `--no-wait` to return after submission and `--fetch` to retrieve an existing job.
Downloaded policies are under `OUTPUT/remote/artifacts/agent_genG`.

Multi-node collection still needs the separate MPI adapter because DynaPlex 1.14's native DCL loop is single-node.
Each rank uses a native DCL plan, collects its share of samples, and rank zero fits one shared network.
MPI requires `dcl.sample_info=true` to retain rank and sample provenance.
The adapter keeps disjoint collection-generation coordinates and validates the shard manifests before merging them.
It does not implement hardware-selection rules or distributed gradient training.
