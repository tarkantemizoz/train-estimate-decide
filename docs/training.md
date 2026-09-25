# Training locally and on Snellius

The common entry point is `python train.py ENVIRONMENT CONFIG.json`.
It supports `lost_sales_stationary` and `zero_shot_lost_sales_inventory_control`.
The selected live tutorial uses the stationary environment, N=100,000, one generation, M=500, and rollout horizon 21.
No smaller environment was necessary.

## Installation

Use Python 3.13 for this tested tutorial environment.
Create and activate a virtual environment before installing dependencies.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m dynaplex.selftest
```

On Windows, use `py -3.13 -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell.
DynaPlex is pinned to 1.14.0, NumPy to 2.5.3, and PyTorch to 2.14.0.
The project can also be installed with `python -m pip install -e .` after the pinned requirements.

For a Linux CPU-only installation, first install the CPU wheel:

```sh
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For a compatible NVIDIA system, the corresponding PyTorch 2.14 CUDA wheel is at `https://download.pytorch.org/whl/cu130`.
It requires a compatible driver; the Snellius GPU job checks an actual CUDA operation before starting training.
Apple Silicon uses the ordinary macOS PyTorch wheel and MPS.

## Commands and configurations

Run these commands from the repository root.

```sh
python train.py lost_sales_stationary configs/training/smoke.json
python train.py lost_sales_stationary configs/training/laptop.json
python train.py zero_shot_lost_sales_inventory_control configs/training/gc_lsn_power.json
```

The smoke configuration exercises two very small generations and is only a functionality check.
The laptop configuration is the selected demonstration budget.
The power configuration for GC-LSN requests five generations of five million samples on the larger Super-MDP.
The one-GPU example uses one million samples per generation to fit its smaller CPU/controller allocation.
That is a starting configuration for substantial training, not a claim that the new implementation reproduces the paper's results automatically.
Use the paper's reported GC-LSN results when explaining Cases 1, 2, and 3, and identify them as results of the heavily trained paper agent.

| Configuration | Environment | Samples per generation | Generations | Target |
| --- | --- | ---: | ---: | --- |
| `configs/training/laptop.json` | Stationary | 100,000 | 1 | Local automatic CPU/GPU |
| `configs/training/power.json` | Stationary | 5,000,000 | 5 | Local, if the machine has sufficient resources |
| `configs/training/gc_lsn_power.json` | Full GC-LSN | 5,000,000 | 5 | Local, if the machine has sufficient resources |
| `configs/training/snellius_genoa.json` | Full GC-LSN | 5,000,000 | 5 | One Genoa node |
| `configs/training/snellius_mpi.json` | Full GC-LSN | 5,000,000 total across ranks | 5 | Two Genoa nodes |
| `configs/training/snellius_gpu.json` | Full GC-LSN | 1,000,000 | 5 | One H100 GPU and 16 CPU cores |

Model-specific parameters in the full configurations require the full environment name.
To run the stationary model remotely, copy the laptop or stationary power configuration and add the desired `workstation` block.
The training budget is specified once, in `dcl.n` and `generations`; workstation choice does not silently change it.

Inspect a configuration without training, submitting a job, or connecting to SSH:

```sh
python train.py lost_sales_stationary configs/training/laptop.json --dry-run
python train.py zero_shot_lost_sales_inventory_control configs/training/snellius_mpi.json --dry-run
```

The local dry run prints resolved hardware settings.
The remote dry run prints a Slurm script with illustrative paths; actual paths are recorded when submitting.

## Configuration fields

The JSON file contains the model prior, DCL parameters, network architecture, fitting options, execution settings, workstation, and output directory.
The example files expose every hyperparameter used in the selected laptop experiment.

| Field | Meaning |
| --- | --- |
| `model` | Super-MDP parameter bounds; training requires `evaluate=false`. |
| `dcl.n` | Number of labeled states per generation, globally across MPI ranks. |
| `dcl.m` | Rollout replication budget per candidate action, allocated by sequential halving. |
| `dcl.h` | Rollout horizon. |
| `dcl.warmup_time` | Periods before collecting labels on a fresh trajectory. |
| `dcl.restart_after` | Labels before restarting a main trajectory. |
| `dcl.reduction_factor`, `dcl.final_field` | Sequential-halving tournament settings. |
| `dcl.seed` | Fixed experiment and fitting seed. |
| `generations` | Number of policy-improvement generations from the heuristic. |
| `network.hidden` | Hidden-layer widths. |
| `train` | Cross-entropy loss, epochs, batch size, learning rate, early stopping, and validation settings. |
| `execution.device` | `auto`, `cpu`, `cuda`, or `mps`; used for fitting and learned-policy rollouts unless `train.device` explicitly overrides fitting. |
| `execution.processes` | Collection processes per node; `auto` respects CPU and RNG capacity. |
| `execution.workers` | Simulation workers per process; `auto` distributes the allowed CPUs. |
| `execution.slots` | Rollout slots per simulation worker; 512 by default. |
| `execution.torch_threads` | PyTorch threads per collection process; `auto` avoids separate all-core pools. |
| `execution.fit_threads` | PyTorch CPU threads during fitting; `auto` uses the allowed CPU allocation. |
| `output` | Local artifact directory; `--output` overrides it. |

Automatic device selection tries CUDA, then MPS, then CPU.
An explicitly requested unavailable device raises an error rather than silently changing the experiment.
Environment simulation always uses CPUs.
The CPU resolver respects operating-system affinity and `SLURM_CPUS_PER_TASK`, and includes all available CPU types rather than relying on the native laptop performance-core classification.
It assigns the full CPU worker budget across processes; process coordinators and the operating system also need scheduling time, so this does not mean every phase sustains 100% CPU utilization.
Many-core CPU nodes use smaller process groups for neural inference.
The portable path uses one engine per process to avoid completed engines continuing to consume seed space while another engine finishes.

DynaPlex 1.14 gives each controller a finite random-stream band.
The resolver checks sample and generation capacity before collecting and rejects configurations that cannot fit the available layout.
A large power configuration can therefore fail on a small laptop and should be moved to a larger allocation or given a smaller sample budget.
Changing the controller layout or numerical device can change the learned policy even with the same seed.
The earlier approximately five-minute laptop result and 62/64 online comparison wins are measurements of a particular run, not guarantees for every participant's hardware or training realization.
The 25-minute objective covers training through a saved policy, excluding installation and evaluation.

## Snellius setup and submission

SSH authentication must already work without an interactive password or MFA prompt for each subprocess.
Use your existing SSH configuration, keys, agent, or institution-supported authenticated session.
The tutorial never reads private keys or stores passwords.
Check connectivity with `ssh YOUR_USERNAME@snellius.surf.nl` before running the command.
The local machine needs OpenSSH `ssh` and `scp` on PATH.

Copy a remote example to a personal configuration, then set `workstation.host` to your own account.
Files ending in `.local.json` are ignored by Git.
Set `workstation.account` if your Slurm account must be specified explicitly; otherwise Slurm uses your default.

```sh
cp configs/training/snellius_genoa.json configs/training/my_genoa.local.json
# Edit host, account if required, training budget, and output.
python train.py zero_shot_lost_sales_inventory_control configs/training/my_genoa.local.json
```

The command performs these steps:

1. Connect through SSH and create a dedicated directory under `workstation.remote_root`, defaulting to `~/ted-tutorial`.
2. Upload only the Python source and dependency files, addressed by their content hash; unchanged source is reused.
3. Create or reuse a dedicated virtual environment, load the recorded Python/MPI modules, install the exact requirements, and run `pip check`.
4. Report installed and latest PyPI DynaPlex versions while retaining the tested pin.
5. Save the resolved submission configuration and Slurm script locally, then submit the job with `sbatch`.
6. Poll Slurm, collect its accounting record, download logs and policy artifacts, and reload the final checkpoint locally.

Existing results, `.venv`, and personal files are not uploaded.
Changing source creates a new release directory instead of overwriting code used by an active job.
Installation and synchronization happen on the login node; simulation and fitting run inside the Slurm allocation.
The default module stack is `2025`, `Python/3.13.5-GCCcore-14.3.0`, and `OpenMPI/5.0.8-GCC-14.3.0`, verified on the account during preparation.
Override the `modules` list and `python` command if SURF changes the module catalogue.

The three profiles have these resource defaults:

| Profile | Partition | Nodes | CPUs per node | GPUs | Memory per node |
| --- | --- | ---: | ---: | ---: | --- |
| `genoa` | `genoa` | 1 | 192 | 0 | 336G |
| `mpi` | `genoa` | 2 | 192 | 0 | 336G |
| `gpu` | `gpu_h100` | 1 | 16 | 1 | 64G |

Override `partition`, `nodes`, `cpus`, `memory`, `time`, and `account` in the workstation block as appropriate.
For an A100 quarter node, use `partition="gpu_a100"` and `cpus=18`.
Queue waiting, first-time installation, and data transfer are outside the training timer.
The full example jobs request 12 hours of wall time; this is a resource request, not a measured runtime forecast.

By default, the command waits and retrieves results.
Use `--no-wait` to return after submission, then rerun the same command to reconnect.
Use `--fetch` to download an existing completed job without polling.
Interrupting the local waiting command leaves the remote job running.
If cancellation is desired, use `ssh YOUR_USERNAME@snellius.surf.nl scancel JOB_ID` with the recorded ID.
A failed job is not automatically resubmitted; inspect its downloaded logs and use a fresh output directory for the corrected run.

Local submission artifacts include `remote_job.json`, `submitted_config.json`, `job.sbatch`, and `slurm-accounting.txt`.
Downloaded files are under `OUTPUT/remote/`; checkpoints are under `OUTPUT/remote/artifacts/agent_genG`.
Samples remain remotely available for resuming and are excluded from download by default.
Set `workstation.fetch_samples=true` to retrieve them as well.

## What MPI means here

DynaPlex 1.14's native DCL driver is single-node, with local multiprocessing.
The tutorial's MPI adapter uses its public collection, SampleSet, and network-training interfaces.
Each rank collects a share of N on one node, then rank zero combines the samples, fits one network, and publishes its checkpoint to the shared filesystem.
Every rank loads that same policy for the next generation.
Fitting is centralized; this is not distributed gradient training, and the CPU nodes wait while rank zero fits.
The single-GPU profile similarly uses one GPU for fitting.

All ranks retain the same experiment seed and controller shape.
The native collection generation coordinate is `logical_generation * number_of_ranks + rank`, creating disjoint controller seed bands.
Simply adding the rank to the global seed would not establish disjoint streams because the native seed combiner uses XOR.
The aggregate records each original shard manifest, its actual collection coordinate, its hash, and a separate rank column in sample provenance.
Its logical generation is used to name the single resulting policy.
The guard checks `generations * ranks * controllers_per_rank <= 4096` in addition to each controller's sample capacity.
MPI and single-node collection therefore have explicitly different sample identities.

## Saved experiments and reproducibility

Each local run saves `training.log`, `run_config.json`, native `dcl.json`, `samples_gen0`, and `agent_gen1`, with additional numbered artifacts for later generations.
`timing.json` preserves the first completed invocation, including collection/loading and fitting/saving times.
`latest_invocation.json` records the most recent invocation, so a quick resume does not replace the original measurement.
MPI runs instead record per-rank shards and aggregate samples; Slurm captures their logs.
Re-running an unchanged command skips complete checkpoints and can reuse completed sample sets.
The trainer checks the environment, resolved configuration, package versions, and source hash before reusing a run directory.
An incompatible run requires a new output directory.
Policy publication uses a temporary directory followed by rename, and every saved and downloaded final policy is reloaded as a completion check.

The paper's heavily trained GC-LSN checkpoint is not included.
Use its published results as paper results, and evaluate any newly trained checkpoint independently with the [experiment commands](experiments.md).
