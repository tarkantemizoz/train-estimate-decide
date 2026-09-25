# TED: Train, Estimate and Decide with DynaPlex

This tutorial is in preparation for the workshop.

Apply the TED idea from [Zero-shot generalization in inventory management: Train, then Estimate and Decide](https://doi.org/10.1016/j.ejor.2025.12.033) to lost-sales inventory control using Python-based DynaPlex 1.14.0.
Train one policy across a population of inventory problems, estimate unknown parameters from observations, and use those estimates with the fixed policy to choose orders.
The network is not retrained during deployment.

The live exercise uses stationary demand and known deterministic lead times.
It trains with **100,000 samples, one policy-improvement generation, M=500, and rollout horizon 21**.
Training took approximately five minutes on the preparation laptop (Apple M5 Pro, 15 CPU cores, MPS).
The reference run exceeded the rounded online-learning benchmark mean in 62 of 64 instance/checkpoint combinations.
These are measurements of one trained policy; runtime and scores can change with hardware and training randomness.
See the [experiment protocol](docs/experiments.md) before interpreting comparisons.

## 1. Create your copy

On the published GitHub template, select **Use this template > Create a new repository**.
Download or clone your newly created repository and open a terminal at its root.
Run all commands below from that directory.
An organizer preparing the template should first follow [publishing instructions](docs/publishing.md).

## 2. Install

Use Python 3.13:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m dynaplex.selftest
```

On Windows, use `py -3.13 -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell.
See [training and hardware options](docs/training.md#installation) for CPU and NVIDIA installations.

## 3. Check the installation, then train

```sh
python train.py lost_sales_stationary configs/training/smoke.json
python train.py lost_sales_stationary configs/training/laptop.json
```

The smoke run checks two tiny generations; it does not produce a competitive policy.
The laptop run uses all allowed CPU cores for simulation and automatically selects CUDA, MPS, or CPU for the network.
It saves the trained policy at `runs/tutorial_laptop/agent_gen1`, with logs, samples, configuration, and timings alongside it.
An unchanged command resumes completed work; use `--output runs/my_experiment` to start a separate run.
The 25-minute target covers training through a saved policy, excluding installation and evaluation.

## 4. Evaluate the online-learning instances

```sh
python -m ted_tutorial.evaluate_online --agent runs/tutorial_laptop/agent_gen1 --output runs/tutorial_laptop/online_evaluation
python -m ted_tutorial.compare_online runs/tutorial_laptop/online_evaluation
```

This evaluates 16 instances at 200, 500, 1,000, and 2,000 periods, using 1,000 trajectories, censored demand observations, and known deterministic lead times.
Evaluation chooses process and worker counts from the available CPU allocation.
Open `runs/tutorial_laptop/online_evaluation/comparison.csv` for the 64 comparisons, or `comparison.json` for the aggregate counts and interpretation notes.
`summary.csv` reports simulated profits, standard errors, and paired gains over the included greedy heuristic.
The online-learning algorithms themselves are represented by their published reference means.

## Explore further

- [Training guide](docs/training.md): configurations, local hardware, larger training runs, and Snellius CPU/MPI/GPU execution.
- [Experiment guide](docs/experiments.md): known versus estimated parameters, the online protocol, Case 1-3 experiments, and reusable BS/CBS benchmarks.
- [Publishing guide](docs/publishing.md): organizer steps for making this folder a GitHub template.

The stationary model covers the online instances and Case 1.
The larger `zero_shot_lost_sales_inventory_control` model also supports cyclic demand and stochastic lead times for Cases 2 and 3.
The paper's heavily trained GC-LSN results are separate from the laptop demonstration; no pretrained paper checkpoint is bundled here.

## Repository contents

| Path | Purpose |
| --- | --- |
| `train.py` | Common training command |
| `configs/` | Training examples, evaluation settings, and published online reference values |
| `ted_tutorial/` | Training, remote execution, and experiment commands |
| `lost_sales_stationary/` | Stationary environment and single-instance evaluator |
| `zero_shot_lost_sales_inventory_control/` | Full environment, case catalogue, and BS/CBS benchmark support |
| `docs/` | Participant and organizer guides |
| `tests/`, `.github/` | Small training and experiment checks, including CI |

Generated policies, samples, logs, and benchmark caches belong in the ignored `runs/` directory.

To run the included checks:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check ted_tutorial lost_sales_stationary zero_shot_lost_sales_inventory_control tests train.py
```
