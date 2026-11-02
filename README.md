# Train, then Estimate and Decide (TED) with DynaPlex

This repository accompanies the paper [Zero-shot generalization in inventory management: Train, then Estimate and Decide](https://doi.org/10.1016/j.ejor.2025.12.033) (Temizöz, Imdahl, Dijkman, Lamghari-Idrissi and Van Jaarsveld, *European Journal of Operational Research* 333(1), 2026).
You will train deep reinforcement learning agents for lost-sales inventory control with [DynaPlex](https://github.com/DynaPlex/DynaPlex), on your laptop, and test them on the paper's instances.

## Why TED?

Deep reinforcement learning (DRL) finds strong inventory policies, but it is usually trained for one problem instance: one demand distribution, one lead time, one cost structure.
In practice, those parameters are not given.
Demand and lead-time distributions have to be estimated from data, demand observations are censored when stock runs out, and the estimates change as data come in.
Retraining an agent each time an estimate changes, for each of thousands of items, is too expensive: a single training run can take hours.

Online learning methods avoid training altogether: they learn while they operate, often with worst-case guarantees.
But they tune the parameters of a heuristic, such as a base-stock level, so they can do no better than the best policy of that heuristic class.

## The idea

TED separates learning a policy from learning the parameters:

1. **Train** one *generally capable agent* on a *super-MDP*: a whole population of instances, with each instance's parameters as inputs to the network.
   Training happens once, offline, with Deep Controlled Learning (DCL).
2. **Estimate** the parameters while operating: censored demand with a Kaplan-Meier-type estimator, lead times from the observed arrivals.
3. **Decide** by feeding the current state and the latest estimates to the trained network.
   It is never retrained.

The training cost is paid once and shared by every instance and every update of the estimates.
The agent applies zero-shot to instances it has never seen, as long as they lie within its training ranges.
Any estimator can be plugged in, and better estimates immediately give better decisions.

In the paper, the resulting agent GC-LSN outperforms base-stock and capped base-stock policies when the parameters are known.
With estimated parameters, it outperforms the best of six online learning methods on nearly all of their benchmark instances.
The generalization has limits: the agent is only as general as its training ranges, and the paper reports weaker performance well outside them.

In this repository, a network trained on a laptop in a few minutes beats the best published online learning method in 62 of 64 instance and horizon combinations.
[Evaluation](docs/evaluation.md#the-online-comparison) explains what that comparison can and cannot show.

## What is in this repository

| Part | Contents |
| --- | --- |
| [`envs/lost_sales/`](envs/lost_sales) | Three lost-sales environments, the paper's instances, and their evaluation |
| `zero_shot_stationary` | A super-MDP of stationary instances: the workshop environment, trainable on a laptop |
| `zero_shot_large` | The paper's super-MDP, adding cyclic demand and stochastic lead times with order crossover |
| `single_instance` | One instance with known parameters, to train one agent per instance: what TED avoids |
| [`train.py`](train.py), [`evaluate.py`](evaluate.py) | Train any environment with DCL; evaluate agents against heuristics and published results |
| [`machines/`](machines) | Slurm scripts for training on the Snellius supercomputer |
| Shipped networks | `reference/demo`: the laptop network above; `paper/gc_lsn`: the paper's GC-LSN |

Read further in [docs/](docs):

- [Environments](docs/environments.md): the problem, the three environments, the paper's instances, and how to add your own environment.
- [Training](docs/training.md): installation, configurations, machines and Snellius.
- [Evaluation](docs/evaluation.md): the protocol, the benchmarks, and how to read the results.

## Quick start

On GitHub, select **Use this template > Create a new repository**, clone your new repository, and open a terminal at its root.
Install Python 3.13, then:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m dynaplex.selftest
```

On Windows, use `py -3.13 -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell.
On Linux without an NVIDIA GPU, install the CPU build of PyTorch first, as [Training](docs/training.md#installation) shows.

Train the workshop network, with 100,000 samples and one generation:

```sh
python train.py lost_sales/zero_shot_stationary demo
```

The run is saved in `envs/lost_sales/zero_shot_stationary/train_runs/<machine>/demo/`, where `<machine>` is the machine DynaPlex recognized, such as `laptop`.
Evaluate it on the 16 online learning instances, with known and with estimated demand:

```sh
python evaluate.py lost_sales/zero_shot_stationary --agent laptop/demo
```

The last line counts how often your network beats the best published online method.
Use `--agent reference/demo` for the shipped network, or leave out `--agent` to evaluate only the heuristics.

Then train an agent for a single instance, `online-012`, whose parameters the single-instance `demo` config holds, and compare it with the generally capable networks on that instance:

```sh
python train.py lost_sales/single_instance demo
python evaluate.py lost_sales/single_instance --agent laptop/demo
python evaluate.py lost_sales/zero_shot_large --agent paper/gc_lsn --instances online-012 --modes known
```

With `--instance online-003`, the first command trains on another of the paper's instances instead, in the run `laptop/demo-online-003`.

## Repository layout

```text
envs/                                 environments, by problem family
  lost_sales/
    cases/                            the paper's instances and the tuned BS and CBS benchmarks
    distributions.py                  the demand and arrival distributions the models share
    evaluate.py                       the evaluation protocol for these environments
    single_instance/                  each environment: model.py, policies.py, configs/,
    zero_shot_large/                    train_runs/ (trained agents) and eval_runs/ (results)
    zero_shot_stationary/
machines/                             sbatch scripts for Snellius
docs/                                 environments, training and evaluation guides
tests/                                checks run by GitHub Actions on Linux, macOS and Windows
train.py, evaluate.py
```

Your own runs in `train_runs/` and `eval_runs/` stay out of git; the shipped networks are the exception.
To run the checks:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

## License and citation

The code in this repository is available under the [MIT license](LICENSE).
DynaPlex is installed separately, under its own license.
If you use this work, please cite:

> Temizöz, T., Imdahl, C., Dijkman, R., Lamghari-Idrissi, D., & Van Jaarsveld, W. (2026).
> Zero-shot generalization in inventory management: Train, then Estimate and Decide.
> *European Journal of Operational Research*, 333(1), 153-173.
> https://doi.org/10.1016/j.ejor.2025.12.033
