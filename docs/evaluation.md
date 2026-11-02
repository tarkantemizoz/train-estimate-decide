# Evaluation

This guide runs evaluations, explains what they compare and how, and how to read the results.
[Environments](environments.md) describes the instances and the estimated modes, and [Training](training.md) produces the agents.

## Running an evaluation

```sh
python evaluate.py lost_sales/zero_shot_stationary --agent laptop/demo
python evaluate.py lost_sales/zero_shot_stationary --agent reference/demo --cases case1 --modes known
python evaluate.py lost_sales/zero_shot_large --agent paper/gc_lsn --instances case3-000 case3-120
python evaluate.py lost_sales/single_instance --agent laptop/demo
python evaluate.py lost_sales/zero_shot_stationary
```

The last command evaluates only the heuristics.
The first, the paper's protocol on the 16 online instances, took 9 minutes on an Apple M5 Pro.

| Option | Meaning | Default |
| --- | --- | --- |
| `--agent` | A run in the environment's `train_runs/`, whose last generation is used, an `agent_gen<g>` folder in it, or a path | None: heuristics only |
| `--cases` | `case1`, `case2`, `case3`, `online` | `online`; all four for `zero_shot_large`; a `single_instance` agent's own instance |
| `--instances` | Instance IDs such as `case1-000 online-003` | All instances of the cases |
| `--modes` | `known`, `demand`, `leadtime`, `both` | Every mode an instance is evaluated in that the environment simulates |
| `--trajectories` | Simulated trajectories per policy | 1000 |
| `--horizons` | The periods `T` scored with estimated parameters | 200 500 1000 2000 |
| `--known-horizon`, `--warmup` | The periods scored, and the unscored periods before them, with known parameters | 5000, 100 |
| `--seed` | The random seed | 13021984 |
| `--output` | The results folder | Derived from `--agent` |

An environment evaluates the cases its model can represent: the stationary environment cannot run Case 2 or Case 3.
A `single_instance` agent is evaluated on the instance in its config only.
That is one of the paper's instances when the parameters match one exactly, and otherwise an instance named `custom`, with its results in `custom.csv`.
The defaults are the paper's protocol.

## What is compared

Every evaluation scores four policies on each instance and mode:

- **greedy**: the environment's base policy, which orders the most its newsvendor cap allows, and from which DCL starts.
  It has the same information as the network: in an estimated mode, its cap follows the estimates.
- **trained**: the network, with the instance's parameters or their estimates as inputs.
- **BS**: a base-stock policy, which orders up to a fixed inventory position.
- **CBS**: a capped base-stock policy, which also never orders more than a fixed cap.

BS and CBS are the paper's benchmarks.
Their levels are tuned for each instance with its **true parameters**, and they keep those levels in every mode.
In an estimated mode they therefore know what the network and the online methods must learn: they are a reference with more information, not a competitor on equal terms.
Their orders are also not capped by the newsvendor bound that limits greedy and the network.

The tuned levels are in [`benchmark_parameters.csv`](../envs/lost_sales/cases/benchmark_parameters.csv); `evaluate.py` tunes and stores a missing instance on first use, and `python -m envs.lost_sales.cases.tune --cases case2` tunes a whole case.
The levels of a `custom` instance are tuned the same way, and stored in a `benchmark_parameters.csv` next to its results.
The tuning simulates 1000 trajectories of 5000 periods after 100 periods of warm-up, on a seed other than the evaluation's.
It raises the base-stock level until the cost stops improving, and for CBS, at each level from the BS level up, it raises the cap until the cost stops improving: the legacy search, kept as in the paper.
Stopping at the first cap that does not improve can miss a better one.
On `case3-160`, the cost dips slightly near a cap of 10 before falling further, so the stored CBS costs about 0.15 per period, 0.4%, more than the best level and cap within the search's range, and more than BS.
Trying every cap on 46 instances, the 16 online ones and 10 random ones from each other case, found the stored CBS the best on all of them, so such gaps are rare, but a small margin over CBS can still partly reflect the search.

BS and CBS run on `zero_shot_large`, whichever environment is evaluated, because it represents every instance: with the paper's dimensions, widened for a `custom` instance outside them.

## The protocol

- **Known parameters** (mode `known`): each policy runs 1000 trajectories, and scores its average cost per period over 5000 periods after 100 unscored periods of warm-up.
- **Estimated parameters** (modes `demand`, `leadtime`, `both`): each policy starts cold, with no stock, no orders in transit and no observations, and scores its average over the first `T` periods, for `T` = 200, 500, 1000 and 2000.
  This is the cumulative average from the first period, as in the online learning literature, so the cost of learning counts in full.

The score is a cost per period, `h` times the stock left plus `p` times the lost demand, except with estimated demand (modes `demand` and `both`): there it is the negative of the profit per period, `h` times the stock left minus `p` times the sales, the online learning literature's objective.
Lower is better in both.
The `true_cost` column always reports the cost.
On the same trajectory, cost and negative profit differ by `p` times the demand, which does not depend on the policy, so the difference between two policies is the same under both.

All policies share the random numbers of a seed: the same demand and arrivals in the same trajectory, across environments too.
Differences between policies are therefore paired, and far more precise than the standard errors of the two scores suggest.
`evaluate.py` reports each policy's difference with CBS, averaged over the trajectories of the window, with its own standard error.
On `online-012` with known parameters, for example, the shipped laptop network's cost and CBS's both have a standard error of 0.021, and their difference, 0.074 in the network's favour, one of 0.003.
With order crossover the pairing holds too, since every policy uses as many random numbers, whatever it orders; see [Environments](environments.md#common-random-numbers).

## Results and resuming

Results go to `<environment>/eval_runs/<run>/gen<g>/<case>.csv`, or to `eval_runs/benchmarks/` without `--agent`, one row per instance, mode, horizon and policy:

| Column | Meaning |
| --- | --- |
| `instance`, `mode` | The instance and mode |
| `horizon`, `warmup` | The periods scored, and the unscored periods before them |
| `policy` | `greedy`, `trained`, `BS` or `CBS` |
| `mean`, `error` | The score, averaged over trajectories, and its standard error |
| `delta_cbs`, `delta_cbs_error` | The score minus CBS's on the same trajectory, averaged over trajectories, and its standard error; 0 for CBS |
| `true_cost` | The average cost per period, also with estimated demand |

`settings.json` in the same folder records the protocol.
Repeating an interrupted command resumes it, skipping the windows already in the file.
`evaluate.py` refuses to add results with other settings, or to a file with other columns, from an older `evaluate.py`; choose another folder with `--output`.

After each case, `evaluate.py` prints, for each mode, horizon and policy, the mean score over the instances, and against CBS the mean difference and two counts: the instances where the policy is better, and where it is worse, with the 95% confidence interval of the difference entirely below or above zero.
Each count tests every instance separately: among 16 instances where two policies perform the same, about one may still count by chance.
For the online case, it also counts the horizons at which the network's profit beats the best published online method, and how many of those wins hold with the network's 95% lower bound against the rounded published score.

## The online comparison

[`online_reference.json`](../envs/lost_sales/cases/online_reference.json) holds, for each online instance and horizon, the profit of the best of the six online learning methods of Lyu, Zhang and Xin, as the paper's Table 2 reports it, and the source of each value.
Read the count with these limits in mind:

- The published values are rounded to 0.1 and have no standard errors, so the lower-bound count accounts for the network's sampling error and the rounding, not for theirs.
- Each value is the best of six methods, chosen after the fact for each instance and horizon, which favours the benchmark.
- The published values come from other simulations, so the comparison is not paired.
- For `online-013` at `T` = 500, the paper reports 65.5 and the source study 65.0; the file keeps 65.5 and records the discrepancy.
- The estimator here also learns whether demand exceeded the stock, which counts a period whose demand equals the stock exactly as uncensored.
  Check what the online methods observe before reading a small margin as a difference between methods.

## Reference results

`python evaluate.py lost_sales/zero_shot_stationary --agent reference/demo` evaluates the shipped laptop network on the 16 online instances.
Averaged over the instances, with known parameters, cost per period, lower is better:

| greedy | trained | BS | CBS |
| ---: | ---: | ---: | ---: |
| 22.737 | 18.342 | 19.255 | 18.241 |

With estimated demand, profit per period over the first `T` periods, higher is better:

| `T` | greedy | trained | BS, true parameters | CBS, true parameters |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 51.592 | 54.744 | 54.296 | 55.684 |
| 500 | 52.297 | 55.814 | 55.154 | 56.331 |
| 1000 | 52.543 | 56.222 | 55.459 | 56.553 |
| 2000 | 52.678 | 56.470 | 55.633 | 56.686 |

The network beats the best published online method in 62 of the 64 instance and horizon combinations, all 62 with its 95% lower bound above the rounded score; the median margin is 1.4 per period.
The two misses are `online-012`, geometric demand with `p = 10` and lead time 1, at `T = 200` (68.61 against 69.4, standard error 0.19) and `T = 500` (69.69 against 69.7, standard error 0.12).

The laptop network is not the paper's network.
With known parameters it is 0.6% costlier than CBS on average, while the paper reports that GC-LSN beats CBS.
It is costlier than CBS on 11 of the 16 instances and cheaper on the other 5, each beyond its paired 95% confidence interval.
With estimated demand it beats BS with the true parameters at every horizon, and comes within 0.4% of CBS with the true parameters at `T = 2000`.

Your own `demo` run will differ, since DCL's samples depend on the machine's shape.
Trained with the default plan's 3 workers instead of 12, the network costs 18.242 with known parameters, on par with CBS on average: cheaper on 7 instances, costlier on 8.
It earns 54.808, 55.848, 56.237 and 56.476 with estimated demand, and wins the same 62 of the 64 combinations.
Two runs say little about the spread over many, though.

On a single instance, the agents can be compared with the instance-specific one, here on `online-012` with known parameters: the average cost per period, with standard errors of about 0.02, and the paired difference with CBS, with standard errors of at most 0.007:

| Policy | Cost | Difference with CBS |
| --- | ---: | ---: |
| greedy | 30.127 | +0.932 |
| BS | 29.461 | +0.267 |
| CBS | 29.194 | |
| `single_instance` agent, 2 minutes of training on this instance | 29.081 | -0.113 |
| `zero_shot_stationary` reference network | 29.120 | -0.074 |
| GC-LSN | 29.054 | -0.140 |

All three networks are cheaper than CBS on the same trajectories.
The generally capable networks come within 0.04 of the agent trained for this one instance, about 0.1% of the cost, which is the point of TED: GC-LSN slightly below it, the stationary reference network slightly above.
These gaps are several times their standard errors, but each network comes from a single training run; on other instances, and against a longer training of the specific agent, they can differ.

## Running long evaluations

The full paper protocol over all cases is long: 819 instances, up to four modes, and four horizons in each estimated mode.
Evaluate on Snellius with `sbatch machines/genoa.sbatch evaluate.py lost_sales/zero_shot_large --agent paper/gc_lsn`, and restrict local runs with `--cases`, `--instances`, `--modes` or fewer `--trajectories`.

An evaluation uses a few hundred MiB whatever its size.
In DynaPlex 1.14 that takes two measures, because the models allocate in every trajectory's initial state and, with estimated parameters, every period, and DynaPlex neither counts that garbage towards its sweeps nor frees a heuristic's garbage within a trajectory.
`evaluate.py` and the BS and CBS tuning make every engine sweep at each safe point, with `sweeping_at_safe_points` in [`envs/__init__.py`](../envs/__init__.py).
With estimated parameters, the heuristics they run also reach a safe point every period: `SweepingBasePolicy`, BS and CBS, in each environment's `policies.py`.
Results are bit-identical with and without these measures, and tests check that memory does not grow with the trajectories or the horizon.
