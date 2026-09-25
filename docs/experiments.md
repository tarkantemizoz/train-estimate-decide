# Experiments and interpretation

## The online-learning demonstration

Run the training and evaluation commands in the [README](../README.md).
The case catalogue creates the instances at runtime, so no exported per-instance JSON files are needed.
There are 16 instances: Poisson or geometric demand with mean 10, p in {5, 10}, and deterministic lead time in {1, 3, 5, 7}.
The stationary training prior remains broader than this test grid: it varies demand mean and standard deviation, p, and lead time within the bounds in `configs/training/laptop.json`.

Each evaluation starts with empty on-hand inventory and pipeline, unknown demand, and known lead time.
The estimator uses censored demand observations while the trained network stays fixed.
The initial demand estimate is deterministic demand of 12, matching the paper protocol.
At each T in {200, 500, 1000, 2000}, the score is average profit from period 1 through T, with no warmup.
Per-period profit is `p * units_sold - ending_on_hand`, with unit holding cost.
Higher profit is better.

The demand estimator preserves the paper implementation's redistribution rule, which is an approximation rather than an exact Kaplan-Meier estimator.
It assigns unresolved tail mass to the largest observation and distinguishes exact depletion from excess demand using an excess-demand indicator.
That distinction requires information beyond sales alone; the exercise therefore preserves this observation convention rather than establishing performance under a strictly sales-only protocol.
Simulator-only true demand is available for scoring, but the policy's demand features use estimated moments during censored deployment.

The evaluator runs the trained policy and a greedy capped-base-stock heuristic on matched exogenous demand trajectories.
It saves individual trajectory profits in `.npz` files, each window's results in JSON, and `summary.csv`.
An unchanged invocation can reuse completed windows; a changed protocol or checkpoint requires a fresh output directory.
To check the evaluation pipeline quickly, use a separate output directory and a small subset:

```sh
python -m ted_tutorial.evaluate_online --agent runs/tutorial_laptop/agent_gen1 --output runs/online_check --instances paper-000 paper-015 --horizons 200 --trajectories 20
python -m ted_tutorial.compare_online runs/online_check
```

This small check is insufficient for a performance claim.

### Reference and uncertainty

[`configs/evaluation/online_reference.json`](../configs/evaluation/online_reference.json) contains the rounded best-online scores from [TED Table 2](https://arxiv.org/html/2411.00515v3#S7.T2), the paper's GC-LSN-E scores, source provenance, and protocol assumptions.
The comparison command reports how many simulated means exceed those reference targets, together with gaps and pointwise uncertainty checks.
It does not rerun the original online-learning methods.
Confidence intervals describe this trained policy's simulation uncertainty; they exclude benchmark estimation error and uncertainty across training runs.
Paired gains against the included greedy heuristic are available, but paired tests against the published online methods are unavailable without their raw runs.
The 64 intervals are not simultaneous confidence intervals.

One source discrepancy is retained explicitly: for geometric demand, p=10, L=3, T=500 (`paper-013`), TED reports 65.5 while the underlying online-method table's maximum is 65.0.
The default comparison retains TED's 65.5; the reference JSON records both values.

## Known and estimated parameters

Use the same stationary checkpoint on a fixed instance first with known demand, then with demand inferred from censored observations:

```sh
python -m lost_sales_stationary.evaluate --config configs/evaluate.json --agent runs/tutorial_laptop/agent_gen1 --output runs/tutorial_laptop/known.json
python -m lost_sales_stationary.evaluate --config configs/evaluate.json --agent runs/tutorial_laptop/agent_gen1 --censored --output runs/tutorial_laptop/estimated.json
```

The defaults use 100 warmup periods followed by 5,000 scored periods when parameters are known, and 200 scored periods without warmup when demand is inferred.
These illustrate different information settings and evaluation horizons, so their raw objectives should not be compared directly.
Known-demand evaluation reports inventory cost; censored-demand evaluation reports negative profit as its learning objective and separately reports true inventory cost.
Use `--horizon`, `--warmup`, `--trajectories`, `--workers`, and `--seed` to set a specific protocol.

For a checkpoint trained on the full environment, use `python -m zero_shot_lost_sales_inventory_control.evaluate --config configs/zero_shot/evaluate.json --agent PATH`.
Add `--censored` to estimate demand, `--censored-leadtime` to estimate stochastic lead times, or both.
Stationary and full-environment checkpoints have different feature contracts and are not interchangeable.

## Larger Case 1-3 experiments

Train a full-environment checkpoint using `configs/training/gc_lsn_power.json` or a Snellius configuration from the [training guide](training.md).
The supplied large budgets are experiment starting points; they have not been shown to reproduce the paper's heavily trained GC-LSN results.

The shared case catalogue defines 320 Case 1, 243 Case 2, and 240 Case 3 instances.
Case 1 uses stationary demand and deterministic lead times; Case 2 introduces cyclic demand; Case 3 uses stochastic lead times, including settings with order crossover.
Inspect the selected instances without loading a checkpoint:

```sh
python -m zero_shot_lost_sales_inventory_control.evaluate_cases --case all --list
```

After full-environment training, check the pipeline on one instance per case before requesting the full experiment:

```sh
python -m zero_shot_lost_sales_inventory_control.evaluate_cases --case case1 case2 case3 --agent runs/tutorial_gc_lsn/agent_gen5 --smoke
python -m zero_shot_lost_sales_inventory_control.evaluate_cases --case case1 case2 case3 --agent runs/tutorial_gc_lsn/agent_gen5
```

Edit [`configs/zero_shot/evaluate_cases.json`](../configs/zero_shot/evaluate_cases.json) to choose the checkpoint, instance IDs, simulation budget, output directory, or benchmark settings.
The `legacy` mode follows the paper's information settings: known and inferred demand in Cases 1 and 2, and all four combinations of known/inferred demand and lead time in Case 3.
Explicit modes `known`, `demand`, `leadtime`, and `both` are available when applicable to the selected cases.
Use `--help` for case filtering and smoke options.
These full-case runs can be much longer than the live online demonstration.

## Reusable BS and CBS benchmarks

Optimize base-stock (BS) and capped-base-stock (CBS) parameters and evaluate their known-parameter costs:

```sh
python -m ted_tutorial.build_benchmarks --cases case1
```

Run Cases 2 and 3 separately when time permits:

```sh
python -m ted_tutorial.build_benchmarks --cases case2 case3
```

Defaults match the intended large benchmark protocol: 1,000 trajectories, 100 warmup periods, 5,000 scored periods, tuning seed 1122, and independent evaluation seed 13021984.
The optimizer follows the paper's search procedure and bounds; finite simulation does not certify a global optimum.
The cache is stored in `runs/benchmark_cache`, parameter/cost tables in `runs/benchmark_tables/CASE/table.csv`, and phase timings in each case's `timings/` folder.
Timing reports distinguish wall time from summed task and policy-search times.
Precomputed tables are not included in the template.

Parameter reuse checks model dynamics, search protocol, and implementation provenance.
Cached cost reuse additionally requires matching parameters, seed, warmup, horizon, and trajectory count; matching the seed alone is insufficient.
Changing a protocol requires a new table output directory while compatible cache entries can still be reused.
Enable `--benchmarks` in the full-case evaluator to use this cache with matching settings and tune missing entries.
BS/CBS are known-parameter benchmarks; their information advantage must be stated when comparing them with a policy that estimates unknown parameters.
Their oracle action mode also permits orders outside the learned policy's action mask.
