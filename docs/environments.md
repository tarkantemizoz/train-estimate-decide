# Environments

This guide describes the inventory problem, the three environments that model it, the instances they are evaluated on, how to turn an MDP into a super-MDP, and how to add an environment of your own.
[Training](training.md) and [Evaluation](evaluation.md) cover the commands.

## The problem

All three environments model the same periodic-review lost-sales inventory system with a single item.
At the start of each period you place an order, which arrives after a lead time.
Demand that cannot be met from stock is lost.
Each unit left in stock at the end of a period costs a holding cost `h = 1`, and each unit of lost demand costs a penalty `p`.
The goal is the lowest long-run average cost per period.

The timing within a period is:

1. Observe the state: stock on hand and the orders in transit.
2. Place an order of `0` up to a cap, which the action mask enforces.
3. Receive an order placed with lead time 0, then meet demand from stock, losing any excess.
4. Pay the holding cost on the stock left and the penalty on the lost units, then add the orders that arrive at the end of the period.

An order placed in period `t` with lead time `L >= 1` can therefore meet demand from period `t + L` on.
The order cap is a newsvendor bound: the `p / (p + h)` fractile of demand at arrival caps the order, and the same fractile of demand over the lead time plus one period caps the inventory position.

Lost sales with positive lead times have no simple optimal policy, which is why heuristics such as base-stock (BS) and capped base-stock (CBS) policies, and learned policies, are studied.

## From one instance to a super-MDP

An *instance* fixes the parameters: `p`, the demand distribution of each period, and the lead time or lead-time distribution.
Deep controlled learning (DCL) normally trains one network for one instance.

A *super-MDP* draws an instance at the start of every trajectory and keeps its parameters in the state.
The network receives the parameters as inputs, next to the inventory state, so one network learns to act on every instance in the training ranges.
This is the *Train* step of TED: the network is trained once, on the true parameters of many sampled instances.
The 100,000 samples of the `demo` config, for example, come from at least 1,000 trajectories, each with its own instance.

DCL itself does not change.
What makes the agent generally capable is the MDP, which draws the instance in `get_initial_state`, and the config's `restart_after`, the paper's `R`: the number of samples after which a trajectory restarts with a newly drawn instance.
[Turning an MDP into a super-MDP](#turning-an-mdp-into-a-super-mdp) lists what the MDP needs.
A config's `mdp` section therefore holds the training ranges of a super-MDP, and the parameters of a single instance.

In operation the true parameters are unknown.
The *Estimate* step replaces them with estimates from the observations so far, and the *Decide* step feeds those estimates to the same network.
The network is never retrained, and it never saw estimation error during training: it treats the estimates as if they were the truth.

## The three environments

All three live in [`envs/lost_sales/`](../envs/lost_sales).
Each folder holds `model.py` (the MDP and its featurizer), `policies.py` (heuristics), `configs/` (training settings), and, once you run them, `train_runs/` and `eval_runs/`.

| | `zero_shot_stationary` | `zero_shot_large` | `single_instance` |
| --- | --- | --- | --- |
| Role | The workshop super-MDP | The paper's super-MDP, for GC-LSN | One instance, known parameters |
| Demand | Stationary | Cyclic, cycles of 1 to 7 periods | Stationary or cyclic |
| Lead time | Fixed | Fixed or stochastic, with or without order crossover | Fixed or stochastic |
| Network inputs | 14 | 38 | The pipeline, plus the cycle position if cyclic |
| Orders | 0 to 129 | 0 to 129 | Up to the instance's own cap |
| Cases it covers | Case 1, online | All | All, one instance per agent |
| Modes | known, demand | known, demand, leadtime, both | known |
| Trains on | A laptop, in about 4 minutes | A Snellius node: 5 million samples, 5 generations | A laptop, in about 2 minutes |

### `zero_shot_stationary`

Each training trajectory draws an instance:

- the penalty `p` uniformly from `[2, max_p]`;
- the lead time uniformly from `{0, ..., max_leadtime}`;
- the mean demand uniformly from `[2, max_demand]`;
- the standard deviation uniformly from the smallest value a discrete distribution with that mean can have up to twice the mean.

Demand follows the Adan-Eenige-Resing (AER) fit to that mean and standard deviation: a Poisson distribution, or a mixture of two binomial, negative binomial or geometric distributions, with exactly those two moments.
The configs use the paper's ranges, `max_p = 100`, `max_demand = 12` and `max_leadtime = 10`.

The network sees 14 numbers: `p`, the 10 pipeline positions (stock on hand first, then the orders in transit), the lead time, and the mean and standard deviation of demand.
The action space has 130 orders, `0` to `129`: the cap for the most extreme instance in the training ranges.

### `zero_shot_large`

This is the paper's environment for the generally capable lost-sales network GC-LSN.
Each training trajectory draws:

- the penalty `p` uniformly from `[2, 100]`;
- a demand cycle of 1 to 7 periods, each period with its own mean and standard deviation, drawn as above;
- a shortest lead time uniformly from `{0, ..., 10}` and a longest uniformly from the shortest up to 10;
- when they differ, a lead-time distribution: uniform over that range with probability 0.33, and otherwise either an AER fit or random weights;
- when lead times are stochastic, order crossover with probability 0.5.

With order crossover, each unit in transit arrives independently, so orders can overtake each other.
Without it, an arriving order brings every older order along.

The network sees 38 numbers: whether orders can cross, `p`, the 10 pipeline positions, the 11 lead-time probabilities, the cycle length, and the mean and standard deviation of the next 7 periods of the cycle.

### `single_instance`

This environment is what TED avoids: one agent per instance, trained with that instance's true parameters, which it never sees as inputs.
The config holds the parameters, those of `online-012` in `demo`, and `--instance` takes those of another of the paper's instances instead; the agent applies to that instance only.
It simulates exactly like `zero_shot_large` on that instance, so its agent can be compared, on the same random numbers, with the generally capable networks.

## Estimated parameters

In an estimated mode a trajectory starts cold: no stock, no orders in transit, no observations.

- **demand** (and **both**): the demand estimate starts as a point mass at `ceil(max_demand)`, 12.
  From the first period with stock on hand, every period adds an observation: the sales, and whether demand exceeded the stock.
  A period in which demand exceeded the stock gives a censored observation: demand was at least the sales.
  The estimate redistributes the probability of each censored value over the larger values observed, in one pass in the spirit of Kaplan-Meier.
  This is the legacy DynaPlex estimator, kept as it is; it is not the exact Kaplan-Meier estimator.
- **leadtime** (and **both**): the lead-time estimate starts as a point mass at the longest lead time, 10.
  It becomes the relative frequency of the lead times observed in arrivals.
  Orders outstanding for longer than the longest lead time observed so far count towards the longer lead times not yet seen.

In `zero_shot_large`, periods of the cycle that share a demand distribution pool their observations.
The cycle and which periods share a distribution are known; only the distributions are estimated.

Every period, the estimates update the network's inputs and the order caps.
The demand estimator observes whether demand exceeded the stock, which is more than the sales alone reveal when the stock sells out exactly.

## The paper's instances

[`envs/lost_sales/cases/`](../envs/lost_sales/cases) defines the instances every environment is evaluated on, with IDs such as `case1-000` or `online-003`.
Demand of each period is the AER fit to the mean and standard deviation of the named distribution family.
For Poisson and geometric demand the fit is the distribution itself; for binomial and negative binomial demand it matches the first two moments.
`h = 1` throughout.

| Case | Instances | Demand | Lead times | `p` | Modes |
| --- | ---: | --- | --- | --- | --- |
| `case1` | 320 | Stationary; mean 3, 5, 7 or 10; binomial, Poisson, negative binomial or geometric | Fixed: 2, 4, 6, 8 or 10 | 9, 39, 69, 99 | known, demand |
| `case2` | 243 | Cyclic, 3, 5 or 7 periods, 3 mean patterns each; low, high or mixed variability | Fixed: 3, 6 or 9 | 9, 39, 69 | known, demand |
| `case3` | 240 | Cycles of 1, 3, 5 or 7 periods, mixed variability | 10 distributions on 0 to 10, with and without crossover | 9, 39, 69 | known, demand, leadtime, both |
| `online` | 16 | Poisson or geometric, mean 10 | Fixed: 1, 3, 5 or 7 | 5, 10 | known, demand |

The online instances are those of the online learning study by Lyu, Zhang and Xin in *Operations Research* that the paper compares with in its Table 2.
[`online_reference.json`](../envs/lost_sales/cases/online_reference.json) holds, for each of them, the profit of the best of the study's six online learning methods, and its provenance.

Every instance lies within the training ranges of both super-MDPs, except that the stationary environment cannot represent cyclic demand or stochastic lead times.
An instance outside the training ranges would ask the network to extrapolate, and the paper reports clearly weaker results there.

## Common random numbers

The three environments simulate every instance they share identically for the same seed: the same demand, the same arrivals, the same costs under the same orders.
The tests check this.
A comparison of two policies on the same seed, even across environments, is therefore paired: the difference reflects the policies, not the luck of the draw.
`evaluate.py` reports each policy's paired difference with CBS; see [Evaluation](evaluation.md#the-protocol).

Pairing needs every policy to use as many random numbers, whatever it orders.
With order crossover, each age in transit takes one random number per period, and the number of its units that arrive is the binomial quantile of that number: the same distribution as one draw per unit, since units of one age are interchangeable.
The legacy model drew one number per unit, padded to the largest order the action mask allows, so BS and CBS, which may order more, drew more numbers, and their demand soon differed from the other policies'.
The tests check that the benchmarks now see the same demand, and that one draw gives exactly the binomial distribution.

## The shipped networks

- `zero_shot_stationary/train_runs/reference/demo/`: the `demo` config trained on an Apple M5 Pro laptop, with its training log.
- `zero_shot_large/train_runs/paper/gc_lsn/`: the paper's GC-LSN, converted from the legacy C++ DynaPlex checkpoint.
  It was not trained with this repository; the `gc_lsn` config reproduces its training protocol.

## Turning an MDP into a super-MDP

DCL trains a generally capable agent with the same command as an agent for one instance: `python train.py <family>/<environment> <config>`.
What differs is the MDP.
Compare [`single_instance/model.py`](../envs/lost_sales/single_instance/model.py), an MDP of one instance, with [`zero_shot_stationary/model.py`](../envs/lost_sales/zero_shot_stationary/model.py), a super-MDP of the same problem:

| | One instance | Super-MDP |
| --- | --- | --- |
| The config's `mdp` section | The instance's parameters | The training ranges, such as `max_p` |
| The parameters | Attributes of the MDP, such as `self.p` | Fields of the `State`, such as `state.p` |
| `get_initial_state` | An empty pipeline | Draws an instance, then an empty pipeline |
| Network inputs | The pipeline | The pipeline and the parameters |
| Actions | Up to the instance's own cap | Up to the cap of the most extreme instance in the ranges, masked per state |
| One fixed instance | Always | With `evaluate=True` and its parameters, for evaluation |
| Estimated parameters | No | The estimates take the place of the parameters in the state |

A super-MDP needs:

1. **The parameters in the state.**
   Whatever varies between instances moves from the MDP to the `State`, and the transitions and costs read it from there: `state.p * lost` instead of `self.p * lost`.
   The MDP keeps what every instance shares, such as the holding cost `h` and the training ranges.
2. **An instance drawn in `get_initial_state`**, with `context.rng`, so that a run is reproducible from its seed.
   How to draw is a modelling choice: the agent learns where it is sampled, so a corner of the ranges that is rarely drawn is also rarely learned.
   `zero_shot_stationary` draws each parameter uniformly, as listed [above](#zero_shot_stationary).
3. **The parameters as network inputs.**
   `StationaryFeaturizer` writes `p`, the pipeline, the lead time, and the mean and standard deviation of demand.
   Without them, the network cannot tell the instances apart and learns one compromise policy for all of them.
4. **The same shapes for every instance.**
   The network has one input size and one output size, so the ranges set both, not the instance drawn.
   The pipeline has `max_leadtime` positions whatever the lead time, and with the configs' ranges there are 130 actions, enough for the most extreme instance.
   `write_action_validity` masks the orders above `state.order_constraint`, which keeps each instance to its own orders.
5. **A base policy that works on every instance.**
   Generation 0 rolls it out on every instance drawn, so it reads the parameters from the state, as the greedy policy does with `state.order_constraint`.
6. **A mode for one fixed instance, for evaluation.**
   With `evaluate=True` and an instance's parameters, the same class simulates that instance, with the training ranges as its dimensions, so that the network's inputs and outputs keep their sizes.
   `instance_config` builds this model for a case instance, and `evaluate.py` passes it the ranges of the agent's config.
   With `censored_demand=True`, the model starts from a constant demand at the top of the ranges, re-estimates the distribution from each period's sales, and writes the estimated mean and standard deviation into the state where the true ones were: the *Estimate* step.
   The featurizer then feeds the estimates to the same network: the *Decide* step, without retraining.
7. **A config with ranges and `restart_after`.**
   The `mdp` section holds the ranges, the constructor's keyword arguments.
   `restart_after`, the paper's `R`, is the number of samples a trajectory contributes before it restarts with a newly drawn instance.
   A smaller `R` spreads the same `n` samples over more instances, with fewer samples from each.
   With `n` = 100,000 and `R` = 100, the `demo` config draws at least 1,000 instances; `gc_lsn`, with 5 million samples per generation, at least 50,000 per generation.
8. **`SUPER_MDP = True`** in the package's `__init__.py`, so that the lost-sales `evaluate.py` applies the agent to every selected instance, not only to the instance in its config.

## Adding your own environment

An environment is a package `envs/<family>/<name>/`, where `<family>` groups environments of one problem.
`train.py` and `evaluate.py` name it after its folder: `<family>/<name>`.
For training, its `__init__.py` exports:

- `MDP`, the DynaPlex MDP class;
- `Featurizer`, the class that writes the network's inputs;
- `BasePolicy`, the heuristic DCL starts from;
- optionally `mdp_config(instance_id)`, which returns the MDP's keyword arguments for a named instance, for `--instance`.

Its `configs/` folder holds the training settings, each with an `"mdp"` section of the MDP's keyword arguments; see [Training](training.md).
For a generally capable agent, the MDP is a super-MDP; see [Turning an MDP into a super-MDP](#turning-an-mdp-into-a-super-mdp).

For evaluation, the family has its own `evaluate.py` with a `main()`, which the root `evaluate.py` calls.
A new lost-sales environment can reuse [`envs/lost_sales/evaluate.py`](../envs/lost_sales/evaluate.py) by also exporting `CASES`, `DEFAULT_CASES`, `MODES`, `SUPER_MDP`, `DIMENSIONS`, `instance_config(instance, mode, dimensions)` and `SweepingBasePolicy`, as the three environments here do.
`SUPER_MDP` says whether an agent applies to every instance within the ranges of its config, or only to the instance in it.
`SweepingBasePolicy` is the base policy that, when the model estimates, also reaches a garbage-collection safe point every period, which DynaPlex 1.14 needs when a model allocates every period during evaluation.
A new family's `evaluate.py` should likewise compare policies inside `envs.sweeping_at_safe_points()`; see [Evaluation](evaluation.md#running-long-evaluations).

The DynaPlex documentation explains how to write an MDP: start with the [introduction to MDPs](https://github.com/DynaPlex/DynaPlex/blob/main/docs/getting-started/introduction-to-mdps.md) and the [airplane](https://github.com/DynaPlex/DynaPlex/blob/main/docs/tutorials/airplane-mdp.md) and [bin packing](https://github.com/DynaPlex/DynaPlex/blob/main/docs/tutorials/binpacking-mdp.md) tutorials, and see [featurizers](https://github.com/DynaPlex/DynaPlex/blob/main/docs/training/featurizers.md) for the network inputs.

Three lessons from the environments here:

- **Allocate nothing per period.**
  `modify_state_with_event` and `get_action` run billions of times during training.
  DynaPlex 1.14 frees garbage between DCL rounds, and only garbage it has counted triggers a sweep: a model that created one empty list per period grew by 6.4 MiB per sample until a laptop crashed.
  Create lists and arrays in `get_initial_state` and reuse them, as the crossover draws do in `zero_shot_large`.
  To check, measure the peak memory of training with `python tests/peak_memory.py train.py <environment> <config>` for two configs that differ only in `n`: it should not grow with `n`.
- **Draw as many random numbers, in the same order, whatever the policy does**, so that comparisons stay paired.
  With order crossover, the lost-sales models draw one number per age in transit rather than per unit, with [`distributions.py`](../envs/lost_sales/distributions.py); see [Common random numbers](#common-random-numbers).
- **Moving or renaming an environment breaks its saved agents.**
  Each `agent_gen<g>/agent.json` names its featurizer by module path, such as `envs.lost_sales.zero_shot_stationary.model.StationaryFeaturizer`, and `dcl.json` names the classes of its run.
  Update `agent.json` after a move, so that the agents still load; DynaPlex refuses to resume the training of a moved run, whose `dcl.json` no longer matches.
