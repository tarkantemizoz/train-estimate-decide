import math
import sys

import dynaplex as dp
import pytest
from conftest import ENVIRONMENTS, peak_mib

from envs.lost_sales import cases
from envs.lost_sales import single_instance as single
from envs.lost_sales import zero_shot_large as zero_shot
from envs.lost_sales import zero_shot_stationary as stationary
from envs.lost_sales.cases.tune import benchmarks, dimensions, table
from envs.lost_sales.distributions import arriving_units


def comparer(model, warmup: int = 100) -> dp.PolicyComparer:
    return dp.PolicyComparer(
        model, number_of_trajectories=8, horizon=300, warmup_time=warmup, seed=13021984, jit=True
    )


def greedy_mean(environment, config: dict, warmup: int) -> float:
    model = environment.MDP(**config)
    return comparer(model, warmup).assess(environment.BasePolicy(model)).mean


def test_the_environments_cover_their_cases_and_modes():
    reference = cases.online.published_scores()
    assert [len(cases.CASES[case]()) for case in ("case1", "case2", "case3", "online")] == [320, 243, 240, 16]
    assert {item.id for item in cases.CASES["online"]()} == set(reference)
    for environment in (single, stationary, zero_shot):
        for case in environment.CASES:
            for instance in cases.CASES[case]():
                for mode in set(instance.modes) & set(environment.MODES):
                    environment.MDP(**environment.instance_config(instance, mode, environment.DIMENSIONS))
    with pytest.raises(ValueError, match="cyclic demand"):
        stationary.instance_config(cases.find("case2-000"), "known", {})


def test_every_instance_has_benchmarks_tuned_with_the_paper_dimensions():
    instances = [item for case in cases.CASES.values() for item in case()]
    assert {item.id for item in instances} <= set(table())
    assert all(dimensions(item) == zero_shot.DIMENSIONS for item in instances)


# Shared instances must be the same problem in every model, so that comparisons across them are paired.
@pytest.mark.parametrize("instance_id", ["case1-137", "case2-045", "case3-066", "case3-195", "online-012"])
def test_the_single_instance_model_simulates_like_the_zero_shot_model(instance_id):
    instance = cases.find(instance_id)
    expected = greedy_mean(zero_shot, zero_shot.instance_config(instance, "known", zero_shot.DIMENSIONS), 100)
    assert greedy_mean(single, single.instance_config(instance, "known", {}), 100) == expected


# A count is the binomial quantile of the draw when every draw between the distribution function at count - 1
# and at count gives that count: then one uniform draw gives exactly the binomial distribution.
@pytest.mark.parametrize("units", [1, 7, 60, 300])
@pytest.mark.parametrize("probability", [1e-9, 0.03, 0.5, 0.71, 1 - 1e-9])
def test_one_draw_gives_the_binomial_number_of_arriving_units(units, probability):
    below = 0.0
    for count in range(units + 1):
        mass = math.comb(units, count) * probability**count * (1 - probability) ** (units - count)
        if mass > 1e-9:
            for share in (0.01, 0.5, 0.99):
                assert arriving_units(units, probability, below + share * mass) == count
        below += mass
    assert arriving_units(units, 0.0, 0.5) == 0 and arriving_units(units, 1.0, 0.5) == units


# BS and CBS may order more than the action mask allows, and with order crossover every unit in transit
# could arrive on its own: the arrivals must still use as many random numbers whatever the orders.
def test_the_benchmarks_see_the_demand_of_the_masked_policies_with_order_crossover():
    instance = cases.find("case3-120")
    model = zero_shot.MDP(**zero_shot.instance_config(instance, "known", zero_shot.DIMENSIONS))
    oracle, rules = benchmarks(instance, "known")
    greedy = comparer(model).assess(zero_shot.BasePolicy(model)).stats.total_demand
    for name, rule in rules.items():
        assert (comparer(oracle).assess(rule).stats.total_demand == greedy).all(), name


@pytest.mark.parametrize("instance_id", ["case1-137", "online-012"])
@pytest.mark.parametrize("mode", ["known", "demand"])
def test_the_stationary_model_simulates_like_the_zero_shot_model(instance_id, mode):
    instance = cases.find(instance_id)
    warmup = 100 if mode == "known" else 0
    expected = greedy_mean(zero_shot, zero_shot.instance_config(instance, mode, zero_shot.DIMENSIONS), warmup)
    assert greedy_mean(stationary, stationary.instance_config(instance, mode, {}), warmup) == expected


# DynaPlex 1.14 does not count an empty list towards garbage collection, so a model whose only per-period
# allocation was an empty list was never swept: collection grew by 6.4 MiB per sample until a laptop crashed.
@pytest.mark.skipif(sys.platform == "win32", reason="peak memory is read with the POSIX resource module")
@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_collection_memory_does_not_grow_with_the_samples(environment):
    small, large = (
        peak_mib("tests/collect.py", environment, samples) for samples in (40, 240)
    )
    assert large - small < 64, (small, large)


# With stochastic lead times, every trajectory's initial state leaves garbage that DynaPlex 1.14 does not
# count towards its sweeps: tuning BS and CBS grew by about 190 MiB per assessment of 1000 trajectories.
# Without the sweeps the larger run here grows by about 300 MiB, so a failing test stays harmless.
@pytest.mark.skipif(sys.platform == "win32", reason="peak memory is read with the POSIX resource module")
def test_tuning_memory_does_not_grow_with_the_trajectories():
    small, large = (peak_mib("tests/tune.py", "case3-120", trajectories) for trajectories in (10, 50))
    assert large - small < 64, (small, large)
