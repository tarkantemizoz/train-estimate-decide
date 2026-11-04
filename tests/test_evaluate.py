import csv
import json
import sys

import pytest
from conftest import ENVS, SINGLE_INSTANCE, command, peak_mib

from envs.lost_sales.cases.tune import table
from envs.lost_sales.evaluate import window_line

TINY = ("--trajectories", "4", "--horizons", "200", "--known-horizon", "50", "--warmup", "5")


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_online_evaluation_counts_wins_and_resumes(smoke_agents, tmp_path):
    arguments = (
        "evaluate.py",
        "lost_sales/zero_shot_stationary",
        "--agent",
        smoke_agents["lost_sales/zero_shot_stationary"],
        "--modes",
        "demand",
        *TINY,
    )
    output = command(*arguments, "--output", tmp_path).stdout
    assert "beats the best published online method in" in output and "of 16 windows" in output
    windows = [line for line in output.splitlines() if " T=200 " in line]
    assert len(windows) == 16 and all(" vs best online " in line for line in windows)
    assert all(
        line.endswith(("trained better", "best online better", "too close to call")) for line in windows
    )
    assert len(rows(tmp_path / "online.csv")) == 16 * 4
    resumed = command(*arguments, "--output", tmp_path).stdout
    assert len(rows(tmp_path / "online.csv")) == 16 * 4 and " T=200 " not in resumed
    changed = command(*arguments[:-2], "--horizons", "500", "--output", tmp_path, check=False)
    assert changed.returncode != 0 and "other settings" in changed.stderr


def test_modes_heuristics_and_the_shipped_networks(tmp_path):
    stationary = ("evaluate.py", "lost_sales/zero_shot_stationary", *TINY)
    output = command(*stationary, "--instances", "case1-000", "--modes", "known", "--output", tmp_path / "h")
    [window] = [line for line in output.stdout.splitlines() if line.startswith("case1-000 ")]
    assert " T=50 " in window and " greedy " in window and " vs CBS " in window
    heuristics = rows(tmp_path / "h" / "case1.csv")
    assert {row["policy"] for row in heuristics} == {"greedy", "BS", "CBS"}
    assert {row["mode"] for row in heuristics} == {"known"}
    command(*stationary, "--agent", "reference/demo", "--instances", "online-000", "--output", tmp_path / "s")
    online = rows(tmp_path / "s" / "online.csv")
    assert {row["policy"] for row in online} == {"greedy", "trained", "BS", "CBS"}
    assert {row["mode"] for row in online} == {"known", "demand"}
    zero_shot = ("evaluate.py", "lost_sales/zero_shot_large", "--agent", "paper/gc_lsn", *TINY)
    command(*zero_shot, "--instances", "case1-000", "case3-000", "--output", tmp_path / "z")
    case3 = rows(tmp_path / "z" / "case3.csv")
    assert {row["mode"] for row in case3} == {"known", "demand", "leadtime", "both"}
    unsupported = command(*stationary, "--cases", "case2", check=False)
    assert unsupported.returncode != 0 and "cannot represent" in unsupported.stderr


def test_a_window_line_says_which_policy_is_better():
    published = {"online-012": {"200": 69.4}}

    def verdict(mode, horizon, trained, cbs=None):
        """The verdict for a trained (score, error, delta_cbs, delta_cbs_error) and CBS's score."""
        mean, error, delta, delta_error = trained
        rows = [
            {
                "policy": "trained",
                "mean": mean,
                "error": error,
                "delta_cbs": delta,
                "delta_cbs_error": delta_error,
            },
            {"policy": "CBS", "mean": cbs, "error": 0.0, "delta_cbs": 0.0, "delta_cbs_error": 0.0},
        ]
        return window_line("online-012", mode, horizon, rows, "trained", published)

    # The shipped network at T = 200, and nearer to the published 69.4 than its error and the rounding.
    assert verdict("demand", 200, (-68.61, 0.19, 0, 0)).endswith(
        "68.61 ± 0.37 vs best online 69.4: best online better"
    )
    assert verdict("demand", 200, (-69.5, 0.19, 0, 0)).endswith("too close to call")
    assert verdict("demand", 200, (-70.0, 0.1, 0, 0)).endswith("trained better")
    # Otherwise against CBS, paired, in cost or, with estimated demand, in profit.
    known = verdict("known", 5000, (6.786, 0.02, 0.029, 0.001), cbs=6.757)
    assert known.endswith("cost   trained 6.786 vs CBS 6.757, difference +0.029 ± 0.002: CBS better")
    unpublished = verdict("demand", 300, (-50.0, 0.2, -1.0, 0.1), cbs=-49.0)
    assert unpublished.endswith(
        "profit trained 50.000 vs CBS 49.000, difference +1.000 ± 0.196: trained better"
    )


# BS and CBS run on another model than greedy, but on the same random numbers, also with order crossover.
def test_differences_with_cbs_are_paired_per_trajectory(tmp_path):
    arguments = ("evaluate.py", "lost_sales/zero_shot_large", "--instances", "case3-000", "case3-120")
    command(
        *arguments, "--modes", "known", "--trajectories", "32", "--known-horizon", "500", "--output", tmp_path
    )
    results = rows(tmp_path / "case3.csv")
    assert {row["instance"] for row in results} == {"case3-000", "case3-120"}
    for row in results:
        if row["policy"] == "CBS":
            assert float(row["delta_cbs"]) == float(row["delta_cbs_error"]) == 0
        if row["policy"] == "BS":
            assert 0 < float(row["delta_cbs_error"]) < float(row["error"]) / 2, row
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "case3.csv").write_text("instance,mode,horizon,warmup,policy,mean,error,true_cost\n")
    old = command(*arguments, "--output", tmp_path / "old", check=False)
    assert old.returncode != 0 and "older evaluate.py" in old.stderr


def test_a_single_instance_agent_is_evaluated_on_the_instance_of_its_config(machine, smoke_agents, tmp_path):
    environment = "lost_sales/single_instance"
    arguments = ("evaluate.py", environment, "--agent", smoke_agents[environment].parent, *TINY)
    command(*arguments, "--output", tmp_path / "smoke")
    results = rows(tmp_path / "smoke" / "case3.csv")
    assert {row["instance"] for row in results} == {SINGLE_INSTANCE}
    assert {row["mode"] for row in results} == {"known"}
    assert {row["policy"] for row in results} == {"greedy", "trained", "BS", "CBS"}
    other = command(*arguments, "--instances", "case1-000", "--output", tmp_path / "x", check=False)
    assert other.returncode != 0 and f"trained for {SINGLE_INSTANCE} only" in other.stderr
    # --instance trains on a case instance instead of the config's, in a run of its own.
    command("train.py", environment, "smoke", "--machine", machine, "--instance", "online-012")
    run = ENVS / environment / "train_runs" / machine.stem / "smoke-online-012"
    command("evaluate.py", environment, "--agent", run, *TINY, "--output", tmp_path / "online")
    assert {row["instance"] for row in rows(tmp_path / "online" / "online.csv")} == {"online-012"}


# An instance of your own, here with a lead time beyond the paper's dimensions, gets BS and CBS tuned for it.
def test_an_agent_for_an_instance_of_your_own_is_compared_with_benchmarks_tuned_for_it(machine, tmp_path):
    environment = "lost_sales/single_instance"
    config = json.loads((ENVS / environment / "configs" / "smoke.json").read_text())
    config["mdp"] = {
        "p": 4.0,
        "demand_cycles": [0],
        "mean_demand": [2.0],
        "std_demand": [1.5],
        "leadtime": 11,
    }
    (tmp_path / "own.json").write_text(json.dumps(config))
    command("train.py", environment, tmp_path / "own.json", "--machine", machine)
    run = ENVS / environment / "train_runs" / machine.stem / "own"
    command("evaluate.py", environment, "--agent", run, *TINY, "--output", tmp_path / "own")
    results = rows(tmp_path / "own" / "custom.csv")
    assert {row["instance"] for row in results} == {"custom"}
    assert {row["policy"] for row in results} == {"greedy", "trained", "BS", "CBS"}
    assert set(table(tmp_path / "own" / "benchmark_parameters.csv")) == {"custom"}


# DynaPlex 1.14 does not count the garbage of the estimated modes' per-period allocations towards its sweeps,
# and frees a heuristic's garbage only between trajectories: memory grew with the trajectories, past 10 GiB
# for the paper's protocol, and with the horizon, to 4.4 GiB for one window of BS and CBS on a laptop.
@pytest.mark.skipif(sys.platform == "win32", reason="peak memory is read with the POSIX resource module")
@pytest.mark.parametrize("environment", ["lost_sales/zero_shot_stationary", "lost_sales/zero_shot_large"])
def test_evaluation_memory_does_not_grow_with_the_trajectories_or_the_horizon(environment, tmp_path):
    # Geometric demand and lead time 7: the largest per-period allocations of the online instances.
    # Without the sweeps the large run grows to about 1 GiB, so a failing test stays harmless.
    arguments = ("evaluate.py", environment, "--instances", "online-007", "--modes", "demand")
    small, large = (
        peak_mib(*arguments, "--trajectories", n, "--horizons", horizon, "--output", tmp_path / n)
        for n, horizon in (("20", "200"), ("40", "500"))
    )
    assert large - small < 64, (small, large)
