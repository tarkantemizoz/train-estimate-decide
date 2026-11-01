import csv
import sys

import pytest
from conftest import SINGLE_INSTANCE, command, peak_mib

TINY = ("--trajectories", "4", "--horizons", "200", "--known-horizon", "50", "--warmup", "5")


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_online_evaluation_counts_wins_and_resumes(smoke_agents, tmp_path):
    arguments = (
        "evaluate.py",
        "zero_shot_lost_sales_stationary",
        "--agent",
        smoke_agents["zero_shot_lost_sales_stationary"],
        "--modes",
        "demand",
        *TINY,
    )
    output = command(*arguments, "--output", tmp_path).stdout
    assert "beats the best published online method in" in output and "of 16 windows" in output
    assert len(rows(tmp_path / "online.csv")) == 16 * 4
    command(*arguments, "--output", tmp_path)
    assert len(rows(tmp_path / "online.csv")) == 16 * 4
    changed = command(*arguments[:-2], "--horizons", "500", "--output", tmp_path, check=False)
    assert changed.returncode != 0 and "other settings" in changed.stderr


def test_modes_heuristics_and_the_shipped_networks(tmp_path):
    stationary = ("evaluate.py", "zero_shot_lost_sales_stationary", *TINY)
    command(*stationary, "--instances", "case1-000", "--modes", "known", "--output", tmp_path / "h")
    heuristics = rows(tmp_path / "h" / "case1.csv")
    assert {row["policy"] for row in heuristics} == {"greedy", "BS", "CBS"}
    assert {row["mode"] for row in heuristics} == {"known"}
    command(*stationary, "--agent", "reference/demo", "--instances", "online-000", "--output", tmp_path / "s")
    online = rows(tmp_path / "s" / "online.csv")
    assert {row["policy"] for row in online} == {"greedy", "trained", "BS", "CBS"}
    assert {row["mode"] for row in online} == {"known", "demand"}
    zero_shot = ("evaluate.py", "zero_shot_lost_sales_large", "--agent", "paper/gc_lsn", *TINY)
    command(*zero_shot, "--instances", "case1-000", "case3-000", "--output", tmp_path / "z")
    case3 = rows(tmp_path / "z" / "case3.csv")
    assert {row["mode"] for row in case3} == {"known", "demand", "leadtime", "both"}
    unsupported = command(*stationary, "--cases", "case2", check=False)
    assert unsupported.returncode != 0 and "cannot represent" in unsupported.stderr


# BS and CBS run on another model than greedy, but on the same random numbers, also with order crossover.
def test_differences_with_cbs_are_paired_per_trajectory(tmp_path):
    arguments = ("evaluate.py", "zero_shot_lost_sales_large", "--instances", "case3-000", "case3-120")
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


def test_a_single_instance_agent_is_evaluated_on_its_instance(smoke_agents, tmp_path):
    run = smoke_agents["lost_sales"].parent
    arguments = ("evaluate.py", "lost_sales", "--agent", run, *TINY)
    command(*arguments, "--output", tmp_path)
    results = rows(tmp_path / "case3.csv")
    assert {row["instance"] for row in results} == {SINGLE_INSTANCE}
    assert {row["mode"] for row in results} == {"known"}
    assert {row["policy"] for row in results} == {"greedy", "trained", "BS", "CBS"}
    other = command(*arguments, "--instances", "case1-000", "--output", tmp_path / "x", check=False)
    assert other.returncode != 0 and f"trained for {SINGLE_INSTANCE} only" in other.stderr


# DynaPlex 1.14 does not count the garbage of the estimated modes' per-period allocations towards its sweeps,
# and frees a heuristic's garbage only between trajectories: memory grew with the trajectories, past 10 GiB
# for the paper's protocol, and with the horizon, to 4.4 GiB for one window of BS and CBS on a laptop.
@pytest.mark.skipif(sys.platform == "win32", reason="peak memory is read with the POSIX resource module")
@pytest.mark.parametrize("environment", ["zero_shot_lost_sales_stationary", "zero_shot_lost_sales_large"])
def test_evaluation_memory_does_not_grow_with_the_trajectories_or_the_horizon(environment, tmp_path):
    # Geometric demand and lead time 7: the largest per-period allocations of the online instances.
    # Without the sweeps the large run grows to about 1 GiB, so a failing test stays harmless.
    arguments = ("evaluate.py", environment, "--instances", "online-007", "--modes", "demand")
    small, large = (
        peak_mib(*arguments, "--trajectories", n, "--horizons", horizon, "--output", tmp_path / n)
        for n, horizon in (("20", "200"), ("40", "500"))
    )
    assert large - small < 64, (small, large)
