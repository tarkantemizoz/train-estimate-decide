"""Participant commands must work without research scripts or exported instances."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ted_tutorial.evaluate_online import paper_instances
from zero_shot_lost_sales_inventory_control.cases import instances

ROOT = Path(__file__).resolve().parents[1]


def command(*arguments):
    result = subprocess.run(
        [sys.executable, *map(str, arguments)], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    directory = tmp_path_factory.mktemp("experiment-policies")
    config = json.loads((ROOT / "configs/training/smoke.json").read_text())
    config["dcl"].update(machine=None, torch_threads=1)
    config["generations"] = 1
    path = directory / "config.json"
    path.write_text(json.dumps(config))
    outputs = {}
    for environment in ("lost_sales_stationary", "zero_shot_lost_sales_inventory_control"):
        output = directory / environment
        command("train.py", environment, path, "--output", output)
        outputs[environment] = output / "agent_gen1"
    return outputs


def test_online_catalogue_matches_every_published_reference_instance():
    reference = json.loads((ROOT / "configs/evaluation/online_reference.json").read_text())
    targets = {row["instance_id"]: row for row in reference["instances"]}
    catalogue = paper_instances()
    assert len(catalogue) == len(targets) == 16
    for item in catalogue:
        config = item["config"]
        target = targets[item["id"]]
        for field in ("p", "leadtime", "mean_demand", "std_demand"):
            assert config[field] == target[field]
        assert config["evaluate"] and config["censored_demand"]
        assert set(target["online_profit"]) == {"200", "500", "1000", "2000"}
    assert [len(instances(case)) for case in ("case1", "case2", "case3")] == [320, 243, 240]


def test_online_cli_evaluates_compares_and_resumes(checkpoints, tmp_path):
    output = tmp_path / "online"
    args = (
        "-m",
        "ted_tutorial.evaluate_online",
        "--agent",
        checkpoints["lost_sales_stationary"],
        "--output",
        output,
        "--trajectories",
        "4",
        "--horizons",
        "200",
        "--processes",
        "1",
        "--workers",
        "1",
    )
    command(*args)
    command("-m", "ted_tutorial.compare_online", output)
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["windows"] == 16
    assert comparison["expected_full_grid_windows"] == 64
    record_path = output / "paper-000__200.json"
    record = json.loads(record_path.read_text())
    with np.load(record_path.with_suffix(".npz")) as raw:
        assert raw["trained_profit"].mean() == record["policies"]["trained"]["profit_mean"]
        np.testing.assert_array_equal(raw["greedy_demand"], raw["trained_demand"])
    before = record_path.stat().st_mtime_ns
    command(*args)
    assert record_path.stat().st_mtime_ns == before
    changed = subprocess.run(
        [sys.executable, *map(str, args), "--seed", "99"], cwd=ROOT, capture_output=True, text=True
    )
    assert changed.returncode != 0 and "identity differs" in changed.stderr


def test_full_case_cli_runs_each_information_setting(checkpoints, tmp_path):
    output = tmp_path / "cases"
    command(
        "-m",
        "zero_shot_lost_sales_inventory_control.evaluate_cases",
        "--case",
        "case1",
        "case2",
        "case3",
        "--smoke",
        "--agent",
        checkpoints["zero_shot_lost_sales_inventory_control"],
        "--output",
        output,
    )
    for case, count in (("case1", 2), ("case2", 2), ("case3", 4)):
        summary = json.loads((output / "smoke" / case / "summary.json").read_text())
        assert summary["completed_windows"] == count


def test_benchmark_command_reuses_parameters_and_seeded_costs(tmp_path):
    output = tmp_path / "tables"
    args = (
        "-m",
        "ted_tutorial.build_benchmarks",
        "--cases",
        "case1",
        "--instances",
        "case1-000",
        "--processes",
        "1",
        "--workers",
        "1",
        "--trajectories",
        "4",
        "--horizon",
        "32",
        "--warmup",
        "5",
        "--cache",
        tmp_path / "cache",
        "--output",
        output,
    )
    command(*args)
    before = (output / "case1/table.csv").read_bytes()
    command(*args)
    assert (output / "case1/table.csv").read_bytes() == before
    for phase in ("tuning", "costs"):
        record = json.loads(sorted((output / "case1/timings").glob(f"*_{phase}.json"))[-1].read_text())
        assert record["status"] == "complete"
        assert record["new_simulations"] == 0
