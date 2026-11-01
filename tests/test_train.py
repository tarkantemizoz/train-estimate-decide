import json

import dynaplex as dp
import pytest
from conftest import ENVIRONMENTS, FAMILY, command, instance_arguments

import train


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_training_saves_resumes_and_refuses_a_changed_experiment(
    environment, machine, smoke_agents, tmp_path
):
    run = smoke_agents[environment].parent
    for name in ("agent_gen1", "agent_gen2", "config.json", "machine.json", "train.log", "dcl.json"):
        assert (run / name).exists(), name
    extra = instance_arguments(environment)
    resumed = command("train.py", environment, "smoke", "--machine", machine, *extra)
    assert "agent_gen2 exists" in resumed.stdout
    changed = json.loads((FAMILY / environment / "configs" / "smoke.json").read_text())
    changed["dcl"]["m"] += 1
    (tmp_path / "smoke.json").write_text(json.dumps(changed))
    result = command(
        "train.py", environment, tmp_path / "smoke.json", "--machine", machine, *extra, check=False
    )
    assert result.returncode != 0 and "WorkdirMismatchError" in result.stderr
    assert json.loads((run / "config.json").read_text())["dcl"]["m"] == changed["dcl"]["m"] - 1


def test_dry_run_shows_the_plan_without_creating_a_run(machine):
    output = command(
        "train.py", "zero_shot_lost_sales_stationary", "demo", "--machine", machine, "--dry-run"
    ).stdout
    assert "generation 0" in output
    assert not (FAMILY / "zero_shot_lost_sales_stationary" / "train_runs" / machine.stem / "demo").exists()


@pytest.mark.parametrize(
    ("environment", "extra", "message"),
    [
        ("lost_sales", (), "choose it with --instance"),
        ("lost_sales", ("--instance", "case9-000"), "unknown instance"),
        ("zero_shot_lost_sales_stationary", ("--instance", "online-003"), "--instance does not apply"),
    ],
)
def test_instance_flag_matches_the_environment(machine, environment, extra, message):
    result = command("train.py", environment, "smoke", "--machine", machine, "--dry-run", *extra, check=False)
    assert result.returncode != 0 and message in result.stderr


def test_an_unrecognized_machine_trains_with_the_laptop_plan_outside_slurm(monkeypatch):
    recognize = dp.machine.plan

    def plan(machine, detected=None):
        if machine == "auto":
            raise dp.machine.MachineMismatchError("machine='auto': no preset matches")
        return recognize(machine, detected)

    monkeypatch.setattr(dp.machine, "plan", plan)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert train.plan_name("auto") == "laptop"
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    with pytest.raises(dp.machine.MachineMismatchError):
        train.plan_name("auto")
