import json

import dynaplex as dp
import pytest
from conftest import ENVIRONMENTS, ENVS, command

import envs
import train


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_training_saves_resumes_and_refuses_a_changed_experiment(
    environment, machine, smoke_agents, tmp_path
):
    run = smoke_agents[environment].parent
    for name in ("agent_gen1", "agent_gen2", "config.json", "machine.json", "train.log", "dcl.json"):
        assert (run / name).exists(), name
    resumed = command("train.py", environment, "smoke", "--machine", machine)
    assert "agent_gen2 exists" in resumed.stdout
    changed = json.loads((ENVS / environment / "configs" / "smoke.json").read_text())
    changed["dcl"]["m"] += 1
    (tmp_path / "smoke.json").write_text(json.dumps(changed))
    result = command("train.py", environment, tmp_path / "smoke.json", "--machine", machine, check=False)
    assert result.returncode != 0 and "WorkdirMismatchError" in result.stderr
    assert json.loads((run / "config.json").read_text())["dcl"]["m"] == changed["dcl"]["m"] - 1


def test_dry_run_shows_the_plan_without_creating_a_run(machine):
    environment = "lost_sales/zero_shot_stationary"
    output = command("train.py", environment, "demo", "--machine", machine, "--dry-run").stdout
    assert "generation 0" in output
    assert not (ENVS / environment / "train_runs" / machine.stem / "demo").exists()


def test_every_config_holds_the_keyword_arguments_of_its_mdp():
    for name in envs.names():
        environment = envs.find(name)
        for path in (ENVS / name / "configs").glob("*.json"):
            environment.MDP(**json.loads(path.read_text())["mdp"])


@pytest.mark.parametrize(
    ("environment", "extra", "message"),
    [
        ("zero_shot_stationary", (), "invalid choice"),
        ("lost_sales/single_instance", ("--instance", "case9-000"), "unknown instance"),
        ("lost_sales/zero_shot_stationary", ("--instance", "online-003"), "--instance does not apply"),
    ],
)
def test_the_environment_and_instance_must_exist(machine, environment, extra, message):
    result = command("train.py", environment, "smoke", "--machine", machine, "--dry-run", *extra, check=False)
    assert result.returncode != 0 and message in result.stderr


def test_a_config_without_an_mdp_section_is_refused(machine, tmp_path):
    environment = "lost_sales/single_instance"
    config = json.loads((ENVS / environment / "configs" / "smoke.json").read_text())
    del config["mdp"]
    (tmp_path / "bare.json").write_text(json.dumps(config))
    result = command(
        "train.py", environment, tmp_path / "bare.json", "--machine", machine, "--dry-run", check=False
    )
    assert result.returncode != 0 and 'no "mdp" section' in result.stderr


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
