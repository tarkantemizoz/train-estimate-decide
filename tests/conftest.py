import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "envs" / "lost_sales"
ENVIRONMENTS = ("zero_shot_lost_sales_stationary", "zero_shot_lost_sales_large", "lost_sales")
# Cyclic demand and crossing stochastic lead times that can be zero: the hardest dynamics.
SINGLE_INSTANCE = "case3-195"


def command(*arguments, check=True):
    result = subprocess.run(
        [sys.executable, *map(str, arguments)], cwd=ROOT, capture_output=True, text=True, timeout=300
    )
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def peak_mib(*arguments) -> float:
    """The peak memory of python <arguments>, measured by tests/peak_memory.py; POSIX only."""
    return float(command("tests/peak_memory.py", *arguments).stdout.split()[-1])


def instance_arguments(environment: str) -> tuple[str, ...]:
    return ("--instance", SINGLE_INSTANCE) if environment == "lost_sales" else ()


@pytest.fixture(scope="session")
def machine(tmp_path_factory):
    """A uniquely named small CPU machine, so test runs never touch real train_runs folders."""
    path = tmp_path_factory.mktemp("machines") / f"pytest-{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps({"machine": None, "torch_threads": 1}))
    yield path
    for environment in ENVIRONMENTS:
        shutil.rmtree(FAMILY / environment / "train_runs" / path.stem, ignore_errors=True)


@pytest.fixture(scope="session")
def smoke_agents(machine):
    agents = {}
    for environment in ENVIRONMENTS:
        command("train.py", environment, "smoke", "--machine", machine, *instance_arguments(environment))
        run = FAMILY / environment / "train_runs" / machine.stem / "smoke"
        if environment == "lost_sales":
            run /= SINGLE_INSTANCE
        agents[environment] = run / "agent_gen2"
    return agents
