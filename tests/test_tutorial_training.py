import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import dynaplex as dp
import numpy as np
import pytest

from ted_tutorial.config import read_config, resolve_config
from ted_tutorial.distributed import aggregate_shards
from ted_tutorial.remote import job_script, remote_settings, source_files
from ted_tutorial.training import model_parts

ROOT = Path(__file__).resolve().parents[1]


def config():
    return read_config(ROOT / "configs/training/smoke.json", "lost_sales_stationary")


@pytest.mark.parametrize("environment", ["lost_sales_stationary", "zero_shot_lost_sales_inventory_control"])
def test_portable_command_trains_reloads_resumes_and_rejects_changed_run(tmp_path, environment):
    settings = config()
    settings["execution"].update(processes=2, workers=1, device="cpu", torch_threads=1, fit_threads=1)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings))
    output = tmp_path / "run"
    command = [sys.executable, "train.py", environment, str(path), "--output", str(output)]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=150)
    assert first.returncode == 0, first.stdout + first.stderr
    checkpoint = output / "agent_gen2"
    agent = dp.NNAgent.load(str(checkpoint))
    samples = dp.SampleSet.load(str(output / "samples_gen1"))
    assert len(samples) == 128
    assert samples.manifest["rollout"]["generation"] == 1
    assert agent.info["generation"] == 2
    assert agent.info["config_hash"] == samples.manifest["config_hash"]
    actions = agent.act({key: value[:32] for key, value in samples.features.items()}, samples.mask[:32])
    assert np.all(samples.mask[np.arange(32), actions])
    mtimes = {p.name: p.stat().st_mtime_ns for p in checkpoint.iterdir()}
    original_timing = (output / "timing.json").read_text()
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "agent_gen2 exists; skipping" in second.stdout
    assert {p.name: p.stat().st_mtime_ns for p in checkpoint.iterdir()} == mtimes
    assert (output / "timing.json").read_text() == original_timing
    settings["dcl"]["m"] += 1
    path.write_text(json.dumps(settings))
    changed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert changed.returncode != 0
    assert "different run" in changed.stderr


@pytest.mark.parametrize(
    ("cpus", "n", "ranks"), [(1, 100000, 1), (15, 100000, 1), (192, 5000000, 1), (192, 5000000, 2)]
)
def test_auto_uses_available_allocation_and_safe_controller_capacity(cpus, n, ranks):
    settings = config()
    settings["dcl"].update(n=n, m=500)
    settings["execution"]["device"] = "cpu"
    before = copy.deepcopy(settings)
    resolved = resolve_config(settings, cpus=cpus, world_size=ranks)
    execution = resolved["execution"]
    assert execution["processes"] * execution["workers"] == cpus
    assert execution["fit_threads"] == cpus
    assert settings == before
    assert execution["processes"] * resolved["generations"] * ranks <= 4096


def test_unavailable_requested_device_fails(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings = config()
    settings["execution"]["device"] = "cuda"
    with pytest.raises(ValueError, match="unavailable"):
        resolve_config(settings)


def test_mpi_shards_use_disjoint_seed_coordinates_and_one_logical_generation(tmp_path):
    from dynaplex.modelling.combine_seeds import combine_seeds

    settings = config()
    model, policy, features = model_parts("lost_sales_stationary", settings)
    shards = []
    for rank, n in enumerate((9, 8)):
        shards.append(
            dp.collect_samples(
                model,
                policy,
                features=features,
                n=n,
                m=2,
                h=2,
                slots=8,
                workers=1,
                generation=rank,
                seed=0,
                sample_info=True,
                jit=True,
                warmup_time=0,
                restart_after=2,
                checks=False,
            )
        )
    merged = aggregate_shards(shards, generation=0, world_size=2, total_samples=17)
    assert len(merged) == 17
    assert merged.manifest["generation"] == 0
    assert merged.manifest["distributed"]["shards"][1] == shards[1].manifest
    assert merged.manifest["info_columns"][0] == "mpi_rank"
    assert np.bincount(merged.info[:, 0]).tolist() == [9, 8]
    first_seeds = {combine_seeds(0, 0, sample, 0, 0) for sample in range(100)}
    second_seeds = {combine_seeds(0, 0, 131072 + sample, 0, 0) for sample in range(100)}
    assert not first_seeds & second_seeds
    agent = dp.train_network(merged, network=dp.MLP(hidden=[8]), epochs=1, batch_size=8, seed=0)
    assert agent.info["generation"] == 1
    assert agent.info["config_hash"] == merged.manifest["config_hash"]
    with pytest.raises(ValueError, match="RNG generation"):
        aggregate_shards([shards[0], shards[0]], generation=0, world_size=2, total_samples=18)


@pytest.fixture(scope="session")
def bash_executable():
    candidates = []
    if sys.platform == "win32" and (git := shutil.which("git")):
        # Prefer Git Bash to the Windows WSL launcher, which may have no distro.
        candidates.extend(parent / "bin/bash.exe" for parent in Path(git).resolve().parents[:3])
    if bash := shutil.which("bash"):
        candidates.append(Path(bash))
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            probe = subprocess.run([str(candidate), "--version"], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return str(candidate)
    pytest.skip("Slurm script syntax check requires a working Bash installation")


@pytest.mark.parametrize("profile", ["genoa", "mpi", "gpu"])
def test_slurm_scripts_parse_and_use_one_rank_per_node(profile, bash_executable):
    settings = remote_settings(
        {"workstation": {"target": "snellius", "host": "example@snellius.surf.nl", "profile": profile}}
    )
    script = job_script(
        settings, "lost_sales_stationary", "/home/example/release", "/home/example/venv", "/home/example/job"
    )
    subprocess.run([bash_executable, "-n"], input=script, text=True, check=True, timeout=10)
    assert "--ntasks-per-node=1" in script
    if profile == "mpi":
        assert "mpirun --map-by ppr:1:node --bind-to none" in script
        assert "--mpi" in script
    if profile == "gpu":
        assert "--gpus-per-node=1" in script
        assert "torch.cuda.is_available" in script


def test_remote_bundle_excludes_results_credentials_and_legacy_source():
    for path in source_files():
        relative = path.relative_to(ROOT)
        assert relative.parts[0] not in ("DynaPlex", "runs", ".venv", ".ssh")
        assert path.is_file()


@pytest.mark.parametrize(
    "key,value",
    [
        ("nodes", 0),
        ("account", "x\n#SBATCH --nodes=99"),
        ("remote_root", "../other"),
        ("host", "-oProxyCommand=x"),
    ],
)
def test_remote_rejects_invalid_resource_and_destination_settings(key, value):
    settings = {"target": "snellius", "host": "example@snellius.surf.nl", key: value}
    with pytest.raises(ValueError):
        remote_settings({"workstation": settings})
