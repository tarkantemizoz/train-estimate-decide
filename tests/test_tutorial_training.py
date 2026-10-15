import json
import shutil
import subprocess
import sys
from pathlib import Path

import dynaplex as dp
import numpy as np
import pytest

from ted_tutorial.config import read_config
from ted_tutorial.distributed import aggregate_shards
from ted_tutorial.remote import job_script, remote_settings, source_files
from ted_tutorial.training import build_experiment, model_parts, preview

ROOT = Path(__file__).resolve().parents[1]


def config():
    return read_config(ROOT / "configs/training/smoke.json", "lost_sales_stationary")


@pytest.mark.parametrize("environment", ["lost_sales_stationary", "zero_shot_lost_sales_inventory_control"])
def test_portable_command_trains_reloads_resumes_and_rejects_changed_run(tmp_path, environment):
    settings = config()
    settings["dcl"].update(
        machine=None,
        torch_threads=1,
        gen0={"processes": 2, "workers": 1, "slots": 32},
        trained={"processes": 2, "workers": 1, "slots": 32},
    )
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
    assert "agent_gen2 exists" in second.stdout
    assert {p.name: p.stat().st_mtime_ns for p in checkpoint.iterdir()} == mtimes
    assert (output / "timing.json").read_text() == original_timing
    settings["dcl"]["m"] += 1
    path.write_text(json.dumps(settings))
    changed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert changed.returncode != 0
    assert "different experiment" in changed.stderr


def test_omitted_options_are_native_defaults_through_cli(tmp_path):
    # This used to fail because the wrapper required n/m/h, network and train.
    settings = {"dcl": {"machine": None}, "generations": 1}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings))
    output = tmp_path / "must-not-be-created"
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "lost_sales_stationary",
            str(path),
            "--dry-run",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(result.stdout)
    model, policy, features = model_parts("lost_sales_stationary", {})
    native = dp.DCL(model, policy, features=features, machine=None, workdir=str(tmp_path / "native"))
    expected = json.loads((Path(native.workdir) / "dcl.json").read_text())
    assert actual["dcl"] == expected
    assert actual["torch_threads"] is None
    assert not output.exists()
    assert json.loads(path.read_text()) == settings


def test_native_machine_default_and_partial_shape_override(monkeypatch, tmp_path):
    import dynaplex.machine as machine

    # Exercise native auto selection on CI platforms with a deterministic native laptop plan.
    native_plan = machine.plan("laptop")
    calls = []

    def recognize(name):
        calls.append(name)
        return native_plan

    monkeypatch.setattr(machine, "plan", recognize)
    settings = {"generations": 1, "dcl": {"gen0": {"workers": 1}}}
    actual = build_experiment("lost_sales_stationary", settings, tmp_path / "run")
    assert calls == ["auto"]
    assert actual.gen0 == dp.Shape(workers=1).resolve(native_plan.gen0)
    assert actual.trained == native_plan.trained
    assert actual.device == native_plan.device
    assert actual.train_device == native_plan.training_device
    assert actual.torch_threads == native_plan.torch_threads
    assert settings == {"generations": 1, "dcl": {"gen0": {"workers": 1}}}


def test_native_training_defaults_work_without_network_or_train_sections(tmp_path):
    settings = {"dcl": {"machine": None, "n": 32, "m": 2, "h": 2, "torch_threads": 1}, "generations": 1}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(settings))
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "train.py", "lost_sales_stationary", str(path), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    agent = dp.NNAgent.load(str(output / "agent_gen1"))
    assert agent.info["generation"] == 1
    assert len(dp.SampleSet.load(str(output / "samples_gen0"))) == 32
    record = json.loads((output / "dcl.json").read_text())
    assert record["train"] == {}


def test_native_dcl_option_typo_is_reported():
    with pytest.raises(TypeError, match="unexpected keyword argument.*workres"):
        preview("lost_sales_stationary", {"dcl": {"machine": None, "workres": 4}})


def test_legacy_execution_block_is_not_silently_ignored(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"execution": {"workers": "auto"}, "generations": 1}))
    with pytest.raises(ValueError, match="custom execution section was removed"):
        read_config(path, "lost_sales_stationary")


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


def test_mpi_adapter_uses_native_shapes_and_fits_shared_generations(tmp_path):
    from ted_tutorial.distributed import _run

    class LocalComm:
        def allgather(self, value):
            return [value]

        def bcast(self, value, root):
            return value

        def Barrier(self):
            pass

    # Run the real collector, aggregator and fitter with a local communicator.
    # The separate shard test above verifies two-rank seed separation and merging.
    settings = config()
    settings["dcl"].update(
        machine=None,
        torch_threads=1,
        gen0={"workers": 1, "slots": 32},
        trained={"workers": 2, "slots": 32},
    )
    output = tmp_path / "mpi"
    result = _run(LocalComm(), 0, 1, "lost_sales_stationary", settings, output)
    assert result["checkpoint_saved"]
    for generation, workers in [(0, 1), (1, 2)]:
        shard = dp.SampleSet.load(str(output / f"shards_gen{generation}/rank0000"))
        assert shard.manifest["execution"]["workers"] == workers
        assert len(shard) == settings["dcl"]["n"]
    agent = dp.NNAgent.load(str(output / "agent_gen2"))
    assert agent.info["generation"] == 2


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
