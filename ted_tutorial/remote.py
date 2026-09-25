"""Stage only tutorial code, submit Slurm jobs, and retrieve their artifacts over SSH."""

import copy
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

from .training import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["2025", "Python/3.13.5-GCCcore-14.3.0", "OpenMPI/5.0.8-GCC-14.3.0"]
PROFILES = {
    "genoa": {"partition": "genoa", "nodes": 1, "cpus": 192, "memory": "336G", "gpus": 0},
    "mpi": {"partition": "genoa", "nodes": 2, "cpus": 192, "memory": "336G", "gpus": 0},
    "gpu": {"partition": "gpu_h100", "nodes": 1, "cpus": 16, "memory": "64G", "gpus": 1},
}


def remote_settings(config):
    settings = config["workstation"]
    allowed = {
        "target",
        "host",
        "profile",
        "remote_root",
        "account",
        "partition",
        "nodes",
        "cpus",
        "memory",
        "gpus",
        "time",
        "modules",
        "python",
        "poll_seconds",
        "fetch_samples",
    }
    if extra := set(settings) - allowed:
        raise ValueError(f"Unknown workstation keys: {sorted(extra)}")
    profile = settings.get("profile", "genoa")
    if profile not in PROFILES:
        raise ValueError("Snellius profile must be genoa, mpi, or gpu")
    out = (
        PROFILES[profile]
        | {
            "time": "01:00:00",
            "modules": MODULES,
            "python": "python3",
            "remote_root": "ted-tutorial",
            "poll_seconds": 10,
            "fetch_samples": False,
        }
        | settings
    )
    out["profile"] = profile
    if not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+", out.get("host", "")):
        raise ValueError("Set workstation.host to your SSH destination, e.g. username@snellius.surf.nl")
    for key in ("partition", "memory", "account"):
        if key in out and not re.fullmatch(r"[A-Za-z0-9_.-]+", out[key]):
            raise ValueError(f"Invalid workstation.{key}")
    for key in ("nodes", "cpus", "poll_seconds"):
        if type(out[key]) is not int or out[key] < 1:
            raise ValueError(f"workstation.{key} must be a positive integer")
    if type(out["gpus"]) is not int or out["gpus"] < 0:
        raise ValueError("workstation.gpus must be a nonnegative integer")
    if not re.fullmatch(r"(?:\d+-)?\d{1,3}:\d{2}:\d{2}", out["time"]):
        raise ValueError("workstation.time must be HH:MM:SS or D-HH:MM:SS")
    if not isinstance(out["modules"], list) or not all(isinstance(v, str) for v in out["modules"]):
        raise ValueError("workstation.modules must be a list of module names")
    if profile == "mpi" and out["nodes"] < 2:
        raise ValueError("The MPI profile requires at least two nodes")
    if profile != "mpi" and out["nodes"] != 1:
        raise ValueError("Multiple nodes require the MPI profile")
    if profile == "gpu" and out["gpus"] != 1:
        raise ValueError("The GPU tutorial profile uses one GPU; fitting is not distributed across GPUs")
    if profile != "gpu" and out["gpus"] != 0:
        raise ValueError("Use the gpu profile to request GPUs")
    path = PurePosixPath(out["remote_root"])
    if ".." in path.parts or not re.fullmatch(r"[A-Za-z0-9_./-]+", str(path)) or str(path) in (".", "/"):
        raise ValueError("remote_root must be a dedicated directory without '..', spaces, or shell syntax")
    return out


def ssh(settings, script, *, capture=False):
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", settings["host"], "bash -l -s"],
        input=script,
        text=True,
        capture_output=capture,
        check=True,
    )
    return result.stdout.strip() if capture else None


def module_setup(settings):
    # Loading the explicitly recorded stack also replaces the login shell's old Conda Python.
    return "module purge\nmodule load " + shlex.join(settings["modules"]) + "\n"


def source_files():
    files = [
        ROOT / name for name in ("train.py", "pyproject.toml", "requirements.txt", "requirements-mpi.txt")
    ]
    for directory in ("ted_tutorial", "lost_sales_stationary", "zero_shot_lost_sales_inventory_control"):
        files.extend(sorted((ROOT / directory).glob("*.py")))
    return files


def bundle_identity():
    return digest(
        {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}
    )


def setup_script(settings, release, venv):
    q = shlex.quote
    index = (
        "https://download.pytorch.org/whl/cu130"
        if settings["profile"] == "gpu"
        else "https://download.pytorch.org/whl/cpu"
    )
    return (
        "set -euo pipefail\n"
        + module_setup(settings)
        + f"cd {q(release)}\n"
        + f"if [ ! -x {q(venv + '/bin/python')} ]; then {q(settings['python'])} -m venv {q(venv)}; fi\n"
        + f"{q(venv + '/bin/python')} -m pip install 'torch==2.14.0' --index-url {q(index)}\n"
        + f"{q(venv + '/bin/python')} -m pip install -r requirements.txt -r requirements-mpi.txt\n"
        + f"{q(venv + '/bin/python')} -m pip check\n"
        + f"{q(venv + '/bin/python')} - <<'PY'\n"
        + "import json, urllib.request\nfrom importlib.metadata import version\n"
        + "print('Installed DynaPlex:', version('dynaplex'))\n"
        + "assert version('dynaplex') == '1.14.0'\n"
        + "try:\n"
        + "    with urllib.request.urlopen('https://pypi.org/pypi/dynaplex/json', timeout=10) as r:\n"
        + "        latest = json.load(r)['info']['version']\n"
        + "    print('Latest PyPI DynaPlex:', latest, '(tutorial remains pinned)')\n"
        + "except Exception as exc:\n    print('Latest-version lookup unavailable:', exc)\n"
        + "PY\n"
    )


def job_script(settings, environment, release, venv, jobdir):
    q = shlex.quote
    directives = [
        "#!/bin/bash -l",
        "#SBATCH --job-name=ted-training",
        f"#SBATCH --partition={settings['partition']}",
        f"#SBATCH --nodes={settings['nodes']}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={settings['cpus']}",
        f"#SBATCH --mem={settings['memory']}",
        f"#SBATCH --time={settings['time']}",
        f"#SBATCH --output={jobdir}/slurm-%j.out",
        f"#SBATCH --error={jobdir}/slurm-%j.err",
    ]
    if settings.get("account"):
        directives.append(f"#SBATCH --account={settings['account']}")
    if settings["gpus"]:
        directives.append(f"#SBATCH --gpus-per-node={settings['gpus']}")
    command = [
        venv + "/bin/python",
        "-u",
        "train.py",
        environment,
        jobdir + "/config.json",
        "--output",
        jobdir + "/artifacts",
        "--in-allocation",
    ]
    if settings["profile"] == "mpi":
        command.append("--mpi")
        # mpirun uses the Slurm allocation. One rank/node receives the full node CPU set.
        # A Genoa node spans two sockets. OpenMPI rejects a 192-core PE binding
        # across packages; Slurm already restricts the job's CPUs and DynaPlex
        # partitions that allowed set among its local collection processes.
        command = ["mpirun", "--map-by", "ppr:1:node", "--bind-to", "none"] + command
    else:
        command = ["srun", "--cpu-bind=cores"] + command
    body = "\n".join(directives) + "\nset -euo pipefail\n" + module_setup(settings)
    body += "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
    body += f"cd {q(release)}\n"
    body += f"{q(venv + '/bin/python')} -m dynaplex.selftest\n"
    if settings["gpus"]:
        body += "nvidia-smi\n"
        body += (
            f"{q(venv + '/bin/python')} -c "
            + q(
                "import torch; assert torch.cuda.is_available(), 'CUDA unavailable; check wheel and driver'; "
                "print((torch.ones(8, device='cuda') @ torch.ones(8, device='cuda')).item())"
            )
            + "\n"
        )
    body += "set +e\n" + shlex.join(command) + "\nresult=$?\n"
    body += f'printf \'%s\\n\' "$result" > {q(jobdir + "/exit_code")}\nexit "$result"\n'
    return body


def stage(settings, environment, config, output):
    for binary in ("ssh", "scp"):
        if shutil.which(binary) is None:
            raise ValueError(f"Remote training requires {binary} on PATH")
    remote_home = ssh(settings, "pwd\n", capture=True).splitlines()[-1]
    root = str(PurePosixPath(remote_home) / settings["remote_root"])
    release_id = bundle_identity()
    release = root + "/releases/" + release_id
    venv_key = digest(
        {
            "modules": settings["modules"],
            "python": settings["python"],
            "gpu": bool(settings["gpus"]),
            "requirements": (ROOT / "requirements.txt").read_text(),
            "mpi": (ROOT / "requirements-mpi.txt").read_text(),
        }
    )[:16]
    venv = root + "/environments/" + venv_key
    run_id = digest(
        {
            "source": release_id,
            "config": config,
            "environment": environment,
            "local_output": str(output.resolve()),
        }
    )[:20]
    jobdir = root + "/jobs/" + run_id
    q = shlex.quote
    exists = ssh(
        settings, f"if [ -f {q(release + '/ready')} ]; then echo yes; else echo no; fi\n", capture=True
    )
    if exists.splitlines()[-1] != "yes":
        with tempfile.TemporaryDirectory(prefix="ted-upload-") as folder:
            archive = Path(folder) / "source.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for path in source_files():
                    tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
            ssh(settings, f"set -e\nmkdir -p {q(release)}\n")
            subprocess.run(
                ["scp", "-q", str(archive), f"{settings['host']}:{release}/source.tar.gz"], check=True
            )
            ssh(
                settings,
                f"set -e\ntar -xzf {q(release + '/source.tar.gz')} -C {q(release)}\n"
                f"touch {q(release + '/ready')}\n",
            )
    ssh(settings, setup_script(settings, release, venv))
    remote_config = copy.deepcopy(config)
    remote_config["workstation"] = {"target": "local"}
    remote_config.setdefault("execution", {}).setdefault("device", "cuda" if settings["gpus"] else "cpu")
    script = job_script(settings, environment, release, venv, jobdir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "submitted_config.json", remote_config)
    (output / "job.sbatch").write_text(script)
    ssh(settings, f"mkdir -p {q(jobdir)}\n")
    subprocess.run(
        ["scp", "-q", str(output / "submitted_config.json"), f"{settings['host']}:{jobdir}/config.json"],
        check=True,
    )
    subprocess.run(
        ["scp", "-q", str(output / "job.sbatch"), f"{settings['host']}:{jobdir}/job.sbatch"], check=True
    )
    return {
        "host": settings["host"],
        "jobdir": jobdir,
        "release": release,
        "venv": venv,
        "environment": environment,
        "config_hash": digest(config),
        "source_hash": release_id,
        "settings": settings,
    }


def fetch(record, output):
    """Fetch logs/checkpoints by default, without transferring large sample arrays."""
    settings, jobdir = record["settings"], record["jobdir"]
    exclusions = (
        []
        if settings["fetch_samples"]
        else ["--exclude=artifacts/samples_gen*", "--exclude=artifacts/shards_gen*"]
    )
    script = shlex.join(["tar", "-czf", "-", "-C", jobdir, *exclusions, "."])
    with tempfile.TemporaryDirectory(prefix="ted-fetch-") as folder:
        archive = Path(folder) / "artifacts.tar.gz"
        with archive.open("wb") as stream:
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", settings["host"], script], stdout=stream, check=True
            )
        with tarfile.open(archive) as tar:
            tar.extractall(output / "remote", filter="data")
    checkpoint = output / "remote/artifacts" / f"agent_gen{record['generations']}"
    from .training import saved_agent

    if saved_agent(checkpoint) is None:
        raise RuntimeError(f"Remote job has no completed final checkpoint; inspect {output / 'remote'}")
    print(f"Downloaded and reloaded: {checkpoint}", flush=True)


def wait_and_fetch(record, output):
    settings = record["settings"]
    job_id = record["job_id"]
    previous = None
    while True:
        state = ssh(settings, f"squeue -h -j {job_id} -o '%T|%R'\n", capture=True)
        if not state:
            break
        if state != previous:
            print(f"Slurm job {job_id}: {state}", flush=True)
            previous = state
        time.sleep(min(settings["poll_seconds"], 30))
    accounting = ssh(settings, f"sacct -nP -j {job_id} --format=JobID,State,ExitCode,Elapsed\n", capture=True)
    (output / "slurm-accounting.txt").write_text(accounting + "\n")
    fetch(record, output)
    status = output / "remote/exit_code"
    if not status.exists() or status.read_text().strip() != "0":
        raise RuntimeError(f"Slurm job {job_id} did not complete successfully; see downloaded logs")


def run_remote(environment, config, output, *, no_wait=False, fetch_only=False):
    output = Path(output)
    settings = remote_settings(config)
    record_path = output / "remote_job.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["config_hash"] != digest(config) or record["environment"] != environment:
            raise ValueError(
                "Existing remote job uses another configuration; select a fresh output directory"
            )
        if fetch_only:
            fetch(record, output)
        elif not no_wait:
            wait_and_fetch(record, output)
        return
    if fetch_only:
        raise ValueError("No remote_job.json found in the output directory")
    record = stage(settings, environment, config, output)
    response = ssh(
        settings, f"sbatch --parsable {shlex.quote(record['jobdir'] + '/job.sbatch')}\n", capture=True
    )
    job_id = response.splitlines()[-1].split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job ID: {response}")
    record.update(job_id=job_id, generations=config["generations"])
    write_json(record_path, record)
    print(f"Submitted Slurm job {job_id}; record: {record_path}", flush=True)
    print("Rerun this command to reconnect and download; Ctrl-C leaves the remote job running.", flush=True)
    if not no_wait:
        wait_and_fetch(record, output)
