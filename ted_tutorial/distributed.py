"""MPI collection on multiple nodes followed by one shared network fit.

This is a tutorial adapter over DynaPlex's public collection/training interfaces.
It does not claim native MPI support or single-node bitwise equivalence.
"""

import copy
import inspect
import time
import traceback
from pathlib import Path

from .training import build_experiment, digest, record_timing, saved_agent, write_json


def aggregate_shards(shards, *, generation, world_size, total_samples):
    """Concatenate honestly identified shards, retaining their exact manifests."""
    import dynaplex as dp
    import numpy as np

    if len(shards) != world_size or sum(map(len, shards)) != total_samples:
        raise ValueError("MPI shard count or total sample count differs from the requested budget")
    variable = {"config_hash", "n", "generation", "execution"}
    reference = {k: v for k, v in shards[0].manifest.items() if k not in variable}
    for rank, shard in enumerate(shards):
        manifest = shard.manifest
        if {k: v for k, v in manifest.items() if k not in variable} != reference:
            raise ValueError(f"Incompatible MPI sample manifest on rank {rank}")
        if manifest["generation"] != generation * world_size + rank:
            raise ValueError(f"Wrong RNG generation coordinate on rank {rank}")
        expected = total_samples // world_size + (rank < total_samples % world_size)
        if len(shard) != expected:
            raise ValueError(f"Wrong sample quota on rank {rank}")
    manifests = [copy.deepcopy(shard.manifest) for shard in shards]
    manifest = reference | {
        "generation": generation,
        "n": total_samples,
        "config_hash": digest(
            {
                "adapter": "ted-mpi-v1",
                "generation": generation,
                "shards": [s["config_hash"] for s in manifests],
            }
        ),
        "distributed": {
            "adapter": "ted-mpi-v1",
            "world_size": world_size,
            "seed_rule": "fixed seed; collection generation = logical generation * ranks + rank",
            "shards": manifests,
        },
        "execution": {"merged": [s["execution"] for s in manifests]},
    }
    # Local controller IDs remain unchanged. A separate rank column makes provenance unique.
    if any(shard.info is None for shard in shards):
        raise ValueError("MPI sample provenance is required")
    info = np.concatenate(
        [
            np.column_stack((np.full(len(shard), rank, dtype=np.int64), shard.info))
            for rank, shard in enumerate(shards)
        ]
    )
    manifest["info_columns"] = ["mpi_rank", *reference["info_columns"]]
    return dp.SampleSet(
        features={
            key: np.concatenate([shard.features[key] for shard in shards]) for key in shards[0].features
        },
        **{
            key: np.concatenate([getattr(shard, key) for shard in shards])
            for key in ("mask", "y", "q_sum", "z", "q_count")
        },
        manifest=manifest,
        info=info,
    )


def run_mpi(environment, config, output):
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ValueError("MPI mode requires mpi4py and an MPI runtime; see docs/training.md") from exc
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    try:
        if size < 2:
            raise ValueError("MPI mode requires at least two ranks, with one rank per node")
        return _run(comm, rank, size, environment, config, Path(output))
    except BaseException:
        traceback.print_exc()
        comm.Abort(1)
        raise


def _run(comm, rank, size, environment, config, output):
    import dynaplex as dp
    import torch

    started = time.perf_counter()
    identities = comm.allgather(digest(config))
    if len(set(identities)) != 1:
        raise ValueError("MPI ranks have different requested configurations")
    total = config.get("dcl", {}).get("n", inspect.signature(dp.DCL).parameters["n"].default)
    quota = total // size + (rank < total % size)
    if quota < 2:
        raise ValueError("Each MPI rank needs at least two samples")
    local_config = copy.deepcopy(config)
    local_config.setdefault("dcl", {})["n"] = quota
    experiment = build_experiment(environment, local_config, output / "ranks" / f"rank{rank:04d}")
    if not experiment.sample_info:
        raise ValueError("MPI aggregation requires dcl.sample_info=true")
    if experiment.plan is not None:
        from dynaplex.machine import verify_plan

        verify_plan(experiment.plan)
    if rank == 0:
        identity = {"environment": environment, "config": config, "world_size": size}
        identity_path = output / "mpi_config.json"
        if identity_path.exists():
            import json

            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("MPI run configuration changed; select a fresh output directory")
        else:
            write_json(identity_path, identity)
        print(f"MPI: {size} ranks; total N={total:,} per generation", flush=True)
    comm.Barrier()
    stages = []
    for generation in range(config["generations"]):
        agent_path = output / f"agent_gen{generation + 1}"
        complete = saved_agent(agent_path) is not None if rank == 0 else None
        if comm.bcast(complete, root=0):
            continue
        generation_start = time.perf_counter()
        shard_path = output / f"shards_gen{generation}" / f"rank{rank:04d}"
        if not shard_path.exists():
            if experiment.torch_threads is not None:
                torch.set_num_threads(experiment.torch_threads)
            agent = None if generation == 0 else saved_agent(output / f"agent_gen{generation}")
            if agent is not None:
                agent.to(experiment.device)
                if experiment.compile_agent:
                    agent.compile()
            shape = experiment.shape(generation)
            params = {
                key: getattr(experiment, key)
                for key in (
                    "n",
                    "m",
                    "h",
                    "seed",
                    "reduction_factor",
                    "final_field",
                    "warmup_time",
                    "restart_after",
                    "steps_per_round",
                    "jit",
                    "gc_threshold_bytes",
                    "policy_binding",
                    "policy_precision",
                    "torch_threads",
                    "sample_info",
                )
            }
            samples = dp.collect_samples(
                experiment.mdp,
                experiment.policy,
                features=experiment.features,
                **params,
                generation=generation * size + rank,
                rollout_agent=agent,
                processes=shape.processes,
                controllers=shape.engines,
                workers=shape.workers,
                slots=shape.slots,
                verbose=rank == 0,
                checks=experiment.checks if generation == 0 else False,
                provenance={"mpi_rank": rank, "mpi_size": size, "logical_generation": generation},
            )
            samples.save(str(shard_path))
            del samples
        comm.Barrier()
        if rank == 0:
            shards = [
                dp.SampleSet.load(str(shard_path.parent / f"rank{r:04d}"), mmap=True) for r in range(size)
            ]
            merged_path = output / f"samples_gen{generation}"
            if merged_path.exists():
                samples = dp.SampleSet.load(str(merged_path), mmap=True)
            else:
                samples = aggregate_shards(
                    shards, generation=generation, world_size=size, total_samples=total
                )
                samples.save(str(merged_path))
            collected_seconds = time.perf_counter() - generation_start
            fit_start = time.perf_counter()
            if experiment.torch_threads is not None:
                torch.set_num_threads(experiment.torch_threads)
            train = dict(experiment.train)
            train.setdefault("seed", experiment.seed)
            train.setdefault("device", experiment.train_device)
            agent = dp.train_network(samples, network=experiment.network or dp.MLP(), **train)
            agent.save(str(agent_path))
            saved_agent(agent_path)
            stages.append(
                {
                    "generation": generation + 1,
                    "collect_and_merge_seconds": collected_seconds,
                    "fit_and_save_seconds": time.perf_counter() - fit_start,
                }
            )
            del samples, shards
        comm.Barrier()
    if rank == 0:
        result = {
            "elapsed_seconds": time.perf_counter() - started,
            "stages": stages,
            "checkpoint": f"agent_gen{config['generations']}",
            "checkpoint_saved": True,
            "world_size": size,
        }
        record_timing(output, result)
        print(f"Ready: {output / result['checkpoint']}", flush=True)
        return result
