"""Evaluate heuristics, and optionally a trained agent, on the lost-sales cases; run it with the root evaluate.py:

    python evaluate.py lost_sales/zero_shot_stationary --agent reference/demo --cases online --modes demand
    python evaluate.py lost_sales/zero_shot_large --agent paper/gc_lsn --cases case1
    python evaluate.py lost_sales/single_instance --agent laptop/demo

--agent is a run in <environment>/train_runs, whose last generation is used, an agent_gen<g> folder, or a path.
A single-instance agent is evaluated on the instance in its config: a case instance, or else one named "custom",
whose BS and CBS are tuned on first use and stored with the results.
Every evaluation includes the greedy policy DCL starts from and the tuned BS and CBS benchmarks.
Modes: "known" parameters, or estimated from observations: "demand", "leadtime" or "both" (Case 3).
With known parameters a policy is scored over KNOWN_HORIZON periods after WARMUP periods.
With estimated parameters it starts cold and is scored over the first T periods, for each T in HORIZONS.
Every policy is also compared with CBS on the same trajectories: the paired difference and its standard error.
As each window finishes, a line says whether the trained policy, or greedy without --agent, is better than the best
published online method, where there is one, and else than CBS, or too close to call.
Results go to <environment>/eval_runs/<run>/gen<g>/<case>.csv, or to eval_runs/benchmarks without --agent.
Repeating an interrupted command resumes it.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import dynaplex as dp

import envs

from . import cases
from .cases.online import published_scores
from .cases.tune import TABLE, benchmarks

MODES = ("known", "demand", "leadtime", "both")
# The modes scored by the negative profit instead of the cost.
PROFIT_MODES = ("demand", "both")
# A single-instance agent's instance when it is not a case instance.
CUSTOM = "custom"
# The paper's evaluation protocol; the command-line options override it.
TRAJECTORIES = 1000
KNOWN_HORIZON = 5000
WARMUP = 100
HORIZONS = (200, 500, 1000, 2000)
SEED = 13021984
# The two-sided 95% quantile of the normal distribution.
Z95 = 1.96
# The published online scores are rounded to 0.1.
ROUNDING = 0.05
FIELDS = (
    "instance",
    "mode",
    "horizon",
    "warmup",
    "policy",
    "mean",
    "error",
    "delta_cbs",
    "delta_cbs_error",
    "true_cost",
)


def side(difference: float, margin: float) -> int:
    """1 if difference ± margin lies above zero, -1 if below, and 0 if it contains zero."""
    return (difference - margin > 0) - (difference + margin < 0)


def find_agent(folder: Path, name: str) -> Path:
    """An agent folder, from a run in the environment's train_runs (its last generation) or a path."""
    path = Path(name)
    if not path.exists():
        path = folder / "train_runs" / name
    if not (path / "agent.json").exists():
        generations = sorted(
            path.glob("agent_gen*"), key=lambda item: int(item.name.removeprefix("agent_gen"))
        )
        if not generations:
            raise FileNotFoundError(f"no agent in {path}")
        path = generations[-1]
    return path


def output_folder(folder: Path, agent: Path | None) -> Path:
    if agent is None:
        return folder / "eval_runs" / "benchmarks"
    run = agent.resolve().parent.relative_to((folder / "train_runs").resolve())
    return folder / "eval_runs" / run / agent.name.removeprefix("agent_")


def label(folder: Path, agent: Path | None) -> str | None:
    """The agent in settings.json: relative to the environment, so that a moved checkout still resumes."""
    if agent is None:
        return None
    if agent.resolve().is_relative_to(folder.resolve()):
        return agent.resolve().relative_to(folder.resolve()).as_posix()
    return str(agent.resolve())


def assess(model, policies: dict, horizon: int, warmup: int, args) -> dict:
    """Simulate all policies on common random numbers; the result of each."""
    comparer = dp.PolicyComparer(
        model,
        number_of_trajectories=args.trajectories,
        horizon=horizon,
        warmup_time=warmup,
        seed=args.seed,
        jit=True,
    )
    with envs.sweeping_at_safe_points():
        if len(policies) == 1:
            [(name, policy)] = policies.items()
            return {name: comparer.assess(policy, name=name)}
        return {result.name: result for result in comparer.compare(policies)}


def window_rows(results: dict) -> list[dict]:
    """A row per policy, with its difference from CBS paired per trajectory.

    A trajectory has the same random numbers for every policy, even on another model of the instance, such
    as the one BS and CBS run on, so the paired difference is more precise than the scores' errors suggest.
    """
    benchmark = results["CBS"].stats
    rows = []
    for name, result in results.items():
        stats = result.stats
        delta = (
            stats.cumulative_cost / stats.time_elapsed - benchmark.cumulative_cost / benchmark.time_elapsed
        )
        rows.append(
            {
                "policy": name,
                "mean": result.mean,
                "error": result.error,
                "delta_cbs": float(delta.mean()),
                "delta_cbs_error": float(delta.std(ddof=1) / math.sqrt(len(delta))),
                "true_cost": float((stats.true_cost / stats.time_elapsed).mean()),
            }
        )
    return rows


def window_line(
    instance: str, mode: str, horizon: int, rows: list[dict], policy: str, published: dict
) -> str:
    """Whether the policy beats the best published online method in a window, where there is one, and else CBS.

    It reports profit with estimated demand, as the online learning literature does, and cost otherwise.
    """
    profit = mode in PROFIT_MODES
    sign = -1 if profit else 1
    row = next(row for row in rows if row["policy"] == policy)
    score = sign * row["mean"]
    reference = published.get(instance, {}).get(str(horizon)) if mode == "demand" else None
    if reference is not None:
        half_width = Z95 * row["error"]
        outcome = side(score - reference, half_width + ROUNDING)
        comparison = f"{score:.2f} ± {half_width:.2f} vs best online {reference:.1f}"
        rival = "best online"
    else:
        cbs = next(row for row in rows if row["policy"] == "CBS")
        half_width = Z95 * row["delta_cbs_error"]
        outcome = -side(row["delta_cbs"], half_width)
        comparison = (
            f"{score:.3f} vs CBS {sign * cbs['mean']:.3f}, "
            f"difference {sign * row['delta_cbs']:+.3f} ± {half_width:.3f}"
        )
        rival = "CBS"
    verdict = {1: f"{policy} better", -1: f"{rival} better", 0: "too close to call"}[outcome]
    unit = "profit" if profit else "cost"
    return f"{instance:<10} {mode:<8} T={horizon:<5} {unit:<6} {policy} {comparison}: {verdict}"


def evaluate_case(
    environment, instances, modes, agent, dimensions: dict, table: Path, args, path: Path
) -> None:
    done = set()
    if path.exists():
        with path.open(newline="") as stream:
            done = {(row["instance"], row["mode"], int(row["horizon"])) for row in csv.DictReader(stream)}
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if stream.tell() == 0:
            writer.writeheader()
        policy = "trained" if agent else "greedy"
        published = published_scores()
        for instance in instances:
            for mode in [mode for mode in instance.modes if mode in modes]:
                windows = (
                    [(args.known_horizon, args.warmup)]
                    if mode == "known"
                    else [(horizon, 0) for horizon in args.horizons]
                )
                windows = [
                    (horizon, warmup)
                    for horizon, warmup in windows
                    if (instance.id, mode, horizon) not in done
                ]
                if not windows:
                    continue
                model = environment.MDP(**environment.instance_config(instance, mode, dimensions))
                greedy = environment.SweepingBasePolicy(model)
                policies = {"greedy": greedy} | ({"trained": agent} if agent else {})
                # BS and CBS run on zero_shot_large, which draws the same demand for the same seed.
                oracle, rules = benchmarks(instance, mode, table)
                for horizon, warmup in windows:
                    results = assess(model, policies, horizon, warmup, args)
                    results |= assess(oracle, rules, horizon, warmup, args)
                    window = {"instance": instance.id, "mode": mode, "horizon": horizon, "warmup": warmup}
                    rows = window_rows(results)
                    writer.writerows(window | row for row in rows)
                    stream.flush()
                    print(window_line(instance.id, mode, horizon, rows, policy, published), flush=True)


def summarize(path: Path) -> None:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["mode"], int(row["horizon"]), row["policy"])].append(row)
    print(
        f"\n{path.stem}: means over the instances; lower is better, and negative profit with estimated\n"
        "demand. Against CBS, paired per trajectory: the mean difference, and the instances where a policy\n"
        "is better or worse, with the 95% confidence interval of the difference below or above zero."
    )
    print(
        f"  {'mode':<8} {'T':<6} {'policy':<8} {'mean':>10} {'- CBS':>9} {'better':>7} {'worse':>6} {'of':>4}"
    )
    for (mode, horizon, policy), group in sorted(groups.items()):
        mean = sum(float(row["mean"]) for row in group) / len(group)
        line = f"  {mode:<8} {horizon:<6} {policy:<8} {mean:>10.3f}"
        if policy != "CBS":
            deltas = [float(row["delta_cbs"]) for row in group]
            sides = [side(float(row["delta_cbs"]), Z95 * float(row["delta_cbs_error"])) for row in group]
            line += f" {sum(deltas) / len(deltas):>+9.3f} {sides.count(-1):>7} {sides.count(1):>6}"
        else:
            line += f" {'':>9} {'':>7} {'':>6}"
        print(f"{line} {len(group):>4}")
    if path.stem != "online":
        return
    reference = published_scores()
    trained = [
        row
        for row in rows
        if row["policy"] == "trained"
        and row["mode"] == "demand"
        and row["horizon"] in reference[row["instance"]]
    ]
    if trained:
        # The published scores are rounded, without standard errors.
        margins = [-float(row["mean"]) - reference[row["instance"]][row["horizon"]] for row in trained]
        wins = sum(margin > 0 for margin in margins)
        robust = sum(
            side(margin, Z95 * float(row["error"]) + ROUNDING) == 1
            for margin, row in zip(margins, trained, strict=True)
        )
        print(
            f"The trained network beats the best published online method in {wins} of {len(trained)} windows, "
            f"{robust} of them with its 95% lower bound above the rounded score."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("environment", choices=envs.names("lost_sales"))
    parser.add_argument("--agent", help="a trained agent (see above); without it, heuristics only")
    parser.add_argument("--cases", nargs="+", help="default: the environment's DEFAULT_CASES")
    parser.add_argument("--instances", nargs="+", help="only these instance IDs, e.g. case1-000 online-003")
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, help="default: every mode an instance is evaluated in"
    )
    parser.add_argument("--trajectories", type=int, default=TRAJECTORIES)
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(HORIZONS), help="T, estimated parameters"
    )
    parser.add_argument(
        "--known-horizon", type=int, default=KNOWN_HORIZON, help="scored periods, known parameters"
    )
    parser.add_argument("--warmup", type=int, default=WARMUP, help="unscored periods first, known parameters")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, help="default: derived from --agent")
    args = parser.parse_args()
    environment = envs.find(args.environment)
    folder = Path(environment.__file__).parent

    agent, path, trained = None, None, {}
    if args.agent:
        try:
            path = find_agent(folder, args.agent)
        except FileNotFoundError as error:
            parser.error(str(error))
        if not (path.parent / "config.json").exists():
            parser.error(f"{path.parent} has no config.json; train.py writes one next to the agents")
        trained = json.loads((path.parent / "config.json").read_text())
        if "mdp" not in trained:
            parser.error(f'{path.parent / "config.json"} has no "mdp" section; retrain with train.py')
        agent = dp.NNAgent.load(str(path))
    if unsupported := set(args.modes or ()) - set(environment.MODES):
        parser.error(f"{args.environment} simulates modes {environment.MODES}, not {sorted(unsupported)}")
    modes = [mode for mode in environment.MODES if not args.modes or mode in args.modes]
    if agent and not environment.SUPER_MDP:
        # A single-instance agent applies to the instance in its config only.
        own = cases.with_parameters(trained["mdp"]) or cases.Instance(CUSTOM, trained["mdp"], ("known",))
        if (args.cases and cases.case_of(own.id) not in args.cases) or (
            args.instances and set(args.instances) != {own.id}
        ):
            parser.error(f"the agent was trained for {own.id} only")
        selected = {cases.case_of(own.id): [own]}
    else:
        own = None
        ids = args.instances
        names = args.cases or (
            sorted({cases.case_of(i) for i in ids}) if ids else list(environment.DEFAULT_CASES)
        )
        if unknown := set(names) - set(cases.CASES):
            parser.error(f"unknown cases {sorted(unknown)}; choose from {sorted(cases.CASES)}")
        if unsupported := set(names) - set(environment.CASES):
            parser.error(
                f"{args.environment} cannot represent {sorted(unsupported)}; it covers {environment.CASES}"
            )
        selected = {
            case: [
                item
                for item in cases.CASES[case]()
                if (not ids or item.id in ids) and set(item.modes) & set(modes)
            ]
            for case in names
        }
        if missing := set(ids or ()) - {item.id for items in selected.values() for item in items}:
            parser.error(f"not in the selected cases and modes: {sorted(missing)}")
        if not any(selected.values()):
            parser.error(f"no selected instance is evaluated in modes {modes}")

    try:
        output = args.output or output_folder(folder, path)
    except ValueError:
        parser.error(f"--agent is outside {args.environment}/train_runs; choose a folder with --output")
    settings = {
        "agent": label(folder, path),
        "trajectories": args.trajectories,
        "horizons": args.horizons,
        "known_horizon": args.known_horizon,
        "warmup": args.warmup,
        "seed": args.seed,
    }
    output.mkdir(parents=True, exist_ok=True)
    record = output / "settings.json"
    if record.exists() and json.loads(record.read_text()) != settings:
        parser.error(f"{output} holds results for other settings; choose another folder with --output")
    for case in selected:
        results = output / f"{case}.csv"
        if results.exists() and results.read_text().partition("\n")[0] not in ("", ",".join(FIELDS)):
            parser.error(
                f"{results} has the columns of an older evaluate.py; choose another folder with --output"
            )
    record.write_text(json.dumps(settings, indent=2) + "\n")
    dimensions = trained["mdp"] if agent else environment.DIMENSIONS
    # The BS and CBS levels of a custom instance are stored with its results.
    table = output / "benchmark_parameters.csv" if own and own.id == CUSTOM else TABLE
    for case, instances in selected.items():
        if instances:
            evaluate_case(
                environment, instances, modes, agent, dimensions, table, args, output / f"{case}.csv"
            )
            summarize(output / f"{case}.csv")


if __name__ == "__main__":
    main()
