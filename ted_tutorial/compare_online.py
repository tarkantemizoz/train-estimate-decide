"""Compare simulated profits with the fixed, rounded TED Table 2 targets."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    source = Path("configs/evaluation/online_reference.json")
    reference = json.loads(source.read_text())
    targets = {item["instance_id"]: item for item in reference["instances"]}
    rows = []
    for path in sorted(args.evaluation.glob("paper-*__*.json")):
        row = json.loads(path.read_text())
        ref = targets[row["id"]]
        if any(row["model"][key] != ref[key] for key in ["p", "leadtime", "mean_demand", "std_demand"]):
            raise ValueError(f"Reference instance mismatch: {row['id']}")
        horizon = str(row["horizon"])
        metrics = row["policies"]["trained"]
        online = ref["online_profit"][horizon]
        profit = metrics["profit_mean"]
        rows.append(
            {
                "id": row["id"],
                "family": row["family"],
                "p": ref["p"],
                "leadtime": ref["leadtime"],
                "horizon": row["horizon"],
                "profit": profit,
                "standard_error": metrics["profit_standard_error"],
                "online_profit": online,
                "gap": profit - online,
                "above_rounded_target": profit > online,
                "lower_ci_above_target_plus_rounding": metrics["profit_ci95"][0] > online + 0.05,
                "paper_gc_lsn_e_profit": ref["paper_gc_lsn_e_profit"][horizon],
                "greedy_profit": row["policies"]["greedy"]["profit_mean"],
                "paired_gain_vs_greedy": row["paired_profit_gain_vs_greedy"]["mean"],
            }
        )
    if not rows:
        raise ValueError("No completed evaluation windows")
    summary = {
        "windows": len(rows),
        "expected_full_grid_windows": 64,
        "above_rounded_target": sum(row["above_rounded_target"] for row in rows),
        "lower_ci_above_target_plus_rounding": sum(
            row["lower_ci_above_target_plus_rounding"] for row in rows
        ),
        "mean_profit_gap": sum(row["gap"] for row in rows) / len(rows),
        "minimum_profit_gap": min(row["gap"] for row in rows),
        "uncertainty_note": "Pointwise intervals describe this policy's simulation uncertainty only. "
        "Online scores are rounded published estimates without raw runs or standard errors. "
        "These are not paired tests against online algorithms or simultaneous confidence intervals.",
        "reference": str(source),
        "results": rows,
    }
    (args.evaluation / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.evaluation / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    print("Below published target:")
    for row in rows:
        if not row["above_rounded_target"]:
            print(
                f"{row['id']} {row['family']} p={row['p']} L={row['leadtime']} "
                f"T={row['horizon']}: {row['profit']:.3f} vs {row['online_profit']:.1f} "
                f"(gap {row['gap']:+.3f})"
            )


if __name__ == "__main__":
    main()
