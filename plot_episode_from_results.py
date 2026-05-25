from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uav_solar_rl.algorithms import q_learning_mae_curve
from uav_solar_rl.env import UAVSolarEnv


def read_vi_mean_return(results_dir: Path) -> float:
    metrics_path = results_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("algorithm") == "Value Iteration":
                return float(row["mean_return"])
    raise ValueError("Value Iteration row not found in metrics.csv")


def save_curve_csv(out_csv: Path, curves: dict[str, list[float]], sample_mean: list[float]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(curves.keys())
    rows = zip(*([sample_mean] + [curves[k] for k in keys]))
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "sample_mean", *keys])
        for idx, row in enumerate(rows, start=1):
            writer.writerow([idx, *row])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw episode-wise MAE graph vs VI from results directory."
    )
    parser.add_argument("--results-dir", required=True, help="Directory containing metrics.csv")
    parser.add_argument("--scenario", required=True, help="Scenario path used by main.py run")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--alphas", default="0.1,0.5", help="Comma-separated alpha values")
    parser.add_argument("--alpha-end", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--optimistic-init", type=float, default=0.0)
    parser.add_argument("--switch-episode", type=int, default=None)
    parser.add_argument("--scenario-after", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-name", default="episode_mae_vs_vi.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    vi_ref = read_vi_mean_return(results_dir)
    alpha_values = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if not alpha_values:
        raise ValueError("No valid alpha values were provided.")

    curves: dict[str, list[float]] = {}
    for idx, alpha in enumerate(alpha_values):
        label = f"alpha={alpha:g}"
        curves[label] = q_learning_mae_curve(
            env_factory=lambda: UAVSolarEnv(scenario_path=args.scenario, seed=0),
            vi_ref_return=vi_ref,
            episodes=args.episodes,
            alpha=alpha,
            alpha_end=args.alpha_end,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            optimistic_init=args.optimistic_init,
            seed=args.seed + idx * 1000,
            scenario_after=args.scenario_after,
            switch_episode=args.switch_episode,
        )

    values = np.array(list(curves.values()), dtype=float)
    sample_mean = values.mean(axis=0).tolist()

    x = np.arange(1, args.episodes + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, sample_mean, label="sample mean", linewidth=2.0)
    for label, y in curves.items():
        plt.plot(x, y, label=label, linewidth=1.0)

    if args.switch_episode is not None:
        plt.axvline(args.switch_episode, linestyle="--", linewidth=2.0, color="gray", label="env switch")

    plt.xlabel("episodes")
    plt.ylabel("MAE vs VI return")
    plt.title("Episode-wise MAE (Q-learning) vs Value Iteration baseline")
    plt.legend()
    plt.tight_layout()

    out_png = results_dir / args.output_name
    plt.savefig(out_png, dpi=180)
    plt.close()

    out_csv = results_dir / "episode_mae_vs_vi.csv"
    save_curve_csv(out_csv, curves, sample_mean)

    print(f"Saved figure: {out_png}")
    print(f"Saved curve csv: {out_csv}")


if __name__ == "__main__":
    main()

