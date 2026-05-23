from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from uav_solar_rl.algorithms import (
    evaluate_policy,
    q_learning,
    q_to_policy,
    rollout,
    sarsa,
    value_iteration,
)
from uav_solar_rl.env import UAVSolarEnv


def env_factory(scenario_path: str, seed: int = 0):
    return lambda: UAVSolarEnv(scenario_path=scenario_path, seed=seed)


def save_metrics(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "mean_return",
        "std_return",
        "success_rate",
        "avg_steps",
        "battery_failures",
        "no_fly_violations",
        "restricted_visits",
        "charge_count",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [r["algorithm"] for r in rows]
    returns = [r["mean_return"] for r in rows]
    success = [r["success_rate"] for r in rows]

    plt.figure()
    plt.bar(names, returns)
    plt.ylabel("Mean Return")
    plt.title("Algorithm Comparison: Mean Return")
    plt.tight_layout()
    plt.savefig(out_path.parent / "mean_return.png", dpi=180)
    plt.close()

    plt.figure()
    plt.bar(names, success)
    plt.ylabel("Success Rate")
    plt.title("Algorithm Comparison: Success Rate")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path.parent / "success_rate.png", dpi=180)
    plt.close()


def print_rollout(rows):
    print("\nSample rollout from learned Q-learning policy")
    print("t | state | action | reward | info")
    print("-" * 90)
    for row in rows[:30]:
        print(
            f"{row['t']:2d} | {row['state']} | {row['action']:8s} | {row['reward']:7.1f} | {row['info']}"
        )
    if len(rows) > 30:
        print("... truncated ...")


def main():
    parser = argparse.ArgumentParser(
        description="3D UAV solar inspection MDP with Value Iteration, Q-learning, and SARSA"
    )
    parser.add_argument("--scenario", default="data/scenario.json")
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    scenario_path = args.scenario
    out_dir = Path(args.output_dir)
    factory = env_factory(scenario_path)
    env = UAVSolarEnv(scenario_path)

    print("Scenario loaded")
    print(json.dumps(env.scenario, indent=2, ensure_ascii=False))
    print(f"Number of states: {len(env.enumerate_states())}")
    print(f"Actions: {env.actions}\n")

    print("Running Value Iteration...")
    vi_policy, V = value_iteration(env, gamma=0.95, theta=1e-5, max_iter=500)
    vi_metrics = evaluate_policy(
        factory, lambda s: vi_policy.get(s, "hover"), episodes=args.eval_episodes
    )
    vi_metrics["algorithm"] = "Value Iteration"

    print("Training Q-learning...")
    Q = q_learning(factory, episodes=args.episodes, alpha=0.10, gamma=0.95, seed=1)
    q_policy = q_to_policy(Q, env.actions)
    q_metrics = evaluate_policy(
        factory, lambda s: q_policy.get(s, "hover"), episodes=args.eval_episodes
    )
    q_metrics["algorithm"] = "Q-learning"

    print("Training SARSA...")
    QS = sarsa(factory, episodes=args.episodes, alpha=0.08, gamma=0.95, seed=2)
    sarsa_policy = q_to_policy(QS, env.actions)
    sarsa_metrics = evaluate_policy(
        factory, lambda s: sarsa_policy.get(s, "hover"), episodes=args.eval_episodes
    )
    sarsa_metrics["algorithm"] = "SARSA"

    rows = [vi_metrics, q_metrics, sarsa_metrics]
    for r in rows:
        print(r)

    save_metrics(rows, out_dir / "metrics.csv")
    plot_metrics(rows, out_dir / "metrics.csv")

    sample = rollout(
        UAVSolarEnv(scenario_path, seed=100), lambda s: q_policy.get(s, "hover")
    )
    print_rollout(sample)

    with (out_dir / "sample_rollout.json").open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
