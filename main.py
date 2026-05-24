from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from uav_solar_rl.algorithms import (
    evaluate_policy,
    greedy_policy_from_q,
    q_learning,
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
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--q-alpha", type=float, default=0.10)
    parser.add_argument("--sarsa-alpha", type=float, default=0.08)
    parser.add_argument("--alpha-end", type=float, default=0.03)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--optimistic-init", type=float, default=0.0)
    parser.add_argument(
        "--vi-state-mode",
        default="reachable",
        choices=["reachable", "enumerate"],
        help="State set mode for Value Iteration",
    )
    parser.add_argument(
        "--vi-state-limit",
        type=int,
        default=50000,
        help="Reachable-state BFS cap used when --vi-state-mode reachable",
    )
    args = parser.parse_args()

    scenario_path = args.scenario
    out_dir = Path(args.output_dir)
    factory = env_factory(scenario_path)
    env = UAVSolarEnv(scenario_path)
    q_alpha = args.q_alpha if args.q_alpha is not None else args.alpha
    sarsa_alpha = args.sarsa_alpha if args.sarsa_alpha is not None else args.alpha

    print("Scenario loaded")
    print(json.dumps(env.scenario, indent=2, ensure_ascii=False))
    print(f"Number of states: {len(env.enumerate_states())}")
    print(f"Actions: {env.actions}\n")

    print("Running Value Iteration...")
    vi_policy, V = value_iteration(
        env,
        gamma=0.95,
        theta=1e-5,
        max_iter=500,
        state_mode=args.vi_state_mode,
        reachable_limit=args.vi_state_limit,
    )
    vi_metrics = evaluate_policy(
        factory, lambda s: vi_policy.get(s, "hover"), episodes=args.eval_episodes
    )
    vi_metrics["algorithm"] = "Value Iteration"

    print("Training Q-learning...")
    Q = q_learning(
        factory,
        episodes=args.episodes,
        alpha=q_alpha,
        alpha_end=args.alpha_end,
        gamma=0.95,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        optimistic_init=args.optimistic_init,
        seed=1,
    )
    q_policy = greedy_policy_from_q(Q, env.actions, seed=101)
    q_metrics = evaluate_policy(
        factory, q_policy, episodes=args.eval_episodes
    )
    q_metrics["algorithm"] = "Q-learning"

    print("Training SARSA...")
    QS = sarsa(
        factory,
        episodes=args.episodes,
        alpha=sarsa_alpha,
        alpha_end=args.alpha_end,
        gamma=0.95,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        optimistic_init=args.optimistic_init,
        seed=2,
    )
    sarsa_policy = greedy_policy_from_q(QS, env.actions, seed=202)
    sarsa_metrics = evaluate_policy(
        factory, sarsa_policy, episodes=args.eval_episodes
    )
    sarsa_metrics["algorithm"] = "SARSA"

    rows = [vi_metrics, q_metrics, sarsa_metrics]
    for r in rows:
        print(r)

    save_metrics(rows, out_dir / "metrics.csv")
    plot_metrics(rows, out_dir / "metrics.csv")

    sample = rollout(
        UAVSolarEnv(scenario_path, seed=100), q_policy
    )
    print_rollout(sample)

    with (out_dir / "sample_rollout.json").open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
