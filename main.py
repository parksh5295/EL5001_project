from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
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


def normalize_output_dir(output_dir_arg: str) -> Path:
    out_dir = Path(output_dir_arg)
    if out_dir.suffix.lower() == ".json":
        out_dir = out_dir.with_suffix("")
    return out_dir


def get_vi_cache_path(output_dir: Path, scenario_path: str, args) -> Path:
    scenario_file = Path(scenario_path)
    scenario_sig = scenario_path
    if scenario_file.exists():
        scenario_sig = f"{scenario_file.resolve()}|{scenario_file.stat().st_mtime_ns}"
    cache_key = "|".join(
        [
            scenario_sig,
            f"gamma={args.gamma}",
            f"vi_state_mode={args.vi_state_mode}",
            f"vi_state_limit={args.vi_state_limit}",
            "vi_theta=1e-5",
            "vi_max_iter=500",
        ]
    )
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12]
    return output_dir / f"vi_policy_{digest}.pkl"


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
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--q-alpha", type=float, default=0.10)
    parser.add_argument("--sarsa-alpha", type=float, default=0.08)
    parser.add_argument("--alpha-end", type=float, default=0.03)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--optimistic-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.95)
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
    out_dir = normalize_output_dir(args.output_dir)
    factory = env_factory(scenario_path)
    env = UAVSolarEnv(scenario_path)
    q_alpha = args.q_alpha if args.q_alpha is not None else args.alpha
    sarsa_alpha = args.sarsa_alpha if args.sarsa_alpha is not None else args.alpha

    print("Scenario loaded")
    print(json.dumps(env.scenario, indent=2, ensure_ascii=False))
    print(f"Number of states: {len(env.enumerate_states())}")
    print(f"Actions: {env.actions}\n")

    print("Running Value Iteration...")
    vi_cache = get_vi_cache_path(out_dir, scenario_path, args)

    if vi_cache.exists():
        print("Loading cached Value Iteration policy...")
        with vi_cache.open("rb") as f:
            vi_policy = pickle.load(f)
    else:
        vi_policy, V = value_iteration(
            env,
            gamma=args.gamma,
            theta=1e-5,
            max_iter=500,
            state_mode=args.vi_state_mode,
            reachable_limit=args.vi_state_limit,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        with vi_cache.open("wb") as f:
            pickle.dump(vi_policy, f)
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
        gamma=args.gamma,
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
        gamma=args.gamma,
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
