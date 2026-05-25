from __future__ import annotations

import argparse
import csv

import json

from pathlib import Path

import matplotlib.pyplot as plt

from uav_solar_rl.algorithms import (
    evaluate_policy,
    greedy_policy_from_q,
    q_learning_mae_curve,
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


def save_episode_curve_csv(out_path: Path, curves: dict[str, list[float]], sample_mean: list[float]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(curves.keys())
    rows = zip(*([sample_mean] + [curves[k] for k in keys]))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "sample_mean", *keys])
        for idx, row in enumerate(rows, start=1):
            writer.writerow([idx, *row])


def plot_episode_mae_curves(
    out_dir: Path,
    curves: dict[str, list[float]],
    sample_mean: list[float],
    switch_episode: int | None = None,
    output_name: str = "episode_mae_vs_vi.png",
):
    x = list(range(1, len(sample_mean) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(x, sample_mean, label="sample mean", linewidth=2.0)
    for label, values in curves.items():
        plt.plot(x, values, label=label, linewidth=1.0)
    if switch_episode is not None:
        plt.axvline(
            switch_episode,
            linestyle="--",
            linewidth=2.0,
            color="gray",
            label="env switch",
        )
    plt.xlabel("episodes")
    plt.ylabel("MAE (return) vs Value Iteration")
    plt.title("Episode-wise MAE (Q-learning) vs VI baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / output_name, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="3D UAV solar inspection MDP with Value Iteration, Q-learning, and SARSA"
    )
    parser.add_argument("--scenario", default="data/scenario.json")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--q-alpha", type=float, default=0.10)
    parser.add_argument("--sarsa-alpha", type=float, default=0.08)
    parser.add_argument("--alpha-end", type=float, default=0.03)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--optimistic-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--vi-state-mode",
        default="enumerate",
        choices=["reachable", "enumerate"],
        help="State set mode for Value Iteration",
    )
    parser.add_argument(
        "--vi-state-limit",
        type=int,
        default=200000,
        help="Reachable-state BFS cap used when --vi-state-mode reachable",
    )
    parser.add_argument(
        "--episode-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw episode-wise MAE graph vs VI (default: on).",
    )
    parser.add_argument(
        "--episode-plot-episodes",
        type=int,
        default=10000,
        help="Episodes used to draw episode-wise MAE graph.",
    )
    parser.add_argument(
        "--episode-plot-alphas",
        default="0.1,0.5",
        help="Comma-separated alpha values used for episode graph.",
    )
    parser.add_argument(
        "--episode-plot-switch-episode",
        type=int,
        default=None,
        help="Optional env-switch episode marker for graph.",
    )
    parser.add_argument(
        "--episode-plot-scenario-after",
        default=None,
        help="Optional scenario path used after switch episode.",
    )
    parser.add_argument(
        "--episode-plot-output-name",
        default="episode_mae_vs_vi.png",
        help="Output image filename for episode-wise graph.",
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

    vi_policy, V = value_iteration(
    env,
    gamma=args.gamma,
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
    gamma=args.gamma,
    epsilon_start=args.epsilon_start,
    epsilon_end=args.epsilon_end,
    optimistic_init=args.optimistic_init,
    seed=1,
)

    q_policy = greedy_policy_from_q(Q,env, seed=101)

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

    sarsa_policy = greedy_policy_from_q(QS, env, seed=202)

    sarsa_metrics = evaluate_policy(
    factory, sarsa_policy, episodes=args.eval_episodes
)
    sarsa_metrics["algorithm"] = "SARSA"

    rows = [vi_metrics, q_metrics, sarsa_metrics]
    for r in rows:
        print(r)

    save_metrics(rows, out_dir / "metrics.csv")
    plot_metrics(rows, out_dir / "metrics.csv")

    if args.episode_plot:
        alpha_values = [
            float(x.strip())
            for x in args.episode_plot_alphas.split(",")
            if x.strip()
        ]
        if alpha_values:
            curves: dict[str, list[float]] = {}
            vi_ref = float(vi_metrics["mean_return"])
            for idx, alpha_for_curve in enumerate(alpha_values):
                label = f"alpha={alpha_for_curve:g}"
                curves[label] = q_learning_mae_curve(
                    factory,
                    vi_ref_return=vi_ref,
                    episodes=args.episode_plot_episodes,
                    alpha=alpha_for_curve,
                    alpha_end=args.alpha_end,
                    gamma=args.gamma,
                    epsilon_start=args.epsilon_start,
                    epsilon_end=args.epsilon_end,
                    optimistic_init=args.optimistic_init,
                    seed=9001 + idx * 1000,
                    scenario_after=args.episode_plot_scenario_after,
                    switch_episode=args.episode_plot_switch_episode,
                )
            sample_mean = [
                sum(vals) / len(vals)
                for vals in zip(*curves.values())
            ]
            plot_episode_mae_curves(
                out_dir=out_dir,
                curves=curves,
                sample_mean=sample_mean,
                switch_episode=args.episode_plot_switch_episode,
                output_name=args.episode_plot_output_name,
            )
            save_episode_curve_csv(
                out_dir / "episode_mae_vs_vi.csv",
                curves,
                sample_mean,
            )

    sample = rollout(
        UAVSolarEnv(scenario_path, seed=100), q_policy
    )
    print_rollout(sample)

    with (out_dir / "sample_rollout.json").open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
