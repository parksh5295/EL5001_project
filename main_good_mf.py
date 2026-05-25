from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from uav_solar_rl.algorithms import (
    evaluate_policy,
    get_valid_actions,
    greedy_action,
    greedy_policy_from_q,
    make_q,
    masked_epsilon_greedy,
    rollout,
    value_iteration,
)
from uav_solar_rl.env import State, UAVSolarEnv


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


def print_rollout(rows, title: str):
    print(f"\nSample rollout from {title}")
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
    output_name: str = "episode_mae_vs_vi.png",
):
    x = list(range(1, len(sample_mean) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(x, sample_mean, label="sample mean", linewidth=2.0)
    for label, values in curves.items():
        plt.plot(x, values, label=label, linewidth=1.0)
    plt.xlabel("episodes")
    plt.ylabel("MAE (return) vs Value Iteration")
    plt.title("Episode-wise MAE vs VI baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / output_name, dpi=180)
    plt.close()


def epsilon_greedy_probs(
    q_values: Dict[str, float], actions: List[str], epsilon: float
) -> Dict[str, float]:
    n = len(actions)
    probs = {a: epsilon / n for a in actions}
    max_v = max(q_values[a] for a in actions)
    best = [a for a in actions if q_values[a] == max_v]
    bonus = (1.0 - epsilon) / len(best)
    for a in best:
        probs[a] += bonus
    return probs


def expected_sarsa(
    env_factory_fn,
    episodes: int = 100000,
    alpha: float = 0.08,
    alpha_end: float = 0.03,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    epsilon_decay: float = 0.9997,
    optimistic_init: float = 0.0,
    seed: int = 2,
):
    rng = random.Random(seed)
    env = env_factory_fn()
    Q = make_q(env.actions, initial_value=optimistic_init)
    epsilon = epsilon_start

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        alpha_t = max(alpha_end, alpha * (1.0 - frac))
        s = env.reset()
        done = False
        while not done:
            a = masked_epsilon_greedy(Q, env, s, epsilon, rng)
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done
            next_valid = get_valid_actions(env, ns)
            probs = epsilon_greedy_probs(Q[ns], next_valid, epsilon)
            expected_next = sum(probs[na] * Q[ns][na] for na in next_valid)
            target = r + (0.0 if done else gamma * expected_next)
            Q[s][a] += alpha_t * (target - Q[s][a])
            s = ns
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    return Q


def risk_aware_q_learning(
    env_factory_fn,
    episodes: int = 100000,
    alpha: float = 0.10,
    alpha_end: float = 0.03,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    epsilon_decay: float = 0.9995,
    optimistic_init: float = 0.0,
    risk_penalty_weight: float = 40.0,
    seed: int = 1,
):
    """Model-free risk-aware Q-learning using sampled failure signal only."""
    rng = random.Random(seed)
    env = env_factory_fn()
    Q = make_q(env.actions, initial_value=optimistic_init)
    epsilon = epsilon_start

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        alpha_t = max(alpha_end, alpha * (1.0 - frac))
        s = env.reset()
        done = False
        while not done:
            a = masked_epsilon_greedy(Q, env, s, epsilon, rng)
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done
            nofly_fail = result.info.get("failure") == "no_fly_violation"
            shaped_r = r - (risk_penalty_weight if nofly_fail else 0.0)

            next_valid = get_valid_actions(env, ns)
            if done:
                target = shaped_r
            else:
                best_next = max(Q[ns][na] for na in next_valid)
                target = shaped_r + gamma * best_next

            Q[s][a] += alpha_t * (target - Q[s][a])
            s = ns
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    return Q


def expected_sarsa_mae_curve(
    env_factory_fn,
    vi_ref_return: float,
    episodes: int = 10000,
    alpha: float = 0.08,
    alpha_end: float = 0.03,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    epsilon_decay: float = 0.9997,
    optimistic_init: float = 0.0,
    seed: int = 2,
) -> list[float]:
    rng = random.Random(seed)
    env = env_factory_fn()
    Q = make_q(env.actions, initial_value=optimistic_init)
    epsilon = epsilon_start
    curve: list[float] = []

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        alpha_t = max(alpha_end, alpha * (1.0 - frac))
        s = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            a = masked_epsilon_greedy(Q, env, s, epsilon, rng)
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done
            ep_return += r
            next_valid = get_valid_actions(env, ns)
            probs = epsilon_greedy_probs(Q[ns], next_valid, epsilon)
            expected_next = sum(probs[na] * Q[ns][na] for na in next_valid)
            target = r + (0.0 if done else gamma * expected_next)
            Q[s][a] += alpha_t * (target - Q[s][a])
            s = ns
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        curve.append(abs(ep_return - vi_ref_return))
    return curve


def risk_aware_q_learning_mae_curve(
    env_factory_fn,
    vi_ref_return: float,
    episodes: int = 10000,
    alpha: float = 0.10,
    alpha_end: float = 0.03,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    epsilon_decay: float = 0.9995,
    optimistic_init: float = 0.0,
    risk_penalty_weight: float = 40.0,
    seed: int = 1,
) -> list[float]:
    rng = random.Random(seed)
    env = env_factory_fn()
    Q = make_q(env.actions, initial_value=optimistic_init)
    epsilon = epsilon_start
    curve: list[float] = []

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        alpha_t = max(alpha_end, alpha * (1.0 - frac))
        s = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            a = masked_epsilon_greedy(Q, env, s, epsilon, rng)
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done
            nofly_fail = result.info.get("failure") == "no_fly_violation"
            shaped_r = r - (risk_penalty_weight if nofly_fail else 0.0)
            ep_return += shaped_r

            next_valid = get_valid_actions(env, ns)
            if done:
                target = shaped_r
            else:
                best_next = max(Q[ns][na] for na in next_valid)
                target = shaped_r + gamma * best_next

            Q[s][a] += alpha_t * (target - Q[s][a])
            s = ns
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        curve.append(abs(ep_return - vi_ref_return))
    return curve


def main():
    parser = argparse.ArgumentParser(
        description="DP baseline + model-free baseline + our model (risk-aware)"
    )
    parser.add_argument("--scenario", default="data/scenario.json")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--alpha-end", type=float, default=0.03)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--optimistic-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--risk-q-alpha",
        "--q-alpha",
        dest="risk_q_alpha",
        type=float,
        default=0.10,
        help="Learning rate for risk-aware Q-learning.",
    )
    parser.add_argument(
        "--exp-sarsa-alpha",
        "--sarsa-alpha",
        dest="exp_sarsa_alpha",
        type=float,
        default=0.08,
        help="Learning rate for Expected SARSA baseline.",
    )
    parser.add_argument(
        "--risk-penalty-weight",
        type=float,
        default=40.0,
        help="Extra penalty when sampled transition ends by no-fly violation.",
    )
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
        "--episode-plot-output-name",
        default="episode_mae_vs_vi.png",
        help="Output image filename for episode-wise graph.",
    )
    args = parser.parse_args()

    scenario_path = args.scenario
    out_dir = normalize_output_dir(args.output_dir)
    factory = env_factory(scenario_path)
    env = UAVSolarEnv(scenario_path)

    print("Scenario loaded")
    print(json.dumps(env.scenario, indent=2, ensure_ascii=False))
    print(f"Number of states: {len(env.enumerate_states())}")
    print(f"Actions: {env.actions}\n")

    print("Running Value Iteration...")
    vi_policy, _V = value_iteration(
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
    vi_metrics["algorithm"] = "Value Iteration (DP baseline)"

    print("Training Expected SARSA...")
    QES = expected_sarsa(
        factory,
        episodes=args.episodes,
        alpha=args.exp_sarsa_alpha,
        alpha_end=args.alpha_end,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        optimistic_init=args.optimistic_init,
        seed=2,
    )
    exp_sarsa_policy = greedy_policy_from_q(QES, env, seed=202)
    exp_sarsa_metrics = evaluate_policy(
        factory, exp_sarsa_policy, episodes=args.eval_episodes
    )
    exp_sarsa_metrics["algorithm"] = "Expected SARSA (MF baseline)"

    print("Training Risk-aware Q-learning...")
    QR = risk_aware_q_learning(
        factory,
        episodes=args.episodes,
        alpha=args.risk_q_alpha,
        alpha_end=args.alpha_end,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        optimistic_init=args.optimistic_init,
        risk_penalty_weight=args.risk_penalty_weight,
        seed=1,
    )
    risk_policy = greedy_policy_from_q(QR, env, seed=101)
    risk_metrics = evaluate_policy(factory, risk_policy, episodes=args.eval_episodes)
    risk_metrics["algorithm"] = "Risk-aware Q-learning (our model)"

    rows = [vi_metrics, exp_sarsa_metrics, risk_metrics]
    for r in rows:
        print(r)

    save_metrics(rows, out_dir / "metrics.csv")
    plot_metrics(rows, out_dir / "metrics.csv")

    if args.episode_plot:
        vi_ref = float(vi_metrics["mean_return"])
        curves = {
            "Expected SARSA": expected_sarsa_mae_curve(
                factory,
                vi_ref_return=vi_ref,
                episodes=args.episode_plot_episodes,
                alpha=args.exp_sarsa_alpha,
                alpha_end=args.alpha_end,
                gamma=args.gamma,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                optimistic_init=args.optimistic_init,
                seed=2202,
            ),
            "Risk-aware Q-learning": risk_aware_q_learning_mae_curve(
                factory,
                vi_ref_return=vi_ref,
                episodes=args.episode_plot_episodes,
                alpha=args.risk_q_alpha,
                alpha_end=args.alpha_end,
                gamma=args.gamma,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                optimistic_init=args.optimistic_init,
                risk_penalty_weight=args.risk_penalty_weight,
                seed=1101,
            ),
        }
        sample_mean = [sum(vals) / len(vals) for vals in zip(*curves.values())]
        plot_episode_mae_curves(
            out_dir,
            curves,
            sample_mean,
            output_name=args.episode_plot_output_name,
        )
        save_episode_curve_csv(out_dir / "episode_mae_vs_vi.csv", curves, sample_mean)

    sample = rollout(UAVSolarEnv(scenario_path, seed=100), risk_policy)
    print_rollout(sample, "Risk-aware Q-learning policy")
    with (out_dir / "sample_rollout.json").open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()

