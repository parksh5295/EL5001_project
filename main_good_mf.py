from __future__ import annotations

import argparse
import csv
import json
import pickle
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


def rolling_mean(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    w = max(1, window)
    out: List[float] = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= w:
            running -= values[i - w]
        denom = min(i + 1, w)
        out.append(running / denom)
    return out


def save_training_trace_csv(
    out_path: Path,
    expected_returns: List[float],
    expected_success: List[float],
    risk_returns: List[float],
    risk_success: List[float],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "expected_sarsa_return",
                "expected_sarsa_success",
                "risk_aware_return",
                "risk_aware_success",
            ]
        )
        for i, row in enumerate(
            zip(expected_returns, expected_success, risk_returns, risk_success), start=1
        ):
            writer.writerow([i, *row])


def plot_learning_curves(
    out_dir: Path,
    expected_returns: List[float],
    expected_success: List[float],
    risk_returns: List[float],
    risk_success: List[float],
    window: int,
    vi_mean_return: float,
    vi_success_rate: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    x = list(range(1, len(expected_returns) + 1))

    exp_return_smooth = rolling_mean(expected_returns, window)
    risk_return_smooth = rolling_mean(risk_returns, window)

    plt.figure(figsize=(10, 5))
    plt.plot(x, exp_return_smooth, label="Expected SARSA", linewidth=1.5)
    plt.plot(x, risk_return_smooth, label="Risk-aware Q-learning", linewidth=1.5)
    plt.axhline(
        y=vi_mean_return,
        linestyle="--",
        color="gray",
        linewidth=1.2,
        label="VI eval reference",
    )
    plt.xlabel("episodes")
    plt.ylabel(f"return (rolling mean, window={window})")
    plt.title("Learning Curve: Episode Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_return.png", dpi=180)
    plt.close()

    exp_success_smooth = rolling_mean(expected_success, window)
    risk_success_smooth = rolling_mean(risk_success, window)

    plt.figure(figsize=(10, 5))
    plt.plot(x, exp_success_smooth, label="Expected SARSA", linewidth=1.5)
    plt.plot(x, risk_success_smooth, label="Risk-aware Q-learning", linewidth=1.5)
    plt.axhline(
        y=vi_success_rate,
        linestyle="--",
        color="gray",
        linewidth=1.2,
        label="VI eval reference",
    )
    plt.xlabel("episodes")
    plt.ylabel(f"success rate (rolling mean, window={window})")
    plt.title("Learning Curve: Episode Success Rate")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_success.png", dpi=180)
    plt.close()


def _qtable_to_plain(model_q: dict) -> dict:
    return {state: dict(action_values) for state, action_values in model_q.items()}


def save_run_artifacts(
    out_dir: Path,
    args: argparse.Namespace,
    rows: list[dict],
    vi_policy: dict,
    vi_values: dict,
    expected_q: dict,
    risk_q: dict,
    expected_returns: List[float],
    expected_success: List[float],
    risk_returns: List[float],
    risk_success: List[float],
):
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with (artifacts_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "metrics_rows": rows,
                "num_episodes_logged": len(expected_returns),
                "notes": "Use model_tables.pkl and training traces for additional offline plotting.",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with (artifacts_dir / "model_tables.pkl").open("wb") as f:
        pickle.dump(
            {
                "vi_policy": dict(vi_policy),
                "vi_values": dict(vi_values),
                "expected_sarsa_q": _qtable_to_plain(expected_q),
                "risk_aware_q": _qtable_to_plain(risk_q),
            },
            f,
        )

    with (artifacts_dir / "training_traces.pkl").open("wb") as f:
        pickle.dump(
            {
                "expected_returns": expected_returns,
                "expected_success": expected_success,
                "risk_returns": risk_returns,
                "risk_success": risk_success,
            },
            f,
        )


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
    return_trace: List[float] | None = None,
    success_trace: List[float] | None = None,
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
        ep_return = 0.0
        ep_success = 0.0
        while not done:
            a = masked_epsilon_greedy(Q, env, s, epsilon, rng)
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done
            ep_return += r
            if result.info.get("success"):
                ep_success = 1.0
            next_valid = get_valid_actions(env, ns)
            probs = epsilon_greedy_probs(Q[ns], next_valid, epsilon)
            expected_next = sum(probs[na] * Q[ns][na] for na in next_valid)
            target = r + (0.0 if done else gamma * expected_next)
            Q[s][a] += alpha_t * (target - Q[s][a])
            s = ns
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if return_trace is not None:
            return_trace.append(ep_return)
        if success_trace is not None:
            success_trace.append(ep_success)
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
    return_trace: List[float] | None = None,
    success_trace: List[float] | None = None,
):
    # Risk-aware Q-learning using sampled failure signal only.
    rng = random.Random(seed)
    env = env_factory_fn()
    Q = make_q(env.actions, initial_value=optimistic_init)
    epsilon = epsilon_start

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        alpha_t = max(alpha_end, alpha * (1.0 - frac))
        s = env.reset()
        done = False
        ep_return = 0.0
        ep_success = 0.0
        while not done:
            a = masked_epsilon_greedy(Q, env, s, epsilon, rng)
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done
            nofly_fail = result.info.get("failure") == "no_fly_violation"
            shaped_r = r - (risk_penalty_weight if nofly_fail else 0.0)
            ep_return += shaped_r
            if result.info.get("success"):
                ep_success = 1.0

            next_valid = get_valid_actions(env, ns)
            if done:
                target = shaped_r
            else:
                best_next = max(Q[ns][na] for na in next_valid)
                target = shaped_r + gamma * best_next

            Q[s][a] += alpha_t * (target - Q[s][a])
            s = ns
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        if return_trace is not None:
            return_trace.append(ep_return)
        if success_trace is not None:
            success_trace.append(ep_success)
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
    parser.add_argument("--episodes", type=int, default=800000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--alpha-end", type=float, default=0.03)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.08)
    parser.add_argument("--optimistic-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--risk-q-alpha",
        "--q-alpha",
        dest="risk_q_alpha",
        type=float,
        default=0.05,
        help="Learning rate for risk-aware Q-learning.",
    )
    parser.add_argument(
        "--exp-sarsa-alpha",
        "--sarsa-alpha",
        dest="exp_sarsa_alpha",
        type=float,
        default=0.04,
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
    parser.add_argument(
        "--learning-curve-window",
        type=int,
        default=200,
        help="Rolling window for episode return/success learning curves.",
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
    vi_metrics["algorithm"] = "Value Iteration (DP baseline)"

    print("Training Expected SARSA...")
    exp_return_trace: List[float] = []
    exp_success_trace: List[float] = []
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
        return_trace=exp_return_trace,
        success_trace=exp_success_trace,
    )
    exp_sarsa_policy = greedy_policy_from_q(QES, env, seed=202)
    exp_sarsa_metrics = evaluate_policy(
        factory, exp_sarsa_policy, episodes=args.eval_episodes
    )
    exp_sarsa_metrics["algorithm"] = "Expected SARSA (MF baseline)"

    print("Training Risk-aware Q-learning...")
    risk_return_trace: List[float] = []
    risk_success_trace: List[float] = []
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
        return_trace=risk_return_trace,
        success_trace=risk_success_trace,
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

    save_training_trace_csv(
        out_dir / "episode_training_trace.csv",
        exp_return_trace,
        exp_success_trace,
        risk_return_trace,
        risk_success_trace,
    )
    plot_learning_curves(
        out_dir,
        exp_return_trace,
        exp_success_trace,
        risk_return_trace,
        risk_success_trace,
        window=args.learning_curve_window,
        vi_mean_return=float(vi_metrics["mean_return"]),
        vi_success_rate=float(vi_metrics["success_rate"]),
    )

    save_run_artifacts(
        out_dir=out_dir,
        args=args,
        rows=rows,
        vi_policy=vi_policy,
        vi_values=V,
        expected_q=QES,
        risk_q=QR,
        expected_returns=exp_return_trace,
        expected_success=exp_success_trace,
        risk_returns=risk_return_trace,
        risk_success=risk_success_trace,
    )

    sample = rollout(UAVSolarEnv(scenario_path, seed=100), risk_policy)
    print_rollout(sample, "Risk-aware Q-learning policy")
    with (out_dir / "sample_rollout.json").open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()

