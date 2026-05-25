from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Dict, List
from collections import defaultdict

import matplotlib.pyplot as plt

from uav_solar_rl.algorithms import (
    evaluate_policy,
    greedy_action,
    greedy_policy_from_q,
    make_q,
    q_learning,
    rollout,
    sarsa,
)
from uav_solar_rl.env import State, UAVSolarEnv


def env_factory(scenario_path: str, seed: int = 0):
    return lambda: UAVSolarEnv(scenario_path=scenario_path, seed=seed)


def normalize_output_dir(output_dir_arg: str) -> Path:
    out_dir = Path(output_dir_arg)
    if out_dir.suffix.lower() == ".json":
        out_dir = out_dir.with_suffix("")
    return out_dir


def model_free_output_dir(output_dir_arg: str) -> Path:
    base_dir = normalize_output_dir(output_dir_arg)
    # Match main.py style while separating model-free runs by prefix.
    if base_dir == Path("results"):
        return base_dir / "model_free"
    name = base_dir.name
    if not name.startswith("model_free_"):
        base_dir = base_dir.with_name(f"model_free_{name}")
    return base_dir


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


def plot_metrics(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [r["algorithm"] for r in rows]
    returns = [r["mean_return"] for r in rows]
    success = [r["success_rate"] for r in rows]

    plt.figure()
    plt.bar(names, returns)
    plt.ylabel("Mean Return")
    plt.title("Model-Free Comparison: Mean Return")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_return.png", dpi=180)
    plt.close()

    plt.figure()
    plt.bar(names, success)
    plt.ylabel("Success Rate")
    plt.title("Model-Free Comparison: Success Rate")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "success_rate.png", dpi=180)
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


def action_safe_under_nofly(
    env: UAVSolarEnv, state: State, action: str, max_nofly_prob: float
) -> bool:
    return action_nofly_violation_prob(env, state, action) <= max_nofly_prob


def action_nofly_violation_prob(env: UAVSolarEnv, state: State, action: str) -> float:
    nofly_prob = 0.0
    for p, _, _, done, info in env.transition_distribution(state, action):
        if done and info.get("failure") == "no_fly_violation":
            nofly_prob += p
    return nofly_prob


def masked_actions(
    env: UAVSolarEnv, state: State, max_nofly_prob: float
) -> List[str]:
    safe = [
        a
        for a in env.actions
        if action_safe_under_nofly(env, state, a, max_nofly_prob=max_nofly_prob)
    ]
    return safe if safe else list(env.actions)


def masked_epsilon_greedy(
    env: UAVSolarEnv,
    Q,
    state: State,
    epsilon: float,
    rng: random.Random,
    max_nofly_prob: float,
) -> str:
    candidates = masked_actions(env, state, max_nofly_prob=max_nofly_prob)
    if rng.random() < epsilon:
        return rng.choice(candidates)
    max_v = max(Q[state][a] for a in candidates)
    best = [a for a in candidates if Q[state][a] == max_v]
    return rng.choice(best)


def action_masked_q_learning(
    env_factory_fn,
    episodes: int,
    alpha: float,
    alpha_end: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    optimistic_init: float,
    max_nofly_prob: float,
    seed: int,
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
            a = masked_epsilon_greedy(
                env,
                Q,
                s,
                epsilon=epsilon,
                rng=rng,
                max_nofly_prob=max_nofly_prob,
            )
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done
            best_next = max(Q[ns].values())
            target = r + (0.0 if done else gamma * best_next)
            Q[s][a] += alpha_t * (target - Q[s][a])
            s = ns
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    return Q


def risk_aware_q_learning(
    env_factory_fn,
    episodes: int,
    alpha: float,
    alpha_end: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    optimistic_init: float,
    risk_penalty_weight: float,
    seed: int,
):
    """Risk-aware Q-learning via expected no-fly risk penalty (no action masking)."""
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
            if rng.random() < epsilon:
                a = rng.choice(env.actions)
            else:
                a = greedy_action(Q[s], rng)
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done
            risk_penalty = risk_penalty_weight * action_nofly_violation_prob(env, s, a)
            shaped_r = r - risk_penalty
            best_next = max(Q[ns].values())
            target = shaped_r + (0.0 if done else gamma * best_next)
            Q[s][a] += alpha_t * (target - Q[s][a])
            s = ns
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    return Q


def double_q_learning(
    env_factory_fn,
    episodes: int,
    alpha: float,
    alpha_end: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    optimistic_init: float,
    seed: int,
):
    rng = random.Random(seed)
    env = env_factory_fn()
    Q1 = make_q(env.actions, initial_value=optimistic_init)
    Q2 = make_q(env.actions, initial_value=optimistic_init)
    epsilon = epsilon_start

    def qsum(state: State) -> Dict[str, float]:
        return {a: Q1[state][a] + Q2[state][a] for a in env.actions}

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        alpha_t = max(alpha_end, alpha * (1.0 - frac))
        s = env.reset()
        done = False
        while not done:
            if rng.random() < epsilon:
                a = rng.choice(env.actions)
            else:
                a = greedy_action(qsum(s), rng)
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done

            if rng.random() < 0.5:
                best_a = max(env.actions, key=lambda x: Q1[ns][x])
                target = r + (0.0 if done else gamma * Q2[ns][best_a])
                Q1[s][a] += alpha_t * (target - Q1[s][a])
            else:
                best_a = max(env.actions, key=lambda x: Q2[ns][x])
                target = r + (0.0 if done else gamma * Q1[ns][best_a])
                Q2[s][a] += alpha_t * (target - Q2[s][a])
            s = ns
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    return Q1, Q2


def sarsa_lambda(
    env_factory_fn,
    episodes: int,
    alpha: float,
    alpha_end: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    optimistic_init: float,
    lam: float,
    seed: int,
):
    rng = random.Random(seed)
    env = env_factory_fn()
    Q = make_q(env.actions, initial_value=optimistic_init)
    epsilon = epsilon_start

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        alpha_t = max(alpha_end, alpha * (1.0 - frac))
        traces = defaultdict(lambda: {a: 0.0 for a in env.actions})
        active_states = set()

        s = env.reset()
        if rng.random() < epsilon:
            a = rng.choice(env.actions)
        else:
            a = greedy_action(Q[s], rng)
        done = False

        while not done:
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done
            if not done:
                if rng.random() < epsilon:
                    na = rng.choice(env.actions)
                else:
                    na = greedy_action(Q[ns], rng)
                td_target = r + gamma * Q[ns][na]
            else:
                na = env.actions[0]
                td_target = r
            td_error = td_target - Q[s][a]

            traces[s][a] += 1.0
            active_states.add(s)
            if not done:
                active_states.add(ns)

            to_remove = []
            for st in list(active_states):
                max_abs = 0.0
                for act in env.actions:
                    e = traces[st][act]
                    if e != 0.0:
                        Q[st][act] += alpha_t * td_error * e
                        new_e = gamma * lam * e
                        traces[st][act] = new_e
                        if abs(new_e) > max_abs:
                            max_abs = abs(new_e)
                if max_abs < 1e-8:
                    to_remove.append(st)
            for st in to_remove:
                active_states.discard(st)

            s, a = ns, na

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    return Q


def expected_sarsa(
    env_factory_fn,
    episodes: int,
    alpha: float,
    alpha_end: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    optimistic_init: float,
    seed: int,
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
            if rng.random() < epsilon:
                a = rng.choice(env.actions)
            else:
                a = greedy_action(Q[s], rng)
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done
            probs = epsilon_greedy_probs(Q[ns], env.actions, epsilon)
            expected_next = sum(probs[na] * Q[ns][na] for na in env.actions)
            target = r + (0.0 if done else gamma * expected_next)
            Q[s][a] += alpha_t * (target - Q[s][a])
            s = ns
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    return Q


def q_policy_fn(Q, actions: List[str], seed: int):
    rng = random.Random(seed)

    def _policy(state: State) -> str:
        return greedy_action({a: Q[state][a] for a in actions}, rng)

    return _policy


def double_q_policy_fn(Q1, Q2, actions: List[str], seed: int):
    rng = random.Random(seed)

    def _policy(state: State) -> str:
        qsum = {a: Q1[state][a] + Q2[state][a] for a in actions}
        return greedy_action(qsum, rng)

    return _policy


def masked_q_policy_fn(Q, env: UAVSolarEnv, seed: int, max_nofly_prob: float):
    rng = random.Random(seed)

    def _policy(state: State) -> str:
        candidates = masked_actions(env, state, max_nofly_prob=max_nofly_prob)
        max_v = max(Q[state][a] for a in candidates)
        best = [a for a in candidates if Q[state][a] == max_v]
        return rng.choice(best)

    return _policy


def main():
    parser = argparse.ArgumentParser(
        description="Model-free comparison: Q-learning/SARSA variants only"
    )
    parser.add_argument("--scenario", default="data/scenario.json")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--q-alpha", type=float, default=0.10)
    parser.add_argument("--sarsa-alpha", type=float, default=0.08)
    parser.add_argument("--exp-sarsa-alpha", type=float, default=0.08)
    parser.add_argument("--double-q-alpha", type=float, default=0.10)
    parser.add_argument("--sarsa-lambda-alpha", type=float, default=0.08)
    parser.add_argument("--masked-q-alpha", type=float, default=0.10)
    parser.add_argument("--risk-q-alpha", type=float, default=0.10)
    parser.add_argument("--alpha-end", type=float, default=0.03)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-q", type=float, default=0.9995)
    parser.add_argument("--epsilon-decay-sarsa", type=float, default=0.9997)
    parser.add_argument("--epsilon-decay-exp-sarsa", type=float, default=0.9997)
    parser.add_argument("--epsilon-decay-double-q", type=float, default=0.9995)
    parser.add_argument("--epsilon-decay-sarsa-lambda", type=float, default=0.9997)
    parser.add_argument("--epsilon-decay-masked-q", type=float, default=0.9995)
    parser.add_argument("--epsilon-decay-risk-q", type=float, default=0.9995)
    parser.add_argument("--optimistic-init", type=float, default=0.0)
    parser.add_argument("--sarsa-lambda", type=float, default=0.8)
    parser.add_argument(
        "--risk-nofly-threshold",
        type=float,
        default=0.0,
        help="Max allowed one-step no-fly violation probability in action-masked Q",
    )
    parser.add_argument(
        "--risk-penalty-weight",
        type=float,
        default=40.0,
        help="Penalty weight for expected no-fly risk in risk-aware Q-learning",
    )
    args = parser.parse_args()

    scenario_path = args.scenario
    out_dir = model_free_output_dir(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    factory = env_factory(scenario_path)
    env = UAVSolarEnv(scenario_path)

    print("Scenario loaded")
    print(f"Number of states: {len(env.enumerate_states())}")
    print(f"Actions: {env.actions}\n")

    print("Training Q-learning...")
    Q = q_learning(
        factory,
        episodes=args.episodes,
        alpha=args.q_alpha,
        alpha_end=args.alpha_end,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay_q,
        optimistic_init=args.optimistic_init,
        seed=1,
    )
    q_policy = greedy_policy_from_q(Q, env, seed=101)
    q_metrics = evaluate_policy(factory, q_policy, episodes=args.eval_episodes)
    q_metrics["algorithm"] = "Q-learning"

    print("Training SARSA...")
    QS = sarsa(
        factory,
        episodes=args.episodes,
        alpha=args.sarsa_alpha,
        alpha_end=args.alpha_end,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay_sarsa,
        optimistic_init=args.optimistic_init,
        seed=2,
    )
    sarsa_policy = greedy_policy_from_q(QS, env, seed=202)
    sarsa_metrics = evaluate_policy(factory, sarsa_policy, episodes=args.eval_episodes)
    sarsa_metrics["algorithm"] = "SARSA"

    print("Training Expected SARSA...")
    QES = expected_sarsa(
        factory,
        episodes=args.episodes,
        alpha=args.exp_sarsa_alpha,
        alpha_end=args.alpha_end,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay_exp_sarsa,
        optimistic_init=args.optimistic_init,
        seed=3,
    )
    expected_sarsa_policy = q_policy_fn(QES, env.actions, seed=303)
    exp_sarsa_metrics = evaluate_policy(
        factory, expected_sarsa_policy, episodes=args.eval_episodes
    )
    exp_sarsa_metrics["algorithm"] = "Expected SARSA"

    print("Training Double Q-learning...")
    Q1, Q2 = double_q_learning(
        factory,
        episodes=args.episodes,
        alpha=args.double_q_alpha,
        alpha_end=args.alpha_end,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay_double_q,
        optimistic_init=args.optimistic_init,
        seed=4,
    )
    double_q_policy = double_q_policy_fn(Q1, Q2, env.actions, seed=404)
    double_q_metrics = evaluate_policy(
        factory, double_q_policy, episodes=args.eval_episodes
    )
    double_q_metrics["algorithm"] = "Double Q-learning"

    print("Training SARSA(lambda)...")
    QL = sarsa_lambda(
        factory,
        episodes=args.episodes,
        alpha=args.sarsa_lambda_alpha,
        alpha_end=args.alpha_end,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay_sarsa_lambda,
        optimistic_init=args.optimistic_init,
        lam=args.sarsa_lambda,
        seed=5,
    )
    sarsa_lambda_policy = q_policy_fn(QL, env.actions, seed=505)
    sarsa_lambda_metrics = evaluate_policy(
        factory, sarsa_lambda_policy, episodes=args.eval_episodes
    )
    sarsa_lambda_metrics["algorithm"] = "SARSA(lambda)"

    print("Training Action-masked Q-learning...")
    QM = action_masked_q_learning(
        factory,
        episodes=args.episodes,
        alpha=args.masked_q_alpha,
        alpha_end=args.alpha_end,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay_masked_q,
        optimistic_init=args.optimistic_init,
        max_nofly_prob=args.risk_nofly_threshold,
        seed=6,
    )
    masked_q_policy = masked_q_policy_fn(
        QM, env, seed=606, max_nofly_prob=args.risk_nofly_threshold
    )
    masked_q_metrics = evaluate_policy(
        factory, masked_q_policy, episodes=args.eval_episodes
    )
    masked_q_metrics["algorithm"] = "Action-masked Q-learning"

    print("Training Risk-aware Q-learning...")
    QR = risk_aware_q_learning(
        factory,
        episodes=args.episodes,
        alpha=args.risk_q_alpha,
        alpha_end=args.alpha_end,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay_risk_q,
        optimistic_init=args.optimistic_init,
        risk_penalty_weight=args.risk_penalty_weight,
        seed=7,
    )
    risk_q_policy = q_policy_fn(QR, env.actions, seed=707)
    risk_q_metrics = evaluate_policy(factory, risk_q_policy, episodes=args.eval_episodes)
    risk_q_metrics["algorithm"] = "Risk-aware Q-learning"

    rows = [
        q_metrics,
        sarsa_metrics,
        exp_sarsa_metrics,
        double_q_metrics,
        sarsa_lambda_metrics,
        masked_q_metrics,
        risk_q_metrics,
    ]
    for r in rows:
        print(r)

    save_metrics(rows, out_dir / "metrics.csv")
    plot_metrics(rows, out_dir)

    sample = rollout(UAVSolarEnv(scenario_path, seed=100), q_policy)
    print_rollout(sample, "Q-learning policy")
    with (out_dir / "sample_rollout.json").open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()
