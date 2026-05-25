from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import random
import time
from typing import Any, Dict, List
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


def scenario_signature(scenario_path: str) -> str:
    p = Path(scenario_path)
    if p.exists():
        return f"{p.resolve()}|{p.stat().st_mtime_ns}"
    return scenario_path


def build_run_signature(args: argparse.Namespace) -> str:
    fields = [
        "scenario",
        "episodes",
        "eval_episodes",
        "gamma",
        "q_alpha",
        "sarsa_alpha",
        "exp_sarsa_alpha",
        "double_q_alpha",
        "sarsa_lambda_alpha",
        "masked_q_alpha",
        "risk_q_alpha",
        "alpha_end",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_q",
        "epsilon_decay_sarsa",
        "epsilon_decay_exp_sarsa",
        "epsilon_decay_double_q",
        "epsilon_decay_sarsa_lambda",
        "epsilon_decay_masked_q",
        "epsilon_decay_risk_q",
        "optimistic_init",
        "sarsa_lambda",
        "risk_nofly_threshold",
        "risk_penalty_weight",
    ]
    data = {k: getattr(args, k) for k in fields}
    data["scenario_sig"] = scenario_signature(args.scenario)
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def checkpoint_file(out_dir: Path, name: str) -> Path:
    cp_dir = out_dir / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    return cp_dir / f"{name}.pkl"


def _qtable_to_plain(model_q: dict) -> dict:
    return {state: dict(action_values) for state, action_values in model_q.items()}


def _plain_to_qtable(plain_q: dict, actions: list[str]):
    q = make_q(actions, initial_value=0.0)
    for state, action_values in plain_q.items():
        q[state] = {a: float(action_values.get(a, 0.0)) for a in actions}
    return q


def _serialize_model_for_checkpoint(model: Any) -> dict[str, Any]:
    if isinstance(model, tuple) and len(model) == 2:
        return {
            "kind": "double_q",
            "payload": [_qtable_to_plain(model[0]), _qtable_to_plain(model[1])],
        }
    if isinstance(model, dict):
        return {"kind": "q_table", "payload": _qtable_to_plain(model)}
    return {"kind": "raw", "payload": model}


def _deserialize_model_from_checkpoint(
    serialized: dict[str, Any], actions: list[str]
) -> Any:
    kind = serialized.get("kind", "raw")
    payload = serialized.get("payload")
    if kind == "q_table":
        return _plain_to_qtable(payload, actions)
    if kind == "double_q":
        q1 = _plain_to_qtable(payload[0], actions)
        q2 = _plain_to_qtable(payload[1], actions)
        return (q1, q2)
    return payload


def load_checkpoint(
    path: Path, run_sig: str, actions: list[str]
) -> tuple[Any, dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except (EOFError, pickle.UnpicklingError, AttributeError, ValueError, OSError):
        # Corrupted or partially written checkpoint; ignore and retrain.
        return None
    if data.get("run_signature") != run_sig:
        return None
    model = _deserialize_model_from_checkpoint(data["model"], actions)
    return model, data["metrics"]


def save_checkpoint(path: Path, run_sig: str, model: Any, metrics: dict[str, Any]) -> None:
    serialized_model = _serialize_model_for_checkpoint(model)
    with path.open("wb") as f:
        pickle.dump(
            {
                "run_signature": run_sig,
                "model": serialized_model,
                "metrics": metrics,
            },
            f,
        )


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


def _format_duration(seconds: float) -> str:
    sec = max(0, int(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def save_action_masked_progress_plot(
    out_dir: Path,
    completed_episodes: int,
    total_episodes: int,
    elapsed_seconds: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = max(1, total_episodes)
    done = min(max(0, completed_episodes), total)
    remaining = max(0, total - done)
    progress = done / total
    eta_seconds = 0.0 if done == 0 else elapsed_seconds * (remaining / max(1, done))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(["completed", "remaining"], [done, remaining], color=["#2ca02c", "#d3d3d3"])
    axes[0].set_ylabel("episodes")
    axes[0].set_title("Action-masked progress")
    axes[0].text(
        0.5,
        0.95,
        f"{progress * 100:.1f}%",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )

    axes[1].bar(
        ["elapsed (min)", "eta (min)"],
        [elapsed_seconds / 60.0, eta_seconds / 60.0],
        color=["#1f77b4", "#ff7f0e"],
    )
    axes[1].set_ylabel("minutes")
    axes[1].set_title("Elapsed vs ETA")
    axes[1].text(
        0.5,
        0.95,
        f"ETA {_format_duration(eta_seconds)}",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    fig.suptitle("Action-masked Q-learning training status")
    plt.tight_layout()
    plt.savefig(out_dir / "action_masked_progress.png", dpi=180)
    plt.close(fig)

    with (out_dir / "action_masked_progress.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_episodes": done,
                "total_episodes": total,
                "progress_ratio": progress,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
                "elapsed_hms": _format_duration(elapsed_seconds),
                "eta_hms": _format_duration(eta_seconds),
            },
            f,
            indent=2,
            ensure_ascii=False,
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


def action_nofly_violation_prob_cached(
    env: UAVSolarEnv,
    state: State,
    action: str,
    cache: dict[tuple[State, str], float],
) -> float:
    key = (state, action)
    if key in cache:
        return cache[key]
    nofly_prob = 0.0
    for p, _, _, done, info in env.transition_distribution(state, action):
        if done and info.get("failure") == "no_fly_violation":
            nofly_prob += p
    cache[key] = nofly_prob
    return nofly_prob


def masked_actions(
    env: UAVSolarEnv,
    state: State,
    max_nofly_prob: float,
    nofly_cache: dict[tuple[State, str], float] | None = None,
) -> List[str]:
    if nofly_cache is None:
        safe = [
            a
            for a in env.actions
            if action_safe_under_nofly(env, state, a, max_nofly_prob=max_nofly_prob)
        ]
    else:
        safe = [
            a
            for a in env.actions
            if action_nofly_violation_prob_cached(env, state, a, nofly_cache)
            <= max_nofly_prob
        ]
    return safe if safe else list(env.actions)


def masked_epsilon_greedy(
    env: UAVSolarEnv,
    Q,
    state: State,
    epsilon: float,
    rng: random.Random,
    max_nofly_prob: float,
    nofly_cache: dict[tuple[State, str], float] | None = None,
) -> str:
    candidates = masked_actions(
        env,
        state,
        max_nofly_prob=max_nofly_prob,
        nofly_cache=nofly_cache,
    )
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
    progress_dir: Path | None = None,
    progress_update_every: int = 200,
):
    rng = random.Random(seed)
    env = env_factory_fn()
    Q = make_q(env.actions, initial_value=optimistic_init)
    epsilon = epsilon_start
    nofly_cache: dict[tuple[State, str], float] = {}
    started_at = time.time()
    update_every = max(1, progress_update_every)

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
                nofly_cache=nofly_cache,
            )
            result = env.step(a)
            ns, r, done = result.next_state, result.reward, result.done
            best_next = max(Q[ns].values())
            target = r + (0.0 if done else gamma * best_next)
            Q[s][a] += alpha_t * (target - Q[s][a])
            s = ns
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if progress_dir is not None:
            if (ep + 1) % update_every == 0 or ep == 0 or (ep + 1) == episodes:
                elapsed = time.time() - started_at
                save_action_masked_progress_plot(
                    progress_dir,
                    completed_episodes=ep + 1,
                    total_episodes=episodes,
                    elapsed_seconds=elapsed,
                )
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
    nofly_cache: dict[tuple[State, str], float] = {}

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
            risk_penalty = risk_penalty_weight * action_nofly_violation_prob_cached(
                env, s, a, nofly_cache
            )
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
    nofly_cache: dict[tuple[State, str], float] = {}

    def _policy(state: State) -> str:
        candidates = masked_actions(
            env,
            state,
            max_nofly_prob=max_nofly_prob,
            nofly_cache=nofly_cache,
        )
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
    parser.add_argument(
        "--masked-progress-update-every",
        type=int,
        default=200,
        help="Update interval (episodes) for Action-masked progress bar chart.",
    )
    args = parser.parse_args()

    scenario_path = args.scenario
    out_dir = model_free_output_dir(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    factory = env_factory(scenario_path)
    env = UAVSolarEnv(scenario_path)
    run_sig = build_run_signature(args)

    print("Scenario loaded")
    print(f"Number of states: {len(env.enumerate_states())}")
    print(f"Actions: {env.actions}\n")

    q_cp = checkpoint_file(out_dir, "q_learning")
    q_loaded = load_checkpoint(q_cp, run_sig, env.actions)
    if q_loaded is None:
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
        save_checkpoint(q_cp, run_sig, Q, q_metrics)
    else:
        print("Loading cached Q-learning...")
        Q, q_metrics = q_loaded
        q_policy = greedy_policy_from_q(Q, env, seed=101)
        q_metrics["algorithm"] = "Q-learning"

    sarsa_cp = checkpoint_file(out_dir, "sarsa")
    sarsa_loaded = load_checkpoint(sarsa_cp, run_sig, env.actions)
    if sarsa_loaded is None:
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
        save_checkpoint(sarsa_cp, run_sig, QS, sarsa_metrics)
    else:
        print("Loading cached SARSA...")
        QS, sarsa_metrics = sarsa_loaded
        sarsa_metrics["algorithm"] = "SARSA"

    exp_sarsa_cp = checkpoint_file(out_dir, "expected_sarsa")
    exp_loaded = load_checkpoint(exp_sarsa_cp, run_sig, env.actions)
    if exp_loaded is None:
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
        save_checkpoint(exp_sarsa_cp, run_sig, QES, exp_sarsa_metrics)
    else:
        print("Loading cached Expected SARSA...")
        QES, exp_sarsa_metrics = exp_loaded
        exp_sarsa_metrics["algorithm"] = "Expected SARSA"

    double_cp = checkpoint_file(out_dir, "double_q_learning")
    double_loaded = load_checkpoint(double_cp, run_sig, env.actions)
    if double_loaded is None:
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
        save_checkpoint(double_cp, run_sig, (Q1, Q2), double_q_metrics)
    else:
        print("Loading cached Double Q-learning...")
        (Q1, Q2), double_q_metrics = double_loaded
        double_q_metrics["algorithm"] = "Double Q-learning"

    sarsa_l_cp = checkpoint_file(out_dir, "sarsa_lambda")
    sarsa_l_loaded = load_checkpoint(sarsa_l_cp, run_sig, env.actions)
    if sarsa_l_loaded is None:
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
        save_checkpoint(sarsa_l_cp, run_sig, QL, sarsa_lambda_metrics)
    else:
        print("Loading cached SARSA(lambda)...")
        QL, sarsa_lambda_metrics = sarsa_l_loaded
        sarsa_lambda_metrics["algorithm"] = "SARSA(lambda)"

    masked_cp = checkpoint_file(out_dir, "action_masked_q_learning")
    masked_loaded = load_checkpoint(masked_cp, run_sig, env.actions)
    if masked_loaded is None:
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
            progress_dir=out_dir,
            progress_update_every=args.masked_progress_update_every,
        )
        masked_q_policy = masked_q_policy_fn(
            QM, env, seed=606, max_nofly_prob=args.risk_nofly_threshold
        )
        masked_q_metrics = evaluate_policy(
            factory, masked_q_policy, episodes=args.eval_episodes
        )
        masked_q_metrics["algorithm"] = "Action-masked Q-learning"
        save_checkpoint(masked_cp, run_sig, QM, masked_q_metrics)
    else:
        print("Loading cached Action-masked Q-learning...")
        QM, masked_q_metrics = masked_loaded
        masked_q_metrics["algorithm"] = "Action-masked Q-learning"
        save_action_masked_progress_plot(
            out_dir,
            completed_episodes=args.episodes,
            total_episodes=args.episodes,
            elapsed_seconds=0.0,
        )

    risk_cp = checkpoint_file(out_dir, "risk_aware_q_learning")
    risk_loaded = load_checkpoint(risk_cp, run_sig, env.actions)
    if risk_loaded is None:
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
        save_checkpoint(risk_cp, run_sig, QR, risk_q_metrics)
    else:
        print("Loading cached Risk-aware Q-learning...")
        QR, risk_q_metrics = risk_loaded
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
