from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np

def get_valid_actions(env, state):
    """Return a safe candidate action set without mutating env state."""
    x, y, z, battery, wind, _q1, _q2, mask = state
    pos = (x, y, z)
    valid = []

    for action in env.actions:
        ok = True

        if action in ["move_N", "move_S", "move_E", "move_W", "ascend", "descend"]:
            # movement consumes battery (ascend consumes one extra)
            required_battery = 2 if action == "ascend" else 1
            if battery < required_battery:
                continue

            moved = env.move_position(pos, action)
            if not env.in_bounds(moved):
                ok = False
            else:
                # deterministic drift uses the current wind in state
                next_pos = moved
                if wind != "Calm":
                    drifted = env.wind_drift_position(moved, wind)
                    if env.in_bounds(drifted):
                        next_pos = drifted
                # mask actions that immediately violate no-fly
                if next_pos in env.no_fly_cells:
                    ok = False

        elif action == "charge":
            if pos not in env.charging_pads:
                ok = False
            elif battery >= env.max_battery:
                ok = False

        elif action == "inspect":
            if pos not in env.target_to_bit:
                ok = False
            else:
                bit = env.target_to_bit[pos]
                if mask & (1 << bit):
                    ok = False

        elif action == "hover":
            # disallow hover if it would immediately fail by battery depletion
            if battery <= 0:
                ok = False

        if ok:
            valid.append(action)

    if not valid:
        return ["hover"]
    return valid


def masked_epsilon_greedy(Q, env, state, epsilon, rng):
    valid_actions = get_valid_actions(env, state)

    if rng.random() < epsilon:
        return rng.choice(valid_actions)

    q_values = [Q[state][a] for a in valid_actions]
    max_q = max(q_values)

    best_actions = [
        a for a in valid_actions
        if Q[state][a] == max_q
    ]

    return rng.choice(best_actions)

from .env import ACTIONS, State, UAVSolarEnv

QTable = Dict[State, Dict[str, float]]
Policy = Dict[State, str]


def make_q(actions: List[str], initial_value: float = 0.0) -> QTable:
    return defaultdict(lambda: {a: float(initial_value) for a in actions})


def greedy_action(q_values: Dict[str, float], rng: random.Random | None = None) -> str:
    max_v = max(q_values.values())
    best = [a for a, v in q_values.items() if v == max_v]
    return (rng or random).choice(best)


def epsilon_greedy(
    Q: QTable, state: State, actions: List[str], epsilon: float, rng: random.Random
) -> str:
    if rng.random() < epsilon:
        return rng.choice(actions)
    return greedy_action(Q[state], rng)


def reachable_states(env: UAVSolarEnv, limit: int = 60000) -> List[State]:
    """Collect states reachable from the initial state. This keeps DP fast enough for a small project."""
    start = env.initial_state()
    seen = {start}
    frontier = [start]
    while frontier and len(seen) < limit:
        s = frontier.pop(0)
        for a in env.actions:
            for _, ns, _, done, _ in env.transition_distribution(s, a):
                if not done and ns not in seen:
                    seen.add(ns)
                    frontier.append(ns)
                    if len(seen) >= limit:
                        break
            if len(seen) >= limit:
                break
    return list(seen)


def value_iteration(
    env: UAVSolarEnv,
    gamma: float = 0.95,
    theta: float = 1e-5,
    max_iter: int = 500,
    state_mode: str = "reachable",
    reachable_limit: int = 200000,
) -> Tuple[Policy, Dict[State, float]]:
    if state_mode == "enumerate":
        states = env.enumerate_states()
    elif state_mode == "reachable":
        states = reachable_states(env, limit=reachable_limit)
    else:
        raise ValueError(f"Unknown state_mode: {state_mode}")
    V: Dict[State, float] = {s: 0.0 for s in states}
    transition_cache: Dict[
        Tuple[State, str], List[Tuple[float, State, float, bool]]
    ] = {}

    for s in states:
        for a in env.actions:
            transition_cache[(s, a)] = [
                (p, ns, r, done)
                for p, ns, r, done, _ in env.transition_distribution(s, a)
            ]

    def expected_action_value(state: State, action: str) -> float:
        """Bellman expectation over stochastic exogenous transitions."""
        total = 0.0
        for p, ns, r, done in transition_cache[(state, action)]:
            total += p * (r + (0.0 if done else gamma * V.get(ns, 0.0)))
        return total

    for iteration in range(max_iter):
        delta = 0.0
        for s in states:
            old_v = V[s]
            action_values = []
            for a in env.actions:
                action_values.append(expected_action_value(s, a))
            V[s] = max(action_values)
            delta = max(delta, abs(old_v - V[s]))
        if delta < theta:
            break

    policy: Policy = {}
    for s in states:
        best_a = None
        best_v = -float("inf")
        for a in env.actions:
            v = expected_action_value(s, a)
            if v > best_v:
                best_v = v
                best_a = a
        policy[s] = best_a or env.actions[0]
    return policy, V


def q_learning(
    env_factory: Callable[[], UAVSolarEnv],
    episodes: int = 30000,
    alpha: float = 0.10,
    alpha_end: float = 0.03,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9995,
    optimistic_init: float = 0.0,
    seed: int = 1,
) -> QTable:

    rng = random.Random(seed)

    env = env_factory()

    Q = make_q(env.actions, initial_value=optimistic_init)

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        alpha_t = max(alpha_end, alpha * (1.0 - frac))
        epsilon = max(epsilon_end, epsilon_start * (1.0 - frac))

        s = env.reset()

        done = False

        while not done:

            a = masked_epsilon_greedy(
    Q,
    env,
    s,
    epsilon,
    rng,
)

            result = env.step(a)

            ns = result.next_state
            r = result.reward
            done = result.done

            next_valid_actions = get_valid_actions(env, ns)

            if done:
                target = r
            else:
                best_next = max(Q[ns][na] for na in next_valid_actions)
                target = r + gamma * best_next

            Q[s][a] += alpha_t * (target - Q[s][a])

            s = ns

    return Q


def q_learning_mae_curve(
    env_factory: Callable[[], UAVSolarEnv],
    vi_ref_return: float,
    episodes: int = 10000,
    alpha: float = 0.10,
    alpha_end: float = 0.03,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    optimistic_init: float = 0.0,
    seed: int = 1,
    scenario_after: str | None = None,
    switch_episode: int | None = None,
) -> List[float]:
    """Episode-wise MAE curve using the same update semantics as q_learning."""
    rng = random.Random(seed)
    env = env_factory()
    Q = make_q(env.actions, initial_value=optimistic_init)
    mae_curve: List[float] = []

    for ep in range(episodes):
        if (
            scenario_after is not None
            and switch_episode is not None
            and ep == switch_episode
        ):
            env = UAVSolarEnv(scenario_path=scenario_after, seed=seed)

        frac = ep / max(1, episodes - 1)
        alpha_t = max(alpha_end, alpha * (1.0 - frac))
        epsilon = max(epsilon_end, epsilon_start * (1.0 - frac))

        s = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            a = masked_epsilon_greedy(Q, env, s, epsilon, rng)
            result = env.step(a)
            ns = result.next_state
            r = result.reward
            done = result.done
            ep_return += r

            next_valid_actions = get_valid_actions(env, ns)
            if done:
                target = r
            else:
                best_next = max(Q[ns][na] for na in next_valid_actions)
                target = r + gamma * best_next

            Q[s][a] += alpha_t * (target - Q[s][a])
            s = ns

        mae_curve.append(abs(ep_return - vi_ref_return))

    return mae_curve


def sarsa(
    env_factory: Callable[[], UAVSolarEnv],
    episodes: int = 30000,
    alpha: float = 0.10,
    alpha_end: float = 0.03,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9997,
    optimistic_init: float = 0.0,
    seed: int = 2,
) -> QTable:

    rng = random.Random(seed)

    env = env_factory()

    Q = make_q(env.actions, initial_value=optimistic_init)

    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        alpha_t = max(alpha_end, alpha * (1.0 - frac))
        epsilon = max(epsilon_end, epsilon_start * (1.0 - frac))

        s = env.reset()

        a = masked_epsilon_greedy(
    Q,
    env,
    s,
    epsilon,
    rng,
)

        done = False

        while not done:

            result = env.step(a)

            ns = result.next_state
            r = result.reward
            done = result.done

            na = masked_epsilon_greedy(
    Q,
    env,
    ns,
    epsilon,
    rng,
)

            target = r + (0.0 if done else gamma * Q[ns][na])

            Q[s][a] += alpha_t * (target - Q[s][a])

            s = ns
            a = na

    return Q


def q_to_policy(Q: QTable, actions: List[str]) -> Policy:
    policy: Policy = {}
    for s, q_values in Q.items():
        policy[s] = max(actions, key=lambda a: q_values[a])
    return policy


def greedy_policy_from_q(Q: QTable, env, seed: int = 0):
    rng = random.Random(seed)

    def policy_fn(state: State) -> str:
        valid_actions = get_valid_actions(env, state)

        max_q = max(Q[state][a] for a in valid_actions)

        best = [
            a for a in valid_actions
            if Q[state][a] == max_q
        ]

        return rng.choice(best)

    return policy_fn


def evaluate_policy(
    env_factory: Callable[[], UAVSolarEnv],
    policy_fn: Callable[[State], str],
    episodes: int = 300,
    seed: int = 7,
):
    rng = random.Random(seed)
    metrics = {
        "return": [],
        "steps": [],
        "success": 0,
        "battery_failure": 0,
        "no_fly_violation": 0,
        "restricted_visits": 0,
        "charged": 0,
    }

    for ep in range(episodes):
        env = env_factory()
        env.rng.seed(seed + ep)
        s = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            a = policy_fn(s)
            result = env.step(a)
            s = result.next_state
            total += result.reward
            steps += 1
            done = result.done
            info = result.info
            if info.get("success"):
                metrics["success"] += 1
            if info.get("failure") == "battery_depletion":
                metrics["battery_failure"] += 1
            if info.get("failure") == "no_fly_violation":
                metrics["no_fly_violation"] += 1
            if info.get("restricted_area"):
                metrics["restricted_visits"] += 1
            if info.get("charged"):
                metrics["charged"] += 1
        metrics["return"].append(total)
        metrics["steps"].append(steps)

    n = episodes
    return {
        "mean_return": float(np.mean(metrics["return"])),
        "std_return": float(np.std(metrics["return"])),
        "success_rate": metrics["success"] / n,
        "avg_steps": float(np.mean(metrics["steps"])),
        "battery_failures": metrics["battery_failure"],
        "no_fly_violations": metrics["no_fly_violation"],
        "restricted_visits": metrics["restricted_visits"],
        "charge_count": metrics["charged"],
    }


def rollout(env: UAVSolarEnv, policy_fn: Callable[[State], str], max_steps: int = 80):
    s = env.reset()
    rows = []
    for t in range(max_steps):
        a = policy_fn(s)
        result = env.step(a)
        rows.append(
            {
                "t": t,
                "state": s,
                "action": a,
                "reward": result.reward,
                "next_state": result.next_state,
                "info": result.info,
            }
        )
        s = result.next_state
        if result.done:
            break
    return rows
