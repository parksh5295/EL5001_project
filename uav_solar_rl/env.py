from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

State = Tuple[int, int, int, int, str, str, str, int]
Pos3D = Tuple[int, int, int]

ACTIONS = [
    "move_N",
    "move_S",
    "move_E",
    "move_W",
    "ascend",
    "descend",
    "hover",
    "charge",
    "inspect",
]

QUEUE_STATES = ["Short", "Long"]


@dataclass
class StepResult:
    next_state: State
    reward: float
    done: bool
    info: Dict[str, object]


class UAVSolarEnv:
    """
    Small tabular MDP for real-data-grounded UAV solar panel inspection.

    State:
        (x, y, z, battery, wind, queue_P1, queue_P2, target_mask)
    """

    def __init__(self, scenario_path: str | Path = "data/scenario.json", seed: int = 0):

        self.rng = random.Random(seed)

        with open(scenario_path, "r", encoding="utf-8") as f:
            self.scenario = json.load(f)

        self.grid_size = tuple(self.scenario["grid_size"])

        self.base: Pos3D = tuple(self.scenario["base"])

        self.targets: List[Pos3D] = [tuple(t) for t in self.scenario["targets"]]

        self.target_to_bit = {t: i for i, t in enumerate(self.targets)}

        self.all_targets_mask = (1 << len(self.targets)) - 1

        self.charging_pads = [tuple(p) for p in self.scenario["charging_pads"]]

        self.no_fly_cells = {tuple(c) for c in self.scenario["no_fly_cells"]}

        self.restricted_cells = {tuple(c) for c in self.scenario["restricted_cells"]}

        self.wind_states = list(self.scenario["wind_states"])

        self.wind_transition = self.scenario["wind_transition"]

        self.queue_transition = self.scenario["queue_transition"]

        self.max_battery = int(self.scenario["max_battery"])

        self.max_steps = int(self.scenario["max_steps"])

        self.R = self.scenario["reward"]

        self.actions = list(ACTIONS)

        self.steps = 0

        self.state: State = self.initial_state()

    def initial_state(self) -> State:

        x, y, z = self.base

        return (
            x,
            y,
            z,
            self.max_battery,
            self.scenario["initial_wind"],
            "Short",
            "Short",
            0,
        )

    def reset(self) -> State:

        self.steps = 0
        self.state = self.initial_state()

        return self.state

    def in_bounds(self, pos: Pos3D) -> bool:

        x, y, z = pos

        nx, ny, nz = self.grid_size

        return 0 <= x < nx and 0 <= y < ny and 0 <= z < nz

    def move_position(self, pos: Pos3D, action: str) -> Pos3D:

        x, y, z = pos

        if action == "move_N":
            return (x, y + 1, z)

        if action == "move_S":
            return (x, y - 1, z)

        if action == "move_E":
            return (x + 1, y, z)

        if action == "move_W":
            return (x - 1, y, z)

        if action == "ascend":
            return (x, y, z + 1)

        if action == "descend":
            return (x, y, z - 1)

        return pos

    def wind_drift_position(self, pos: Pos3D, wind: str) -> Pos3D:

        x, y, z = pos

        if wind == "EastWind":
            return (x + 1, y, z)

        if wind == "NorthWind":
            return (x, y + 1, z)

        return pos

    def sample_from_dist(self, dist: Dict[str, float]) -> str:

        r = self.rng.random()

        acc = 0.0
        last = None

        for k, p in dist.items():

            acc += float(p)
            last = k

            if r <= acc:
                return k

        return last

    def possible_next_exogenous(self, wind: str, q1: str, q2: str):

        for nw, pw in self.wind_transition[wind].items():

            for nq1, pq1 in self.queue_transition[q1].items():

                for nq2, pq2 in self.queue_transition[q2].items():

                    yield (
                        nw,
                        nq1,
                        nq2,
                        float(pw) * float(pq1) * float(pq2),
                    )

    def transition_distribution(
        self,
        state: State,
        action: str,
    ):

        outcomes = []

        for nw, nq1, nq2, p_exo in self.possible_next_exogenous(
            state[4],
            state[5],
            state[6],
        ):

            ns, reward, done, info = self._deterministic_part(
                state,
                action,
                nw,
                nq1,
                nq2,
            )

            outcomes.append(
                (
                    p_exo,
                    ns,
                    reward,
                    done,
                    info,
                )
            )

        return outcomes

    def step(self, action: str) -> StepResult:

        wind, q1, q2 = (
            self.state[4],
            self.state[5],
            self.state[6],
        )

        nw = self.sample_from_dist(self.wind_transition[wind])

        nq1 = self.sample_from_dist(self.queue_transition[q1])

        nq2 = self.sample_from_dist(self.queue_transition[q2])

        ns, reward, done, info = self._deterministic_part(
            self.state,
            action,
            nw,
            nq1,
            nq2,
        )

        self.steps += 1

        if self.steps >= self.max_steps and not done:

            done = True
            info["timeout"] = True

        self.state = ns

        return StepResult(ns, reward, done, info)

    def _deterministic_part(
        self,
        state: State,
        action: str,
        next_wind: str,
        next_q1: str,
        next_q2: str,
    ):

        x, y, z, battery, wind, q1, q2, mask = state

        pos = (x, y, z)

        reward = float(self.R["step_cost"])

        done = False

        info: Dict[str, object] = {}

        next_pos = pos
        next_battery = battery
        next_mask = mask

        # --------------------------------------------------
        # movement actions
        # --------------------------------------------------

        if action in [
            "move_N",
            "move_S",
            "move_E",
            "move_W",
            "ascend",
            "descend",
        ]:

            next_pos = self.move_position(pos, action)

            if not self.in_bounds(next_pos):

                next_pos = pos

                reward += float(self.R["invalid_action_cost"])

            else:

                next_battery -= 1

                reward += float(self.R["move_battery_cost"])

                if action == "ascend":

                    next_battery -= 1

                    reward += float(self.R["ascend_extra_battery_cost"])

                # deterministic wind drift
                if wind != "Calm":

                    drifted = self.wind_drift_position(
                        next_pos,
                        wind,
                    )

                    if self.in_bounds(drifted):

                        next_pos = drifted

                        if drifted != self.move_position(
                            pos,
                            action,
                        ):
                            info["wind_drift"] = True

        # --------------------------------------------------
        # hover
        # --------------------------------------------------

        elif action == "hover":

            next_battery -= 1

            reward += float(self.R["move_battery_cost"])

        # --------------------------------------------------
        # charge
        # --------------------------------------------------

        elif action == "charge":

            if pos in self.charging_pads:

                pad_index = self.charging_pads.index(pos)

                queue = q1 if pad_index == 0 else q2

                reward += float(
                    self.R["charge_short_queue_cost"]
                    if queue == "Short"
                    else self.R["charge_long_queue_cost"]
                )

                next_battery = self.max_battery

                info["charged"] = True

            else:

                reward += float(self.R["invalid_action_cost"])

        # --------------------------------------------------
        # inspect
        # --------------------------------------------------

        elif action == "inspect":

            if pos in self.target_to_bit:

                bit = self.target_to_bit[pos]

                if not (mask & (1 << bit)):

                    next_mask = mask | (1 << bit)

                    reward += float(self.R["inspect_new_target"])

                    info["new_target"] = bit

                else:

                    reward += float(self.R["invalid_action_cost"])

            else:

                reward += float(self.R["invalid_action_cost"])

        else:

            raise ValueError(f"Unknown action: {action}")

        # --------------------------------------------------
        # terminal conditions
        # --------------------------------------------------

        if next_battery < 0 and not done:

            reward += float(self.R["battery_depletion"])

            done = True

            info["failure"] = "battery_depletion"

        if next_pos in self.no_fly_cells and not done:

            reward += float(self.R["no_fly_violation"])

            done = True

            info["failure"] = "no_fly_violation"

        if next_pos in self.restricted_cells and not done:

            reward += float(self.R["restricted_area_cost"])

            info["restricted_area"] = True

        if next_pos == self.base and next_mask == self.all_targets_mask and not done:

            reward += float(self.R["finish_all_and_return_base"])

            done = True

            info["success"] = True

        nx, ny, nz = next_pos

        next_state: State = (
            nx,
            ny,
            nz,
            max(
                0,
                min(
                    self.max_battery,
                    next_battery,
                ),
            ),
            next_wind,
            next_q1,
            next_q2,
            next_mask,
        )

        return (
            next_state,
            reward,
            done,
            info,
        )

    def enumerate_states(self) -> List[State]:

        nx, ny, nz = self.grid_size

        states: List[State] = []

        for x in range(nx):

            for y in range(ny):

                for z in range(nz):

                    for b in range(
                        0,
                        self.max_battery + 1,
                    ):

                        for w in self.wind_states:

                            for q1 in QUEUE_STATES:

                                for q2 in QUEUE_STATES:

                                    for mask in range(self.all_targets_mask + 1):

                                        states.append(
                                            (
                                                x,
                                                y,
                                                z,
                                                b,
                                                w,
                                                q1,
                                                q2,
                                                mask,
                                            )
                                        )

        return states
