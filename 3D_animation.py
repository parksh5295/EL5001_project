from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = PROJECT_ROOT / (
    "results/good_mf_real_gwangju_v1_20a781b_ep800000_qal005_salal004_epend008_enum"
)
DEFAULT_ROLLOUT = DEFAULT_RESULT_DIR / "sample_rollout.json"
DEFAULT_SCENARIO = PROJECT_ROOT / "data/scenario_real_gwangju_v1_20a781b.json"
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "uav_rollout_3d.gif"
DEFAULT_ENV_IMAGE = DEFAULT_RESULT_DIR / "uav_environment_3d.png"

ACTION_DELTAS = {
    "move_N": np.array([0, 1, 0]),
    "move_S": np.array([0, -1, 0]),
    "move_E": np.array([1, 0, 0]),
    "move_W": np.array([-1, 0, 0]),
    "ascend": np.array([0, 0, 1]),
    "descend": np.array([0, 0, -1]),
}

WIND_DRIFT_DELTAS = {
    "EastWind": np.array([1, 0, 0]),
    "NorthWind": np.array([0, 1, 0]),
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_scenario_from_result_dir(result_dir: Path) -> Path:
    data_dir = PROJECT_ROOT / "data"
    tokens = re.findall(r"[0-9a-f]{7,}", result_dir.name)
    for token in tokens:
        matched = sorted(data_dir.glob(f"scenario*{token}*.json"))
        if matched:
            return matched[0]
    return DEFAULT_SCENARIO


def rollout_to_trajectory(rows):
    if not rows:
        raise ValueError("Rollout rows are empty.")
    points = [np.array(rows[0]["state"][:3], dtype=float)]
    for row in rows:
        points.append(np.array(row["next_state"][:3], dtype=float))
    return np.vstack(points)


def cube_faces_from_cell(x: int, y: int, z: int, size: float = 1.0):
    x0, y0, z0 = x - 0.5 * size, y - 0.5 * size, z - 0.5 * size
    x1, y1, z1 = x + 0.5 * size, y + 0.5 * size, z + 0.5 * size
    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    return [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]


def draw_cell_cubes(ax, cells, color: str, alpha: float, edge_color: str | None = None):
    for x, y, z in cells:
        faces = cube_faces_from_cell(x, y, z, size=1.0)
        poly = Poly3DCollection(faces, alpha=alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor(edge_color or color)
        ax.add_collection3d(poly)


def setup_axes(ax, grid_size, title: str):
    nx, ny, nz = grid_size
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_zlim(-0.5, nz - 0.5)
    ax.set_xlabel("X (East)", labelpad=10)
    ax.set_ylabel("Y (North)", labelpad=10)
    ax.set_zlabel("Z (Altitude)", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    ax.view_init(elev=24, azim=-55)
    ax.grid(True, alpha=0.4)


def draw_static_elements(ax, scenario):
    targets = scenario["targets"]
    charging_pads = scenario["charging_pads"]
    no_fly_cells = scenario["no_fly_cells"]
    restricted_cells = scenario.get("restricted_cells", [])

    if restricted_cells:
        draw_cell_cubes(
            ax,
            restricted_cells,
            color="gold",
            alpha=0.08,
            edge_color="darkgoldenrod",
        )
    if no_fly_cells:
        draw_cell_cubes(
            ax,
            no_fly_cells,
            color="red",
            alpha=0.24,
            edge_color="firebrick",
        )

    for idx, point in enumerate(targets):
        x, y, z = point
        ax.scatter(x, y, z, s=90, c="limegreen", edgecolors="darkgreen", marker="o")
        ax.text(
            x,
            y,
            z + 0.25,
            f"Inspect #{idx + 1}",
            color="green",
            fontsize=9,
            fontweight="bold",
        )

    for idx, pad in enumerate(charging_pads):
        x, y, z = pad
        ax.scatter(x, y, z, s=95, c="mediumpurple", edgecolors="purple", marker="o")
        label = "Base/Recharge" if idx == 0 else f"Recharge #{idx}"
        ax.text(x, y, z - 0.35, label, color="purple", fontsize=8, fontweight="bold")


def make_legend(ax, has_restricted: bool):
    handles = [
        Line2D(
            [0],
            [0],
            color="royalblue",
            lw=2.2,
            marker="o",
            markersize=4,
            label="UAV Path (time step)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="limegreen",
            markeredgecolor="darkgreen",
            markersize=9,
            label="Inspection Target",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="mediumpurple",
            markeredgecolor="purple",
            markersize=9,
            label="Charging Station",
        ),
        Patch(facecolor="red", edgecolor="firebrick", alpha=0.24, label="No-Fly Cell"),
        Line2D([0], [0], color="darkorange", lw=2, label="Wind Drift"),
    ]
    if has_restricted:
        handles.append(
            Patch(
                facecolor="gold",
                edgecolor="darkgoldenrod",
                alpha=0.08,
                label="Restricted Cell",
            )
        )
    ax.legend(handles=handles, loc="upper left", fontsize=8)


def build_wind_arrows(rows):
    arrows = []
    for row in rows:
        if not row.get("info", {}).get("wind_drift"):
            continue
        state = np.array(row["state"][:3], dtype=float)
        next_state = np.array(row["next_state"][:3], dtype=float)
        wind_state = row["state"][4]
        drift_vec = WIND_DRIFT_DELTAS.get(wind_state, np.array([0.0, 0.0, 0.0]))
        if np.allclose(drift_vec, 0):
            continue
        action = row["action"]
        intent = ACTION_DELTAS.get(action, np.array([0.0, 0.0, 0.0]))
        anchor = state + intent
        actual_delta = next_state - anchor
        if np.linalg.norm(actual_delta) < 1e-8:
            actual_delta = drift_vec.astype(float)
        arrows.append((anchor, actual_delta))
    return arrows


def save_static_environment_image(scenario, output_path: Path, dpi: int):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    setup_axes(ax, scenario["grid_size"], title="UAV Mission Environment (Static Map)")
    draw_static_elements(ax, scenario)
    make_legend(ax, has_restricted=bool(scenario.get("restricted_cells")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a 3D UAV rollout animation from saved RL results."
    )
    parser.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--scenario", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-output", type=Path, default=None)
    parser.add_argument("--interval-ms", type=int, default=350)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rollout_path = args.rollout.resolve()
    result_dir = rollout_path.parent
    scenario_path = args.scenario.resolve() if args.scenario else detect_scenario_from_result_dir(result_dir)
    output_path = args.output.resolve()
    env_output_path = args.env_output.resolve() if args.env_output else (result_dir / DEFAULT_ENV_IMAGE.name).resolve()

    if not rollout_path.exists():
        raise FileNotFoundError(f"Rollout file not found: {rollout_path}")
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    rows = load_json(rollout_path)
    scenario = load_json(scenario_path)
    trajectory = rollout_to_trajectory(rows)
    wind_arrows = build_wind_arrows(rows)

    success_reached = any(r.get("info", {}).get("success", False) for r in rows)
    success_point = trajectory[-1]

    save_static_environment_image(scenario, env_output_path, dpi=args.dpi)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.cla()
        setup_axes(ax, scenario["grid_size"], title="Sample Rollout from Learned Q-learning Policy")
        draw_static_elements(ax, scenario)

        path = trajectory[: frame + 1]
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color="royalblue",
            linewidth=2.2,
            marker="o",
            markersize=4,
        )

        for i, p in enumerate(path):
            ax.text(p[0], p[1], p[2] + 0.12, str(i), color="royalblue", fontsize=8, fontweight="bold")

        current = trajectory[frame]
        ax.scatter(
            current[0],
            current[1],
            current[2],
            s=190,
            c="royalblue",
            marker="^",
            edgecolors="navy",
            linewidths=1.1,
        )

        for anchor, vec in wind_arrows[: frame + 1]:
            ax.quiver(
                anchor[0],
                anchor[1],
                anchor[2],
                vec[0],
                vec[1],
                vec[2],
                color="darkorange",
                arrow_length_ratio=0.30,
                linewidth=1.8,
            )

        if frame > 0:
            step_row = rows[min(frame - 1, len(rows) - 1)]
            action = step_row["action"]
            battery = step_row["next_state"][3]
            wind = step_row["next_state"][4]
            status = f"step={frame:02d} | action={action} | battery={battery} | wind={wind}"
            ax.text2D(0.63, 0.95, status, transform=ax.transAxes, fontsize=9, color="black")

        if success_reached and frame == len(trajectory) - 1:
            ax.scatter(
                success_point[0],
                success_point[1],
                success_point[2],
                s=220,
                c="gold",
                marker="*",
                edgecolors="orange",
                linewidths=1.5,
            )
            ax.text(
                success_point[0],
                success_point[1],
                success_point[2] + 0.3,
                "Mission Success",
                color="darkgoldenrod",
                fontsize=10,
                fontweight="bold",
            )

        make_legend(ax, has_restricted=bool(scenario.get("restricted_cells")))
        return ax,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        interval=args.interval_ms,
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        ani.save(str(output_path), writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    else:
        ani.save(str(output_path), dpi=args.dpi, fps=args.fps)

    print(f"Saved animation: {output_path}")
    print(f"Saved static environment image: {env_output_path}")
    print(f"Loaded rollout: {rollout_path}")
    print(f"Loaded scenario: {scenario_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
