# EL5001 Project

**Real-data-grounded 3D UAV solar inspection MDP with DP / model-free RL comparisons**

This project builds `scenario.json` from real sources (solar CSV + VWorld + KMA) and
compares tabular RL/DP methods on the resulting MDP.

- DP baseline: `Value Iteration`
- Model-free baseline: `Expected SARSA`
- Our model: `Risk-aware Q-learning`
- Main training/evaluation entry point: `main_good_mf.py`

---

## At a glance

| Item | Detail |
|---|---|
| Python | `3.10.x` recommended (`setup_pipenv.ps1` uses 3.10 by default) |
| Main entry points | `build_scenario.py`, `main_good_mf.py` |
| Quick run | `python -m pipenv run python main_good_mf.py --scenario data/scenario.json` |
| Outputs | `results/<run_name>/...` |
| Run location | Always from repo root (`c:\EL5001_project`) |

---

## Execution environment (current local profile)

The following values were collected from the current machine for reproducibility reference.

- OS: Windows 10
- CPU: `12th Gen Intel(R) Core(TM) i5-1235U` (10 cores / 12 threads)
- RAM: `34,046,570,496` bytes (about 31.7 GiB)
- Python: `3.10.11`

---

## 1) Installation

### A. Recommended: Pipenv setup scripts

PowerShell:

```powershell
./setup_pipenv.ps1
```

Bash:

```bash
./setup_pipenv.sh
```

### B. Manual install

```bash
python -m pip install -r requirements.txt
```

---

## 2) `.env` configuration

Copy `.env.example` to `.env` and fill in your keys:

```env
DATA_GO_KR_SERVICE_KEY=your_kma_key
VWORLD_KEY=your_vworld_key
VWORLD_DOMAIN=
```

Notes:
- `VWORLD_KEY` is required for `--use-vworld`
- `DATA_GO_KR_SERVICE_KEY` is required for `--use-kma`
- Never commit `.env`

---

## 3) Execution order (recommended)

Use the workflow below in order:

1. Use the provided `data/solar_api.csv`
2. Build scenario JSON (`build_scenario.py`)
3. Train/evaluate with the main script (`main_good_mf.py`)

---

## 4) Scenario generation

Input sources:
- `data/solar_api.csv` (provided solar facility data)
- VWorld API (no-fly / restricted cells)
- KMA API (initial wind state)

Example:

```bash
python build_scenario.py --solar-csv data/solar_api.csv --use-vworld --use-kma --output data/scenario.json
```

Region-filtered example:

```bash
python build_scenario.py --solar-csv data/solar_api.csv --region-keyword 광주 --use-vworld --use-kma --output data/scenario_real_gwangju_v1.json
```

---

## 5) Training / evaluation

### Main script (`main_good_mf.py`)
- DP baseline: VI
- Model-free baseline: Expected SARSA
- Our model: Risk-aware Q-learning

```bash
python -m pipenv run python main_good_mf.py --scenario data/scenario.json --output-dir results/good_mf_run
```

---

## 6) Output files

Common outputs:
- `metrics.csv`
- `mean_return.png`
- `success_rate.png`
- `sample_rollout.json`

Additional outputs:

- `episode_mae_vs_vi.png`
- `episode_mae_vs_vi.csv`
- `episode_training_trace.csv`
- `learning_curve_return.png`
- `learning_curve_success.png`
- `artifacts/run_manifest.json`
- `artifacts/model_tables.pkl`
- `artifacts/training_traces.pkl`

---

## 7) Reproducibility / post-processing tips

- For new visualizations, reuse saved traces/csv/pkl instead of rerunning training
- `run.txt` contains practical experiment command history
- Large generated files are excluded by `.gitignore` (`artifacts`, `*.pkl`, etc.)

---

## 8) Project flow

```text
data/solar_api.csv
    -> build_scenario.py + VWorld + KMA
    -> data/scenario*.json
    -> main_good_mf.py train/evaluate
    -> results/<run_name>/*
```



---

## Project Report

### 1. Project Overview

This project formulates a risk-aware 3D UAV solar panel inspection task as a tabular Markov Decision Process (MDP). The UAV agent must inspect multiple solar panel targets and return to the base while considering limited battery capacity, stochastic wind disturbance, no-fly zones, restricted areas, and charging-station queue conditions.

The main objective is not to build a full real-world UAV simulator, but to construct a simplified and interpretable tabular MDP that captures key sequential decision-making challenges in UAV inspection. The agent must decide when to move, change altitude, inspect a target, charge its battery, or avoid risky regions.

### 2. Why This Problem is Sequential

This problem is sequential because each action changes the future decision context. For example, moving toward a target consumes battery and may expose the UAV to wind drift or no-fly risk. Charging may reduce battery failure risk but introduces queue-related waiting cost. Inspecting a target changes the mission progress, and returning to base is only meaningful after all required targets have been inspected.

Therefore, the quality of an action cannot be evaluated only by its immediate reward. The agent must learn a long-term policy that balances mission completion, energy use, charging delay, and safety.

### 3. MDP Formulation

#### Agent

The decision-making agent is a UAV operating in a discretized 3D inspection environment.

#### State

The state is defined as:

\[
s_t = (x_t, y_t, z_t, b_t, w_t, q^c_t, q^i_t, n^{inspect}_t)
\]

where:

- `x, y, z`: UAV position in the 3D grid
- `b`: remaining battery level
- `w`: wind condition, one of `{Calm, EastWind, NorthWind}`
- `q_c`: charging queue state
- `q_i`: inspection progress or inspection-related queue state
- `n_inspect`: number of inspected targets

These variables are included because they are necessary for navigation, battery-safe planning, wind-aware movement, charging-delay decisions, and mission progress tracking.

In our simplified setting, the inspection order is assumed to be fixed, so `n_inspect` is sufficient to determine the next inspection target. If the inspection order is made flexible in future work, `n_inspect` should be replaced by a target inspection mask.

#### Action

The action space is:

\[
A = \{
Move\_N, Move\_S, Move\_E, Move\_W,
Ascend, Descend,
Hover, Charge, Inspect
\}
\]

The actions are grouped into three categories:

- Horizontal movement: `Move_N`, `Move_S`, `Move_E`, `Move_W`
- Vertical control: `Ascend`, `Descend`
- Task actions: `Hover`, `Charge`, `Inspect`

This action space is realistic for a simplified UAV inspection task because the UAV must navigate in 3D, manage altitude, inspect targets, and recharge when needed.

#### Transition Dynamics

The next state depends on the current state and selected action:

\[
P(s_{t+1} \mid s_t, a_t)
\]

The transition model includes:

1. Movement dynamics based on the selected action
2. Wind-dependent stochastic disturbance
3. Battery update after movement, ascent, descent, hovering, charging, or inspection
4. Queue-related cost when charging
5. Mission progress update after successful inspection
6. Safety outcomes when entering restricted or no-fly areas

The wind transition model is:

```text
Calm      -> Calm: 0.85, EastWind: 0.10, NorthWind: 0.05
EastWind  -> EastWind: 0.70, Calm: 0.20, NorthWind: 0.10
NorthWind -> NorthWind: 0.70, Calm: 0.20, EastWind: 0.10