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

---

### 2. Why This Problem is Sequential

This problem is sequential because each action changes the future decision context. For example, moving toward a target consumes battery and may expose the UAV to wind drift or no-fly risk. Charging may reduce battery failure risk but introduces queue-related waiting cost. Inspecting a target changes the mission progress, and returning to base is only meaningful after all required targets have been inspected.

Therefore, the quality of an action cannot be evaluated only by its immediate reward. The agent must learn a long-term policy that balances mission completion, energy use, charging delay, and safety.

---

### 3. MDP Formulation

#### Agent

The decision-making agent is a UAV operating in a discretized 3D inspection environment.

#### State

The state is defined as:

$$
s_t = (x_t, y_t, z_t, b_t, w_t, q^c_t, q^i_t, n^{inspect}_t)
$$

where:

| Symbol | Meaning |
|---|---|
| $x, y, z$ | UAV position in the 3D grid |
| $b$ | Remaining battery level |
| $w$ | Wind condition: Calm, EastWind, or NorthWind |
| $q_c$ | Charging queue state |
| $q_i$ | Inspection progress or inspection-related state |
| $n_{inspect}$ | Number of inspected targets |

These variables are included because they are necessary for navigation, battery-safe planning, wind-aware movement, charging-delay decisions, and mission progress tracking.

In this simplified setting, the inspection order is assumed to be fixed, so `n_inspect` is sufficient to determine the next inspection target. If the inspection order is made flexible in future work, `n_inspect` should be replaced by a target inspection mask.

#### Action

The action space is:

$$
A = \{
Move\_N, Move\_S, Move\_E, Move\_W,
Ascend, Descend,
Hover, Charge, Inspect
\}
$$

The actions are grouped into three categories:

| Category | Actions |
|---|---|
| Horizontal movement | Move_N, Move_S, Move_E, Move_W |
| Vertical control | Ascend, Descend |
| Task actions | Hover, Charge, Inspect |

This action space is simplified but realistic for a tabular UAV inspection task because the UAV must navigate in 3D, manage altitude, inspect targets, and recharge when needed.

#### Transition Dynamics

The next state depends on the current state and selected action:

$$
P(s_{t+1} \mid s_t, a_t)
$$

The transition model includes:

1. Movement dynamics based on the selected action
2. Wind-dependent stochastic disturbance
3. Battery update after movement, ascent, descent, hovering, charging, or inspection
4. Queue-related cost when charging
5. Mission progress update after successful inspection
6. Safety outcomes when entering restricted or no-fly areas

The wind transition model is:

| Current wind | Next wind probabilities |
|---|---|
| Calm | Calm: 0.85, EastWind: 0.10, NorthWind: 0.05 |
| EastWind | EastWind: 0.70, Calm: 0.20, NorthWind: 0.10 |
| NorthWind | NorthWind: 0.70, Calm: 0.20, EastWind: 0.10 |

The state is approximately Markovian because the next state and reward can be determined from the current UAV position, battery level, wind condition, queue state, inspection progress, and selected action without requiring the full history.

---

### 4. Reward Function

The reward is designed as:

$$r_t = R_t^{\mathrm{mission}} - C_t^{\mathrm{operation}} - C_t^{\mathrm{safety}}$$

Mission reward:

$$R_t^{\mathrm{mission}} = 40 \cdot \phi_{\mathrm{new}} + 200 \cdot \phi_{\mathrm{success}}$$

Operation cost:

$$C_t^{\mathrm{operation}} = 1 + \phi_{\mathrm{flight}} + \phi_{\mathrm{ascend}} + c_q(q_t) \cdot \phi_{\mathrm{charge}}$$

Safety cost:

$$C_t^{\mathrm{safety}} = 10 \cdot \phi_{\mathrm{restricted}} + 80 \cdot \phi_{\mathrm{nofly}} + 120 \cdot \phi_{\mathrm{battery}} + 20 \cdot \phi_{\mathrm{invalid}}$$

Charging queue cost:

$$
c_q(q_t)=
\begin{cases}
1, & q_t = \text{short queue} \\
5, & q_t = \text{long queue}
\end{cases}
$$

Here, each $\phi$ is a binary feature that becomes 1 when the corresponding event occurs and 0 otherwise.

| Term | Meaning |
|---|---|
| $\phi_{\mathrm{new}}$ | 1 if a new inspection target is inspected |
| $\phi_{\mathrm{success}}$ | 1 if all targets are inspected and the UAV returns to base |
| $\phi_{\mathrm{flight}}$ | 1 if the UAV takes a flight movement action |
| $\phi_{\mathrm{ascend}}$ | 1 if the UAV ascends |
| $\phi_{\mathrm{charge}}$ | 1 if the UAV charges |
| $\phi_{\mathrm{restricted}}$ | 1 if the UAV enters a restricted area |
| $\phi_{\mathrm{nofly}}$ | 1 if the UAV violates a no-fly zone |
| $\phi_{\mathrm{battery}}$ | 1 if the UAV depletes its battery |
| $\phi_{\mathrm{invalid}}$ | 1 if the UAV takes an invalid action |

The reward function is designed to balance four objectives:

1. Complete the inspection mission
2. Reduce energy consumption
3. Reduce charging delay
4. Avoid safety-critical failures

The new-target reward provides intermediate feedback, while the final success reward encourages full mission completion. Safety failures such as battery depletion and no-fly violations receive large penalties so that the agent does not take risky shortcuts merely to inspect one more target.

---

### 5. Episode Termination

An episode terminates under the following conditions.

#### Success

- All inspection targets are inspected.
- The UAV returns to the base.

#### Failure

- The battery is depleted.
- The UAV enters a no-fly zone.
- The maximum step limit is reached.

The maximum episode length is set to 150 steps.

---

### 6. Algorithms

We compare three algorithms: Value Iteration, Expected SARSA, and Risk-aware Q-learning.

#### 6.1 Value Iteration

Value Iteration is used as the model-based Dynamic Programming baseline. It assumes access to the full transition and reward model and computes a reference policy for the designed MDP.

Value Iteration is not the main model-free RL solution. Instead, it provides a model-based reference policy that shows how well the MDP can be solved when the transition model is known.

#### 6.2 Expected SARSA

Expected SARSA is used as the model-free RL baseline. It is an on-policy method that updates action values using the expected value of the next action under the current policy.

Expected SARSA is useful as a baseline because it accounts for the exploration policy during learning and can produce more stable, exploration-aware updates.

#### 6.3 Risk-aware Q-learning

Risk-aware Q-learning is the main model-free RL solution. It uses the standard tabular Q-learning update rule, but the reward function explicitly includes risk-related penalties such as no-fly violation, battery depletion, restricted-area visits, and invalid actions.

The Q-learning update is:

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t) \right]$$

This method is suitable because it learns an adaptive state-dependent policy from sampled interaction without requiring a known transition model during training.

---

### 7. Training Setup

The experiment uses the following training configuration:

| Parameter | Value |
|---|---:|
| Grid size | 8 × 8 × 3 |
| Training episodes | 800,000 |
| Evaluation episodes | 300 |
| Discount factor $\gamma$ | 0.99 |
| Q-learning $\alpha$ | 0.05 |
| Expected SARSA $\alpha$ | 0.04 |
| $\alpha$ schedule | decays to 0.03 |
| $\epsilon$ schedule | 1.0 → 0.08 |
| Max battery | 18 |
| Max steps | 150 |

A high discount factor is used because the task requires long-term planning. The UAV may need to temporarily move away from a target, visit a charging station, or avoid a risky area in order to complete the full mission later.

The exploration rate starts high and decays gradually to allow the agent to explore the large 3D state-action space before converging to a more stable policy.

---

### 8. Evaluation Metrics

We evaluate each method using the following metrics:

| Metric | Meaning |
|---|---|
| Success rate | Fraction of episodes where all targets are inspected and the UAV returns to base |
| Mean return | Average cumulative reward per evaluation episode |
| Average steps | Average number of steps per episode |
| Battery failure count | Number of episodes where the UAV fails due to battery depletion |
| Charging count | Total number of charging actions |

These metrics are selected because the task requires not only high reward but also safe and efficient mission completion.

When available, no-fly violations and restricted-area visits should also be reported because they directly measure safety behavior.

---

### 9. Results

The quantitative evaluation is summarized below.

| Algorithm | Success Rate | Mean Return | Avg Steps | Battery Failure | Charging Count |
|---|---:|---:|---:|---:|---:|
| Value Iteration | 1.00 | 203.70 | 31.81 | 0 | 1117 |
| Risk-aware Q-learning | 0.78 | 135.58 | 34.17 | 66 | 679 |
| Expected SARSA | 0.74 | 127.75 | 30.07 | 79 | 684 |

Value Iteration achieves the highest success rate and return because it uses the known transition model. Risk-aware Q-learning performs best among the model-free methods, achieving a higher success rate and mean return than Expected SARSA. Expected SARSA shows more stable and exploration-aware learning behavior, but it learns more slowly and reaches a slightly lower final return.

The learning curves show that model-free RL initially struggles because successful trajectories are rare in the large stochastic 3D environment. After sufficient exploration, both Risk-aware Q-learning and Expected SARSA improve their success rates and returns. However, both methods remain below the Value Iteration reference, showing a remaining optimality gap between model-based planning and sample-based learning.

---

### 10. Interpretation and Design Justification

#### 10.1 Why the State Definition is Reasonable

The state definition includes position, battery, wind, queue, and inspection progress because each component affects either the transition dynamics or the reward.

| State component | Reason for inclusion |
|---|---|
| $x, y, z$ | Required for navigation and obstacle avoidance |
| $b$ | Required for battery-safe planning |
| $w$ | Required because wind changes movement outcomes |
| $q_c$ | Required for charging-delay decisions |
| $q_i, n_{inspect}$ | Required to track mission progress |

The following information is excluded to keep the problem tabular and computationally tractable:

- Continuous UAV dynamics
- Exact wind speed
- Real-time solar panel image quality
- Continuous battery voltage
- Detailed charger physics

These exclusions simplify the real UAV inspection problem while preserving the key sequential decision-making structure.

#### 10.2 Why the Action Space is Reasonable

The action space includes horizontal movement, vertical control, hovering, charging, and inspection. These actions are sufficient for a simplified 3D UAV inspection mission because the UAV must navigate through space, adjust altitude, inspect targets, and recharge when necessary.

#### 10.3 Why the Reward Function Reflects the Real-world Objective

The reward function reflects the real-world objective by rewarding mission progress and completion while penalizing time, energy use, charging delay, invalid actions, restricted-area visits, no-fly violations, and battery depletion.

The reward scale was chosen so that mission completion dominates normal travel costs, while safety failures dominate partial progress. This prevents the agent from taking risky shortcuts just to inspect one additional target.

#### 10.4 Learned Policy Behavior

The learned Q-learning policy shows reasonable behavior in important states.

| Situation | Expected behavior | Learned behavior |
|---|---|---|
| Low battery near charging station | Charge | Q-learning chooses Charge |
| Near no-fly zone with wind | Avoid risky direction | Q-learning moves away or ascends |
| At uninspected target | Inspect | Q-learning chooses Inspect |
| After all targets inspected | Return to base | Q-learning moves toward base |
| Long charging queue | Avoid if possible | Q-learning prefers an alternative route |

These examples show that the learned policy is not simply following a fixed route. Instead, it chooses actions based on the current battery, wind, queue, and inspection progress.

---

### 11. Limitations

This project uses a simplified tabular environment. Continuous UAV dynamics, exact wind speed, continuous battery voltage, real-time solar panel image quality, and detailed charger physics are excluded.

The wind and queue models are simplified as discrete stochastic processes. Battery consumption is also represented using discrete costs rather than detailed physical power modeling.

The model-free methods require many interaction episodes to learn useful policies. Scalability to larger real-world UAV environments was not fully explored.

#### Method-specific Limitations

| Method | Strength | Limitation |
|---|---|---|
| Value Iteration | Provides an optimal reference when the transition model is known | Requires full environment dynamics and scales poorly in large state spaces |
| Risk-aware Q-learning | Learns adaptive policies through interaction | Requires many training episodes and is sensitive to reward and $\epsilon$ schedule settings |
| Expected SARSA | Provides stable and exploration-aware learning | Generally converges more slowly and achieves slightly lower final return than Q-learning |

---

### 12. Summary

This project demonstrates that a risk-aware 3D UAV solar panel inspection task can be formulated as a tabular MDP and solved using classical model-free reinforcement learning.

Value Iteration provides a strong model-based reference policy, while Risk-aware Q-learning and Expected SARSA learn adaptive policies from interaction. The results show that model-free RL can learn meaningful UAV mission behavior, including inspection, charging, and risk avoidance.

Risk-aware Q-learning learns faster and achieves better model-free performance than Expected SARSA in this experiment. However, both model-free methods remain below the Value Iteration reference, showing the difficulty of learning safe long-horizon behavior in a large stochastic MDP using only sampled experience.