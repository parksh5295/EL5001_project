# EL5001 Project

**Real-data-grounded 3D UAV solar inspection MDP with DP / model-free RL comparisons**

This project builds `scenario.json` from real sources (solar CSV + VWorld + KMA) and
compares tabular RL/DP methods on the resulting MDP.

- DP baseline: `Value Iteration`
- Model-free baselines: `Q-learning`, `SARSA`, `Expected SARSA`
- Model-free variants: `Double Q-learning`, `SARSA(lambda)`, `Action-masked Q-learning`, `Risk-aware Q-learning`
- Assignment-focused script: `main_good_mf.py` (VI + Expected SARSA + Risk-aware Q-learning)

---

## At a glance

| Item | Detail |
|---|---|
| Python | `3.10.x` recommended (`setup_pipenv.ps1` uses 3.10 by default) |
| Main entry points | `build_scenario.py`, `main.py`, `main_model_free.py`, `main_good_mf.py` |
| Quick run | `python -m pipenv run python main.py --scenario data/scenario.json` |
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

## 3) Scenario generation

Input sources:
- `data/solar.csv` (solar facility data)
- VWorld API (no-fly / restricted cells)
- KMA API (initial wind state)

Example:

```bash
python build_scenario.py --solar-csv data/solar.csv --use-vworld --use-kma --output data/scenario.json
```

Region-filtered example:

```bash
python build_scenario.py --solar-csv data/solar.csv --region-keyword 광주 --use-vworld --use-kma --output data/scenario_real_gwangju_v1.json
```

---

## 4) Training / evaluation scripts

### 4-1. Base comparison (`main.py`)
- VI + Q-learning + SARSA
- Includes `episode_mae_vs_vi.png` by default

```bash
python -m pipenv run python main.py --scenario data/scenario.json --output-dir results/base_run
```

### 4-2. Extended model-free comparison (`main_model_free.py`)
- Q / SARSA / Expected SARSA / Double Q / SARSA(lambda) / Action-masked / Risk-aware
- Reuses checkpoints on rerun with same config (`results/.../checkpoints`)

```bash
python -m pipenv run python main_model_free.py --scenario data/scenario.json --output-dir results/model_free_run
```

### 4-3. Assignment-focused 3-model setup (`main_good_mf.py`)
- DP baseline: VI
- Model-free baseline: Expected SARSA
- Our model: Risk-aware Q-learning

```bash
python -m pipenv run python main_good_mf.py --scenario data/scenario.json --output-dir results/good_mf_run
```

---

## 5) Output files

Common outputs:
- `metrics.csv`
- `mean_return.png`
- `success_rate.png`
- `sample_rollout.json`

Additional outputs by script:

- `main.py`, `main_good_mf.py`:
  - `episode_mae_vs_vi.png`
  - `episode_mae_vs_vi.csv`
- `main_good_mf.py`:
  - `episode_training_trace.csv`
  - `learning_curve_return.png`
  - `learning_curve_success.png`
  - `artifacts/run_manifest.json`
  - `artifacts/model_tables.pkl`
  - `artifacts/training_traces.pkl`
- `main_model_free.py`:
  - `action_masked_progress.png`
  - `action_masked_progress.json`
  - `checkpoints/*.pkl`

---

## 6) Reproducibility / post-processing tips

- For new visualizations, reuse saved traces/csv/pkl instead of rerunning training
- `run.txt` contains practical experiment command history
- Large generated files are excluded by `.gitignore` (`artifacts`, `checkpoints`, `*.pkl`, etc.)

---

## 7) Project flow

```text
data/solar.csv + VWorld + KMA
    -> build_scenario.py
    -> data/scenario*.json
    -> main*.py train/evaluate
    -> results/<run_name>/*
```
