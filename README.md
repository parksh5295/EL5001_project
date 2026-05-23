# Real-Data-Grounded 3D UAV Solar Panel Inspection RL

This project builds a small 3D tabular MDP for UAV solar panel inspection and compares:

- Value Iteration
- Q-learning
- SARSA

The scenario is grounded by:

- solar facility CSV data for inspection targets
- VWorld API for no-fly / restricted cells
- KMA API for initial wind state

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Prepare `.env`

Copy the example file:

```bash
cp .env.example .env
```

Fill it like this:

```env
DATA_GO_KR_SERVICE_KEY=your_data_go_kr_key_for_kma
VWORLD_KEY=your_vworld_key
VWORLD_DOMAIN=
```

Notes:

- `VWORLD_KEY` is needed for `--use-vworld`.
- `DATA_GO_KR_SERVICE_KEY` is needed for `--use-kma`.
- `VWORLD_DOMAIN` can usually be left blank. If VWorld requires it, put the registered domain such as `http://localhost`.
- Do not upload `.env` to GitHub.

## 3. Put solar CSV in `data/`

Download the public solar facility CSV and save it as:

```text
data/solar.csv
```

The builder automatically detects common Korean/English columns:

- 태양광발전시설명 / SOLAR_GEN_FCLT_NM
- 위도 / LATITUDE
- 경도 / LONGITUDE
- 가동상태구분명 / OPRTNG_STTS_SE_NM
- 설비용량 / CAPA
- 소재지도로명주소 / 소재지지번주소

## 4. Build scenario from real data

Solar CSV + VWorld + KMA:

```bash
python build_scenario.py --solar-csv data/solar.csv --use-vworld --use-kma --output data/scenario.json
```

If you do not have the KMA key yet, use VWorld and set wind manually:

```bash
python build_scenario.py --solar-csv data/solar.csv --use-vworld --wind EastWind --output data/scenario.json
```

If you want to limit the solar data to one region:

```bash
python build_scenario.py --solar-csv data/solar.csv --region-keyword 나주 --use-vworld --use-kma --output data/scenario.json
```

## 5. Run RL algorithms

```bash
python main.py --scenario data/scenario.json
```

## 6. Output

Results are saved in:

```text
results/metrics.csv
results/mean_return.png
results/success_rate.png
results/sample_rollout.json
```

## 7. Project logic

The real data are not used to simulate the whole world directly. Instead, they are converted into a small 3D grid MDP:

```text
solar CSV      -> inspection targets
VWorld API     -> no-fly/restricted cells
KMA API        -> initial wind state
scenario.json  -> tabular MDP for DP/Q-learning/SARSA
```
