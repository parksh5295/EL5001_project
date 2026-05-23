from __future__ import annotations

"""
Build data/scenario.json from real data.

Recommended run:
    python build_scenario.py --solar-csv data/solar.csv --use-vworld --use-kma --output data/scenario.json

Data roles:
    - Solar CSV: inspection targets
    - VWorld API: no-fly / restricted cells
    - KMA API: initial wind state

If KMA key is missing, run without --use-kma and set --wind manually.
"""

import argparse
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from urllib.parse import unquote
from dotenv import load_dotenv
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.ops import unary_union


GRID_SIZE = [4, 4, 3]
DEFAULT_WIND_TRANSITION = {
    "Calm": {"Calm": 0.70, "EastWind": 0.15, "NorthWind": 0.15},
    "EastWind": {"EastWind": 0.60, "Calm": 0.30, "NorthWind": 0.10},
    "NorthWind": {"NorthWind": 0.60, "Calm": 0.30, "EastWind": 0.10},
}
DEFAULT_QUEUE_TRANSITION = {
    "Short": {"Short": 0.70, "Long": 0.30},
    "Long": {"Long": 0.60, "Short": 0.40},
}


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def load_solar_csv(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def solar_targets_from_dataframe(
    df: pd.DataFrame,
    max_targets: int = 3,
    region_keyword: str | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    lat_col = pick_column(df, ["위도", "LATITUDE", "latitude", "lat"])
    lon_col = pick_column(df, ["경도", "LONGITUDE", "longitude", "lon", "lng"])
    status_col = pick_column(df, ["가동상태구분명", "OPRTNG_STTS_SE_NM", "status"])
    cap_col = pick_column(df, ["설비용량", "CAPA", "capacity"])
    name_col = pick_column(df, ["태양광발전시설명", "SOLAR_GEN_FCLT_NM", "name"])
    road_addr_col = pick_column(df, ["소재지도로명주소", "rdnmadr", "road_address"])
    lot_addr_col = pick_column(df, ["소재지지번주소", "lnmadr", "jibun_address"])

    if lat_col is None or lon_col is None:
        raise ValueError(f"Latitude/longitude columns not found. Columns: {list(df.columns)}")

    work = df.copy()
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=[lat_col, lon_col])

    if region_keyword:
        mask = pd.Series(False, index=work.index)
        for col in [road_addr_col, lot_addr_col, name_col]:
            if col is not None:
                mask = mask | work[col].astype(str).str.contains(region_keyword, na=False)
        if mask.any():
            work = work[mask]

    if status_col is not None:
        status = work[status_col].astype(str)
        normal = status.str.contains("정상|가동|운영|normal|active", case=False, na=False)
        if normal.any():
            work = work[normal]

    if cap_col is not None:
        work[cap_col] = pd.to_numeric(work[cap_col], errors="coerce")
        work = work.sort_values(cap_col, ascending=False)

    work = work.drop_duplicates(subset=[lat_col, lon_col]).head(max(30, max_targets * 10))
    if len(work) == 0:
        raise ValueError("No usable solar rows after filtering.")

    min_lat, max_lat = float(work[lat_col].min()), float(work[lat_col].max())
    min_lon, max_lon = float(work[lon_col].min()), float(work[lon_col].max())

    # Add padding so VWorld/KMA query covers the mission area.
    lat_pad = max((max_lat - min_lat) * 0.15, 0.01)
    lon_pad = max((max_lon - min_lon) * 0.15, 0.01)
    bbox = [min_lon - lon_pad, min_lat - lat_pad, max_lon + lon_pad, max_lat + lat_pad]

    lat_span = max(max_lat - min_lat, 1e-9)
    lon_span = max(max_lon - min_lon, 1e-9)

    nx, ny, _ = GRID_SIZE
    targets: list[list[int]] = []
    selected_rows = []
    seen = set()

    for _, row in work.iterrows():
        ix = int((float(row[lon_col]) - min_lon) / lon_span * (nx - 1))
        iy = int((float(row[lat_col]) - min_lat) / lat_span * (ny - 1))
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))
        cell = (ix, iy, 1)
        if cell in seen or cell == (0, 0, 0):
            continue
        targets.append([ix, iy, 1])
        seen.add(cell)
        selected_rows.append({
            "name": str(row[name_col]) if name_col else "solar_facility",
            "lat": float(row[lat_col]),
            "lon": float(row[lon_col]),
            "grid_cell": [ix, iy, 1],
        })
        if len(targets) >= max_targets:
            break

    fallback = [[1, 2, 1], [3, 1, 1], [2, 3, 2]]
    for cell in fallback:
        if len(targets) >= max_targets:
            break
        if tuple(cell) not in seen:
            targets.append(cell)
            seen.add(tuple(cell))

    metadata = {
        "solar_rows_used": selected_rows,
        "solar_bbox": bbox,
        "solar_source_rows_after_filter": int(len(work)),
        "region_keyword": region_keyword,
    }
    return targets, metadata


def fetch_vworld_layer(layer_code: str, bbox: list[float]) -> dict[str, Any]:
    key = os.getenv("VWORLD_KEY")
    domain = os.getenv("VWORLD_DOMAIN", "").strip()
    if not key:
        raise RuntimeError("VWORLD_KEY is missing. Put it in .env first.")

    min_lon, min_lat, max_lon, max_lat = bbox
    params = {
        "service": "data",
        "version": "2.0",
        "request": "GetFeature",
        "format": "json",
        "size": 1000,
        "page": 1,
        "data": layer_code,
        "geomFilter": f"BOX({min_lon},{min_lat},{max_lon},{max_lat})",
        "geometry": "true",
        "attribute": "true",
        "crs": "EPSG:4326",
        "key": key,
    }
    if domain:
        params["domain"] = domain

    r = requests.get("https://api.vworld.kr/req/data", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_vworld_features(resp: dict[str, Any]) -> list[dict[str, Any]]:
    # VWorld responses vary a little by layer/version, so search several common paths.
    candidates: list[Any] = [
        resp.get("features"),
        resp.get("response", {}).get("result", {}).get("featureCollection", {}).get("features"),
        resp.get("response", {}).get("result", {}).get("features"),
    ]
    for c in candidates:
        if isinstance(c, list):
            return c
    return []


def geometries_from_features(features: list[dict[str, Any]]) -> list[Any]:
    geoms = []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            geoms.append(shape(geom))
        except Exception:
            continue
    return geoms


def grid_cell_centers(bbox: list[float]) -> dict[tuple[int, int], Point]:
    min_lon, min_lat, max_lon, max_lat = bbox
    nx, ny, _ = GRID_SIZE
    dx = (max_lon - min_lon) / nx
    dy = (max_lat - min_lat) / ny
    centers = {}
    for ix in range(nx):
        for iy in range(ny):
            lon = min_lon + (ix + 0.5) * dx
            lat = min_lat + (iy + 0.5) * dy
            centers[(ix, iy)] = Point(lon, lat)
    return centers


def expand_to_3d(cells_2d: Iterable[tuple[int, int]], altitude_mode: str = "all") -> list[list[int]]:
    _, _, nz = GRID_SIZE
    out = []
    for ix, iy in sorted(set(cells_2d)):
        if altitude_mode == "middle":
            out.append([ix, iy, 1])
        else:
            for iz in range(nz):
                out.append([ix, iy, iz])
    return out


def cells_from_vworld_geometries(geoms: list[Any], bbox: list[float]) -> list[tuple[int, int]]:
    if not geoms:
        return []
    try:
        merged = unary_union(geoms)
    except Exception:
        merged = geoms[0]
    cells = []
    for cell, pt in grid_cell_centers(bbox).items():
        try:
            # contains center OR the polygon is very close to the center.
            if merged.contains(pt) or merged.distance(pt) < 1e-9:
                cells.append(cell)
        except Exception:
            continue
    return cells


def vworld_cells(use_vworld: bool, bbox: list[float]) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    meta: dict[str, Any] = {}
    if not use_vworld:
        return [], [], {"vworld_mode": "not_used"}

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    no_fly_cells: list[list[int]] = []
    restricted_cells: list[list[int]] = []

    try:
        nofly_resp = fetch_vworld_layer("LT_C_AISPRHC", bbox)
        restricted_resp = fetch_vworld_layer("LT_C_AISRESC", bbox)

        (raw_dir / "vworld_nofly.json").write_text(json.dumps(nofly_resp, ensure_ascii=False, indent=2), encoding="utf-8")
        (raw_dir / "vworld_restricted.json").write_text(json.dumps(restricted_resp, ensure_ascii=False, indent=2), encoding="utf-8")

        nofly_features = extract_vworld_features(nofly_resp)
        restricted_features = extract_vworld_features(restricted_resp)

        nofly_geoms = geometries_from_features(nofly_features)
        restricted_geoms = geometries_from_features(restricted_features)

        nofly_2d = cells_from_vworld_geometries(nofly_geoms, bbox)
        restricted_2d = cells_from_vworld_geometries(restricted_geoms, bbox)

        no_fly_cells = expand_to_3d(nofly_2d, altitude_mode="all")
        restricted_cells = expand_to_3d(restricted_2d, altitude_mode="middle")

        meta = {
            "vworld_mode": "api_polygon_to_grid",
            "vworld_layers": {
                "no_fly": "LT_C_AISPRHC",
                "restricted": "LT_C_AISRESC",
            },
            "vworld_raw_files": ["data/raw/vworld_nofly.json", "data/raw/vworld_restricted.json"],
            "vworld_feature_counts": {
                "no_fly": len(nofly_features),
                "restricted": len(restricted_features),
            },
            "vworld_cells_2d": {
                "no_fly": [list(c) for c in sorted(nofly_2d)],
                "restricted": [list(c) for c in sorted(restricted_2d)],
            },
        }
    except Exception as e:
        meta = {"vworld_mode": "api_failed", "vworld_error": str(e)}

    return no_fly_cells, restricted_cells, meta


def latlon_to_kma_grid(lat: float, lon: float) -> tuple[int, int]:
    RE = 6371.00877
    GRID = 5.0
    SLAT1 = 30.0
    SLAT2 = 60.0
    OLON = 126.0
    OLAT = 38.0
    XO = 43
    YO = 136
    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf ** sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro ** sn)
    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / (ra ** sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn
    x = int(ra * math.sin(theta) + XO + 0.5)
    y = int(ro - ra * math.cos(theta) + YO + 0.5)
    return x, y


def kma_base_time_candidates(hours_back: int = 8) -> list[tuple[str, str]]:
    """Return recent base_date/base_time candidates for KMA ultra-short nowcast.

    KMA values can be delayed, so trying only the latest hour often fails.
    This function tries several previous full hours until a valid response is found.
    """
    now = datetime.now()
    candidates: list[tuple[str, str]] = []
    for h in range(1, hours_back + 1):
        base = now - timedelta(hours=h)
        candidates.append((base.strftime("%Y%m%d"), base.strftime("%H00")))
    return candidates


def _clean_kma_key(raw_key: str) -> str:
    """Accept both Encoding and Decoding keys copied from data.go.kr."""
    return unquote(raw_key.strip())


def _parse_kma_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    response = data.get("response", {})
    header = response.get("header", {})
    code = str(header.get("resultCode", ""))
    msg = str(header.get("resultMsg", ""))
    if code and code != "00":
        raise RuntimeError(f"KMA resultCode={code}, resultMsg={msg}")

    body = response.get("body", {})
    items = body.get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    if not items:
        raise RuntimeError("KMA response has no weather items for this base_time.")
    return items


def fetch_kma_wind_state(center_lat: float, center_lon: float) -> tuple[str, dict[str, Any]]:
    key = os.getenv("DATA_GO_KR_SERVICE_KEY", "").strip()
    if not key:
        raise RuntimeError("DATA_GO_KR_SERVICE_KEY is missing. Put your data.go.kr KMA key in .env.")
    key = _clean_kma_key(key)

    nx, ny = latlon_to_kma_grid(center_lat, center_lon)
    endpoint = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

    last_error = None
    for base_date, base_time in kma_base_time_candidates(hours_back=8):
        params = {
            "serviceKey": key,
            "pageNo": "1",
            "numOfRows": "1000",
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": str(nx),
            "ny": str(ny),
        }
        try:
            r = requests.get(endpoint, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            items = _parse_kma_items(data)
            obs = {str(item["category"]): item["obsrValue"] for item in items if "category" in item}

            wsd = float(obs.get("WSD", 0.0))
            if wsd < 3.0:
                wind = "Calm"
            elif "VEC" in obs:
                # VEC is the direction the wind comes FROM; drone drift is roughly opposite.
                drift = (float(obs["VEC"]) + 180.0) % 360.0
                if 45 <= drift < 135:
                    wind = "EastWind"
                elif 315 <= drift or drift < 45:
                    wind = "NorthWind"
                else:
                    wind = "EastWind"
            else:
                wind = "EastWind"

            raw_dir = Path("data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)
            with open(raw_dir / "kma_nowcast.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return wind, {
                "weather_mode": "kma_api",
                "kma_obs": obs,
                "kma_grid": [nx, ny],
                "base_date": base_date,
                "base_time": base_time,
                "mission_center": [center_lat, center_lon],
                "kma_endpoint": endpoint,
            }
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"KMA API failed for all recent base times. Last error: {last_error}")


def build_scenario(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()

    if not args.solar_csv:
        raise RuntimeError("Use --solar-csv data/solar.csv. Solar targets are built from downloaded real CSV data.")

    solar_df = load_solar_csv(args.solar_csv)
    targets, solar_meta = solar_targets_from_dataframe(
        solar_df,
        max_targets=args.targets,
        region_keyword=args.region_keyword,
    )

    bbox = solar_meta["solar_bbox"]
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2

    no_fly, restricted, vworld_meta = vworld_cells(args.use_vworld, bbox)

    wind = args.wind
    weather_meta: dict[str, Any] = {"weather_mode": "manual", "manual_wind": wind}
    if args.use_kma:
        wind, weather_meta = fetch_kma_wind_state(center_lat, center_lon)

    scenario = {
        "grid_size": GRID_SIZE,
        "base": [0, 0, 0],
        "targets": targets,
        "charging_pads": [[0, 3, 0], [3, 0, 0]],
        "no_fly_cells": no_fly,
        "restricted_cells": restricted,
        "initial_wind": wind,
        "wind_states": ["Calm", "EastWind", "NorthWind"],
        "wind_transition": DEFAULT_WIND_TRANSITION,
        "queue_transition": DEFAULT_QUEUE_TRANSITION,
        "max_battery": 16,
        "max_steps": 120,
        "reward": {
            "inspect_new_target": 25,
            "finish_all_and_return_base": 100,
            "step_cost": -1,
            "move_battery_cost": -1,
            "ascend_extra_battery_cost": -1,
            "charge_short_queue_cost": -3,
            "charge_long_queue_cost": -12,
            "restricted_area_cost": -10,
            "no_fly_violation": -80,
            "battery_depletion": -80,
            "invalid_action_cost": -4,
        },
        "metadata": {
            "data_grounding": "solar_csv + VWorld_API + KMA_API_or_manual_wind",
            **solar_meta,
            **vworld_meta,
            **weather_meta,
        },
    }
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scenario.json from real solar CSV, VWorld API, and KMA API")
    parser.add_argument("--solar-csv", required=True, help="Path to downloaded solar CSV file, e.g., data/solar.csv")
    parser.add_argument("--region-keyword", default=None, help="Optional Korean address keyword such as 나주, 광주, 전남")
    parser.add_argument("--use-vworld", action="store_true", help="Use VWorld API to create no-fly/restricted cells")
    parser.add_argument("--use-kma", action="store_true", help="Use KMA API to set initial wind")
    parser.add_argument("--wind", default="Calm", choices=["Calm", "EastWind", "NorthWind"], help="Manual wind if --use-kma is not used")
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--output", default="data/scenario.json")
    args = parser.parse_args()

    scenario = build_scenario(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved scenario to {out}")
    print(json.dumps(scenario, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
