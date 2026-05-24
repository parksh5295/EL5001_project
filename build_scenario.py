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
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from urllib.parse import unquote
from dotenv import load_dotenv
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.ops import unary_union


GRID_SIZE = [8, 8, 3]
DEFAULT_WIND_TRANSITION = {
    "Calm": {"Calm": 0.85, "EastWind": 0.10, "NorthWind": 0.05},
    "EastWind": {"EastWind": 0.70, "Calm": 0.20, "NorthWind": 0.10},
    "NorthWind": {"NorthWind": 0.70, "Calm": 0.20, "EastWind": 0.10},
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
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


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

    work = work.drop_duplicates(subset=[lat_col, lon_col])
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
        ix = int((float(row[lon_col]) - min_lon) / lon_span * nx)
        iy = int((float(row[lat_col]) - min_lat) / lat_span * ny)
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))
        cell = (ix, iy, 1)
        if cell in seen:
            continue

    # avoid targets too close to base
        if abs(ix - 0) + abs(iy - 0) + abs(1 - 0) <= 1:
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


def _parse_region_candidates(region_keyword: str | None, fallback_keywords: str | None) -> list[str | None]:
    candidates: list[str | None] = []
    if region_keyword:
        candidates.append(region_keyword)
    if fallback_keywords:
        for kw in fallback_keywords.split(","):
            clean = kw.strip()
            if clean and clean not in candidates:
                candidates.append(clean)
    if not candidates:
        candidates.append(None)
    return candidates


def choose_targets_with_region_fallback(
    solar_df: pd.DataFrame,
    max_targets: int,
    region_keyword: str | None,
    fallback_keywords: str | None,
    min_solar_rows: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    candidates = _parse_region_candidates(region_keyword, fallback_keywords)
    best_targets: list[list[int]] | None = None
    best_meta: dict[str, Any] | None = None

    for kw in candidates:
        targets, meta = solar_targets_from_dataframe(
            solar_df,
            max_targets=max_targets,
            region_keyword=kw,
        )
        rows = int(meta.get("solar_source_rows_after_filter", 0))
        meta["region_keyword_selected"] = kw
        if rows >= min_solar_rows:
            return targets, meta
        if best_meta is None or rows > int(best_meta.get("solar_source_rows_after_filter", 0)):
            best_targets, best_meta = targets, meta

    assert best_targets is not None and best_meta is not None
    best_meta["region_fallback_warning"] = (
        f"Best candidate has only {best_meta.get('solar_source_rows_after_filter', 0)} rows "
        f"(required >= {min_solar_rows})."
    )
    return best_targets, best_meta


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

        nx, ny, _ = GRID_SIZE
        nofly_ratio = len(set(nofly_2d)) / (nx * ny)

# If VWorld no-fly covers almost the entire small mission area,
# treating it as terminal no-fly makes the MDP infeasible.
        if nofly_ratio > 0.40:
            broad_nofly_2d = nofly_2d
            nofly_2d = []
        else:
            broad_nofly_2d = []

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
            "nofly_coverage_ratio": nofly_ratio,
            "broad_nofly_cells_2d": [list(c) for c in sorted(broad_nofly_2d)],
        }
    except Exception as e:
        meta = {"vworld_mode": "api_failed", "vworld_error": str(e)}

    return no_fly_cells, restricted_cells, meta


def _to_2d_set(cells_3d: list[list[int]]) -> set[tuple[int, int]]:
    return {(int(c[0]), int(c[1])) for c in cells_3d}


def _to_3d_set(cells_3d: list[list[int]]) -> set[tuple[int, int, int]]:
    return {(int(c[0]), int(c[1]), int(c[2])) for c in cells_3d}


def _neighbors(pos: tuple[int, int, int], grid_size: list[int]) -> list[tuple[int, int, int]]:
    x, y, z = pos
    nx, ny, nz = grid_size
    cand = [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1),
    ]
    return [p for p in cand if 0 <= p[0] < nx and 0 <= p[1] < ny and 0 <= p[2] < nz]


def _reachable_from(start: tuple[int, int, int], blocked: set[tuple[int, int, int]], grid_size: list[int]) -> set[tuple[int, int, int]]:
    if start in blocked:
        return set()
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in _neighbors(cur, grid_size):
            if nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    return seen


def relax_and_validate_airspace(
    no_fly_cells: list[list[int]],
    restricted_cells: list[list[int]],
    base: list[int],
    targets: list[list[int]],
    charging_pads: list[list[int]],
    grid_size: list[int],
    nofly_coverage_limit: float,
) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    nx, ny, nz = grid_size
    nofly2d = _to_2d_set(no_fly_cells)
    restricted2d = _to_2d_set(restricted_cells)
    protected2d = {(base[0], base[1])}
    protected2d |= {(p[0], p[1]) for p in charging_pads}
    protected2d |= {(t[0], t[1]) for t in targets}

    # Keep mission-critical cells feasible even when VWorld polygon fully covers a tiny grid.
    nofly2d -= protected2d
    restricted2d -= nofly2d

    coverage_before = len(nofly2d) / max(1, nx * ny)
    downgraded = False
    if coverage_before > nofly_coverage_limit:
        downgraded = True
        restricted2d |= nofly2d
        nofly2d = set()

    # Use middle-altitude no-fly to avoid over-constraining tiny toy grids.
    no_fly_relaxed: list[list[int]] = [[x, y, 1] for (x, y) in sorted(nofly2d) if nz > 1]
    restricted_relaxed = expand_to_3d(sorted(restricted2d), altitude_mode="middle")

    blocked = _to_3d_set(no_fly_relaxed)
    start = (base[0], base[1], base[2])
    reachable = _reachable_from(start, blocked, grid_size)
    targets_reachable = all((t[0], t[1], t[2]) in reachable for t in targets)

    meta = {
        "vworld_relaxation": {
            "protected_cells_removed_from_nofly_2d": [list(c) for c in sorted(protected2d)],
            "nofly_2d_coverage_before_relax": coverage_before,
            "downgraded_nofly_to_restricted_due_to_coverage": downgraded,
            "nofly_altitude_mode_after_relax": "middle" if nz > 1 else "all",
            "targets_reachable_from_base_after_relax": targets_reachable,
        }
    }
    return no_fly_relaxed, restricted_relaxed, meta

def auto_place_charging_pads(
    base: list[int],
    targets: list[list[int]],
    grid_size: list[int],
    no_fly_cells: list[list[int]],
    restricted_cells: list[list[int]],
) -> list[list[int]]:

    charging_pads = []

    no_fly_set = {tuple(c) for c in no_fly_cells}
    restricted_set = {tuple(c) for c in restricted_cells}

    def is_valid(cell):
        x, y, z = cell

        return (
            0 <= x < grid_size[0]
            and 0 <= y < grid_size[1]
            and z == 0
            and tuple(cell) not in no_fly_set
        )

    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def add_if_valid(cell):

        if not is_valid(cell):
            return False

        # 너무 가까운 charging station 중복 방지
        if any(manhattan(cell, p) < 2 for p in charging_pads):
            return False

        charging_pads.append(cell)
        return True

    # 1. base는 항상 charging station
    add_if_valid([base[0], base[1], 0])

    # 2. target 근처 charging station 자동 배치
    for target in targets:

        x, y, _ = target

        # target 바로 아래 ground 우선 시도
        candidate = [x, y, 0]

        if add_if_valid(candidate):
            continue

        # 실패 시 주변 safe cell 탐색
        for radius in range(1, 4):

            found = False

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):

                    if abs(dx) + abs(dy) != radius:
                        continue

                    nearby = [x + dx, y + dy, 0]

                    if add_if_valid(nearby):
                        found = True
                        break

                if found:
                    break

            if found:
                break

    return charging_pads


def validate_scenario(
    scenario: dict[str, Any],
    min_solar_rows: int,
    nofly_coverage_limit: float,
) -> None:
    grid_size = scenario["grid_size"]
    nx, ny, _ = grid_size
    nofly3d = scenario.get("no_fly_cells", [])
    base = scenario["base"]
    targets = scenario["targets"]
    meta = scenario.get("metadata", {})

    base_in_nofly = tuple(base) in _to_3d_set(nofly3d)
    nofly2d = _to_2d_set(nofly3d)
    nofly_coverage = len(nofly2d) / max(1, nx * ny)
    solar_rows = int(meta.get("solar_source_rows_after_filter", 0))
    reachable = _reachable_from(tuple(base), _to_3d_set(nofly3d), grid_size)
    targets_reachable = all(tuple(t) in reachable for t in targets)

    errors = []
    if base_in_nofly:
        errors.append("base is inside no_fly_cells")
    if nofly_coverage > nofly_coverage_limit:
        errors.append(f"no_fly 2D coverage is too high ({nofly_coverage:.2%} > {nofly_coverage_limit:.0%})")
    if solar_rows < min_solar_rows:
        errors.append(f"solar_source_rows_after_filter is too small ({solar_rows} < {min_solar_rows})")
    if not targets_reachable:
        errors.append("at least one target is unreachable from base under no-fly constraints")

    if errors:
        raise RuntimeError("Scenario validation failed: " + "; ".join(errors))


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
    targets, solar_meta = choose_targets_with_region_fallback(
        solar_df,
        max_targets=args.targets,
        region_keyword=args.region_keyword,
        fallback_keywords=args.region_fallback_keywords if args.enable_region_fallback else None,
        min_solar_rows=args.min_solar_rows,
    )

    bbox = solar_meta["solar_bbox"]
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2

    no_fly, restricted, vworld_meta = vworld_cells(args.use_vworld, bbox)
    charging_pads = auto_place_charging_pads(
    base=[0, 0, 0],
    targets=targets,
    grid_size=GRID_SIZE,
    no_fly_cells=no_fly,
    restricted_cells=restricted,
)
    no_fly, restricted, relax_meta = relax_and_validate_airspace(
        no_fly_cells=no_fly,
        restricted_cells=restricted,
        base=[0, 0, 0],
        targets=targets,
        charging_pads=charging_pads,
        grid_size=GRID_SIZE,
        nofly_coverage_limit=args.max_nofly_coverage,
    )

    # Protect start/base area so the episode does not fail immediately.
    protected_cells = {
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1,1,0),
    (1,0,1),
    (0,1,1),
}

    no_fly = [
        cell for cell in no_fly
        if tuple(cell) not in protected_cells
    ]

    restricted = [
        cell for cell in restricted
        if tuple(cell) not in protected_cells
    ]

    essential_cells = set(protected_cells)

    for x, y, z in targets:
        essential_cells.add((x, y, z))
        essential_cells.add((x, y, 0))
        essential_cells.add((x, y, 1))

    for x, y, z in charging_pads:
        essential_cells.add((x, y, z))

    no_fly = [
    cell for cell in no_fly
    if tuple(cell) not in essential_cells
]

    restricted = [
    cell for cell in restricted
    if tuple(cell) not in essential_cells
]

    wind = args.wind
    weather_meta: dict[str, Any] = {"weather_mode": "manual", "manual_wind": wind}
    if args.use_kma:
        wind, weather_meta = fetch_kma_wind_state(center_lat, center_lon)

    scenario = {
        "grid_size": GRID_SIZE,
        "base": [0, 0, 0],
        "targets": targets,
        "charging_pads": charging_pads,
        "no_fly_cells": no_fly,
        "restricted_cells": restricted,
        "initial_wind": wind,
        "wind_states": ["Calm", "EastWind", "NorthWind"],
        "wind_transition": DEFAULT_WIND_TRANSITION,
        "queue_transition": DEFAULT_QUEUE_TRANSITION,
        "max_battery": 18,
        "max_steps": 150,
        "reward": {
            "inspect_new_target": 25,
            "finish_all_and_return_base": 150,
            "step_cost": -1,
            "move_battery_cost": -1,
            "ascend_extra_battery_cost": -1,
            "charge_short_queue_cost": -3,
            "charge_long_queue_cost": -12,
            "restricted_area_cost": -10,
            "no_fly_violation": -80,
            "battery_depletion": -80,
            "invalid_action_cost": -10,
        },
        "metadata": {
            "data_grounding": "solar_csv + VWorld_API + KMA_API_or_manual_wind",
            **solar_meta,
            **vworld_meta,
            **relax_meta,
            **weather_meta,
        },
    }
    validate_scenario(
        scenario=scenario,
        min_solar_rows=args.min_solar_rows,
        nofly_coverage_limit=args.max_nofly_coverage,
    )
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scenario.json from real solar CSV, VWorld API, and KMA API")
    parser.add_argument("--solar-csv", required=True, help="Path to downloaded solar CSV file, e.g., data/solar_api.csv")
    parser.add_argument("--region-keyword", default=None, help="Optional Korean address keyword such as 나주, 광주, 전남")
    parser.add_argument("--use-vworld", action="store_true", help="Use VWorld API to create no-fly/restricted cells")
    parser.add_argument("--use-kma", action="store_true", help="Use KMA API to set initial wind")
    parser.add_argument("--wind", default="Calm", choices=["Calm", "EastWind", "NorthWind"], help="Manual wind if --use-kma is not used")
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--enable-region-fallback", action="store_true", help="Enable fallback region keywords when primary keyword has insufficient rows")
    parser.add_argument("--region-fallback-keywords", default="", help="Comma-separated fallback region keywords (used only with --enable-region-fallback)")
    parser.add_argument("--min-solar-rows", type=int, default=3, help="Minimum filtered solar rows required by validation")
    parser.add_argument("--max-nofly-coverage", type=float, default=0.7, help="Maximum allowed 2D no-fly coverage ratio")
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
