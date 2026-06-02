"""
GPX ターン検出・強化ツール
点X の前後の点 A,B のベアリング差でコーナーを検出する
手動編集機能付き（追加・削除・名前変更）
マップマッチング（Valhalla）・標高補正（国土地理院）対応
交差点名取得（Overpass API / OSM直接）対応
"""

import streamlit as st
import gpxpy
import gpxpy.gpx
import math
import numpy as np
from leaflet_map import render_map
from routing import calc_route_segment
import requests
import urllib.parse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit.components.v1 as components
from rdp import rdp as rdp_simplify
import plotly.graph_objects as go

APP_VERSION = "3.0.0"

st.set_page_config(page_title="gpx-navi エディター", layout="wide", page_icon="🚴")
st.markdown(f'# 🚴 gpx-navi エディター <span style="font-size:0.35em; color:#9ca3af; font-weight:normal; vertical-align:middle;">v{APP_VERSION}</span>', unsafe_allow_html=True)
st.caption("ルートの作成・編集とナビ用ターンポイントの追加ができます")


# ─────────────────────────────────────────────
# ユーティリティ関数
# ─────────────────────────────────────────────

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def angle_diff(a, b):
    """符号付き角度差（-180〜180度）。正=右旋回、負=左旋回"""
    return (b - a + 180) % 360 - 180

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return R * 2 * math.asin(math.sqrt(max(0, a)))

# ─────────────────────────────────────────────
# RoutePoint ヘルパー関数
# ─────────────────────────────────────────────

def make_route_point(lat, lon, ele_org=None, is_acpt=False, wpt=None, changed=True):
    return {
        "lat": float(lat),
        "lon": float(lon),
        "ele_org": ele_org,
        "ele_fix": None,
        "is_acpt": is_acpt,
        "wpt": wpt,
        "changed": changed,
    }

def _deep_copy_rp(rp):
    return [
        {
            "lat": p["lat"],
            "lon": p["lon"],
            "ele_org": p["ele_org"],
            "ele_fix": p["ele_fix"],
            "is_acpt": p["is_acpt"],
            "wpt": dict(p["wpt"]) if p["wpt"] else None,
            "changed": p["changed"],
        }
        for p in rp
    ]

def _prev_boundary(idx, rp):
    """idxより前にあるis_acptまたはwptのうち最大のindexを返す"""
    cands = [i for i, p in enumerate(rp[:idx]) if p["is_acpt"] or p["wpt"] is not None]
    return max(cands) if cands else 0

def _next_boundary(idx, rp):
    """idxより後にあるis_acptまたはwptのうち最小のindexを返す"""
    cands = [i for i, p in enumerate(rp[idx+1:], start=idx+1) if p["is_acpt"] or p["wpt"] is not None]
    return min(cands) if cands else len(rp) - 1

def _contiguous_ranges(indices):
    """連続するインデックスをまとめて (start, end) タプルのリストにする"""
    if not indices:
        return []
    ranges, start, end = [], indices[0], indices[0]
    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append((start, end))
            start = end = i
    ranges.append((start, end))
    return ranges


def _render_elevation_profile(route_points):
    n = len(route_points)
    if n < 2:
        return None

    dists_all = [
        haversine(route_points[i]["lat"], route_points[i]["lon"],
                  route_points[i+1]["lat"], route_points[i+1]["lon"])
        for i in range(n - 1)
    ]
    cum_dist = [0.0]
    for d in dists_all:
        cum_dist.append(cum_dist[-1] + d / 1000)
    total_km = cum_dist[-1]

    org_e_raw = [p["ele_org"] for p in route_points]
    fix_e_raw = [p["ele_fix"] for p in route_points]

    def _prep(elevs):
        if not elevs or len(elevs) < n:
            return None
        return list(elevs[:n])

    org_e = _prep(org_e_raw)
    fix_e = _prep(fix_e_raw)

    all_vals = [v for e in [org_e, fix_e] if e for v in e if v is not None]
    if all_vals:
        ele_min, ele_max = min(all_vals), max(all_vals)
        if ele_min == ele_max:
            ele_max = ele_min + 1
    else:
        ele_min, ele_max = 0, 1

    def _fill(elevs):
        if elevs is None:
            return [ele_min] * n
        return [v if v is not None else ele_min for v in elevs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_dist, y=_fill(org_e), mode="lines",
        line=dict(color="black", width=1.5, dash="dot"),
        name="元データ標高",
    ))
    fig.add_trace(go.Scatter(
        x=cum_dist, y=_fill(fix_e), mode="lines",
        line=dict(color="black", width=1.5),
        name="国土地理院補正標高",
    ))

    if ele_min <= 0 <= ele_max:
        fig.add_hline(y=0, line_color="gray", line_width=1)

    for i, p in enumerate(route_points):
        if p["wpt"] is not None and 0 <= i < len(cum_dist):
            fig.add_vline(x=cum_dist[i], line_color="rgba(100,100,200,0.5)", line_width=1)

    tick_step = 5 if total_km < 50 else 10
    tick_vals = list(range(tick_step, int(total_km) + 1, tick_step))

    fig.update_layout(
        height=150,
        margin=dict(l=50, r=10, t=5, b=35),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=0.99,
            xanchor="right", x=0.99,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.7)",
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=[f"{v}km" for v in tick_vals],
            range=[0, total_km],
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            range=[ele_min, ele_max],
            tickformat=".0f",
            ticksuffix="m",
            tickfont=dict(size=9),
            nticks=3,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def detect_turns(points, min_turn_angle=45, min_dist=100, smooth=1):
    """
    角度法ターン検出
    点 X の前後 smooth 点 (A, B) に対して
        bearing_in  = ベアリング(A → X)
        bearing_out = ベアリング(X → B)
        turn = angle_diff(bearing_in, bearing_out)
    |turn| >= min_turn_angle ならコーナーとみなす。
    """
    n = len(points)
    candidates = []
    for i in range(smooth, n - smooth):
        A = points[i - smooth]
        X = points[i]
        B = points[i + smooth]
        bearing_in  = calculate_bearing(A[0], A[1], X[0], X[1])
        bearing_out = calculate_bearing(X[0], X[1], B[0], B[1])
        turn = angle_diff(bearing_in, bearing_out)
        if abs(turn) >= min_turn_angle:
            candidates.append({"lat": X[0], "lon": X[1], "delta": turn, "index": i})

    if not candidates:
        return []

    candidates_sorted = sorted(candidates, key=lambda x: abs(x["delta"]), reverse=True)
    used = set()
    turns = []
    for c in candidates_sorted:
        if c["index"] in used:
            continue
        turns.append(c)
        for c2 in candidates:
            if haversine(c["lat"], c["lon"], c2["lat"], c2["lon"]) < min_dist:
                used.add(c2["index"])

    turns.sort(key=lambda x: x["index"])
    return turns

def turn_label(delta):
    if delta >= 60:    return "右折",     "⇒", "#e74c3c"
    elif delta >= 25:  return "やや右",   "↗", "#e67e22"
    elif delta <= -60: return "左折",     "⇐", "#2980b9"
    elif delta <= -25: return "やや左",   "↖", "#8e44ad"
    else:              return "直進維持", "↑", "#7f8c8d"

def nearest_trkpt_index(lat, lon, points):
    """クリック位置に最も近いトラックポイントのインデックスを返す"""
    return min(range(len(points)),
               key=lambda i: haversine(lat, lon, points[i][0], points[i][1]))

def with_name(wpt_info, intersection_name=None):
    """nameフィールドを持つWptInfo辞書を返す。
    intersection_name がある場合は「△△交差点を右折」形式、
    ない場合は従来通り「右折」形式。
    """
    d = dict(wpt_info)
    dir_label = turn_label(wpt_info["delta"])[0]
    if intersection_name:
        d["name"] = f"{intersection_name}を{dir_label}"
    else:
        d["name"] = dir_label
    return d

def wpt_style(wpt_info):
    """(arrow, hex_color) を返す"""
    delta = wpt_info.get("delta") if wpt_info else None
    if delta is not None:
        _, arrow, color = turn_label(delta)
    else:
        arrow, color = "📍", "#27ae60"
    return arrow, color

# ─────────────────────────────────────────────
# 交差点名取得（Overpass API / OSM直接）
# ─────────────────────────────────────────────

_OVERPASS_URLS = [
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

def fetch_intersection_names(turns, radius=20):
    """
    Overpass API でターンポイント付近のOSMノードから交差点名を取得する。
    nameタグを持つノードを半径radius(m)以内で検索し、最近傍のものを採用する。
    戻り値: {trkpt_index: intersection_name} の辞書。名前が取れなかった点はキーなし。
    """
    if not turns:
        return {}

    n = len(turns)

    _HW = '"^(traffic_signals|crossing|give_way|stop|mini_roundabout|motorway_junction)$"'
    union_parts = "".join(
        f'node(around:{radius},{t["lat"]},{t["lon"]})[name][highway~{_HW}];'
        for t in turns
    )
    query = f"[out:json][timeout:25];({union_parts});out body;"

    elements = None
    with st.spinner(f"交差点名を取得中…（{n} 件）"):
        for url in _OVERPASS_URLS:
            try:
                resp = requests.post(
                    url,
                    data="data=" + urllib.parse.quote(query),
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                        "User-Agent": "GPXTurnDetector/1.0",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                elements = resp.json().get("elements", [])
                break
            except Exception:
                pass

    if elements is None:
        return {}

    result = {}
    for i, t in enumerate(turns):
        nearest_name = None
        nearest_dist = float("inf")
        for node in elements:
            d = haversine(t["lat"], t["lon"], node["lat"], node["lon"])
            if d < nearest_dist:
                nearest_dist = d
                nearest_name = node.get("tags", {}).get("name")
        if nearest_name and nearest_dist <= radius:
            result[t["index"]] = nearest_name

    return result

def fetch_spot_name(lat, lon, radius=20):
    """クリック位置付近のPOI名を取得する（交差点名が見つからなかった場合のフォールバック）。"""
    tags = ["tourism", "amenity", "leisure", "historic", "natural", "shop"]
    union_parts = "".join(
        f'node(around:{radius},{lat},{lon})[name]["{tag}"];'
        for tag in tags
    )
    query = f"[out:json][timeout:10];({union_parts});out body;"
    for url in _OVERPASS_URLS:
        try:
            resp = requests.post(
                url,
                data="data=" + urllib.parse.quote(query),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "User-Agent": "GPXTurnDetector/1.0",
                },
                timeout=15,
            )
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            if not elements:
                return None
            nearest = min(elements, key=lambda e: haversine(lat, lon, e["lat"], e["lon"]))
            return nearest.get("tags", {}).get("name")
        except Exception:
            pass
    return None

# ─────────────────────────────────────────────
# マップマッチング（Valhalla / OSM公式インスタンス・自転車固定）
# ─────────────────────────────────────────────

_VALHALLA_URL = "https://valhalla1.openstreetmap.de/trace_attributes"

def _valhalla_match_chunk(chunk, search_radius=50):
    """Valhalla trace_attributes で1チャンクをスナップ（bicycle固定）"""
    resp = requests.post(
        _VALHALLA_URL,
        json={
            "shape":         [{"lat": lat, "lon": lon} for lat, lon in chunk],
            "costing":       "bicycle",
            "shape_match":   "map_snap",
            "search_radius": search_radius,
            "filters": {
                "attributes": ["matched.point", "matched.type"],
                "action":     "include",
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ─────────────────────────────────────────────
# 標高補正（国土地理院 / OpenTopoData / Open-Meteo）
# フォールバック順: 国土地理院(5s) → OpenTopoData(10s) → Open-Meteo → 元データ保持
# ─────────────────────────────────────────────

def _is_in_japan(lat, lon):
    return 24.0 <= lat <= 46.0 and 122.0 <= lon <= 154.0


def _cumulative_distances(points):
    cum = [0.0]
    for i in range(1, len(points)):
        cum.append(cum[-1] + haversine(
            points[i - 1][0], points[i - 1][1],
            points[i][0], points[i][1],
        ))
    return cum

def _elevation_grades(points, elevations, cum_dists=None):
    if cum_dists is None:
        cum_dists = _cumulative_distances(points)

    grades = []
    for i in range(len(points) - 1):
        if elevations[i] is None or elevations[i + 1] is None:
            grades.append(None)
            continue
        dist = cum_dists[i + 1] - cum_dists[i]
        if dist <= 0:
            grades.append(None)
            continue
        grades.append((elevations[i + 1] - elevations[i]) / dist * 100)
    return grades

def _local_median_elevation(i, cum_dists, elevations, window_m):
    lo = cum_dists[i] - window_m
    hi = cum_dists[i] + window_m
    vals = [
        e for j, e in enumerate(elevations)
        if e is not None and lo <= cum_dists[j] <= hi
    ]
    return float(np.median(vals)) if vals else None

def _cluster_segments(seg_indexes, cum_dists, cluster_gap_m):
    if not seg_indexes:
        return []

    clusters = []
    cur = {"start_seg": seg_indexes[0], "end_seg": seg_indexes[0]}
    for seg_idx in seg_indexes[1:]:
        gap_m = cum_dists[seg_idx] - cum_dists[cur["end_seg"] + 1]
        if gap_m <= cluster_gap_m:
            cur["end_seg"] = seg_idx
        else:
            clusters.append(cur)
            cur = {"start_seg": seg_idx, "end_seg": seg_idx}
    clusters.append(cur)
    return clusters

def clean_elevation_spikes(points, elevations, bad_grade_threshold=15.0, cluster_gap_m=250.0):
    """
    標高API由来の局所スパイクを前後アンカーの線形補間で除去する。
    戻り値: (cleaned_elevations, stats)
    """
    n = len(points)
    if n < 4 or not elevations or len(elevations) != n:
        return elevations, {"clusters": 0, "points": 0, "max_grade_before": 0.0, "max_grade_after": 0.0}

    BAD_GRADE_THRESHOLD = bad_grade_threshold
    HARD_SPIKE_THRESHOLD = 35.0
    NEAR_BAD_THRESHOLD = 15.0
    MIN_ELEVATION_JUMP_M = 2.0
    NEAR_ELEVATION_JUMP_M = 3.0
    SHORT_SEG_M = 10.0  # この距離以下のセグメントはdz条件を免除（密なGPS点でのSRTMグリッド境界対策）
    CLUSTER_GAP_M = cluster_gap_m
    MERGE_GAP_M = 50.0
    MAX_ANCHOR_SEARCH_M = 600.0
    ANCHOR_GRADE_LIMIT = 12.0
    BOUNDARY_GRADE_LIMIT = 13.0
    MEDIAN_WINDOW_M = 150.0
    ANCHOR_MEDIAN_DEV_M = 5.0
    MAX_ANCHOR_GRADE = 15.0
    MIN_CORRECTION_GRADE_PCT = 1.0  # 隣接最短セグメントの1%未満の補正は無視（短距離セグメント対策）

    cleaned = list(elevations)
    cum_dists = _cumulative_distances(points)
    grades = _elevation_grades(points, cleaned, cum_dists)
    max_grade_before = max((abs(g) for g in grades if g is not None), default=0.0)

    bad_segments = []
    for i, grade in enumerate(grades):
        if grade is None or cleaned[i] is None or cleaned[i + 1] is None:
            continue
        dz = cleaned[i + 1] - cleaned[i]
        short_seg = (cum_dists[i + 1] - cum_dists[i]) < SHORT_SEG_M
        if (
            abs(grade) >= BAD_GRADE_THRESHOLD and (short_seg or abs(dz) >= MIN_ELEVATION_JUMP_M)
        ) or (
            abs(grade) >= HARD_SPIKE_THRESHOLD and (short_seg or abs(dz) >= NEAR_ELEVATION_JUMP_M)
        ):
            bad_segments.append(i)

    if not bad_segments:
        return cleaned, {"clusters": 0, "points": 0, "max_grade_before": max_grade_before, "max_grade_after": max_grade_before}

    near_segments = set(bad_segments)
    for bad_idx in bad_segments:
        center = (cum_dists[bad_idx] + cum_dists[bad_idx + 1]) / 2
        for i, grade in enumerate(grades):
            if grade is None or cleaned[i] is None or cleaned[i + 1] is None:
                continue
            seg_center = (cum_dists[i] + cum_dists[i + 1]) / 2
            dz = cleaned[i + 1] - cleaned[i]
            short_seg = (cum_dists[i + 1] - cum_dists[i]) < SHORT_SEG_M
            if (
                abs(seg_center - center) <= CLUSTER_GAP_M
                and abs(grade) >= NEAR_BAD_THRESHOLD
                and (short_seg or abs(dz) >= NEAR_ELEVATION_JUMP_M)
            ):
                near_segments.add(i)

    clusters = _cluster_segments(sorted(near_segments), cum_dists, CLUSTER_GAP_M)

    def is_anchor_candidate(i):
        if i <= 0 or i >= n - 1 or cleaned[i] is None:
            return False
        prev_g = grades[i - 1]
        next_g = grades[i]
        if prev_g is None or next_g is None:
            return False
        if abs(prev_g) > ANCHOR_GRADE_LIMIT or abs(next_g) > ANCHOR_GRADE_LIMIT:
            return False
        local_med = _local_median_elevation(i, cum_dists, cleaned, MEDIAN_WINDOW_M)
        return local_med is not None and abs(cleaned[i] - local_med) <= ANCHOR_MEDIAN_DEV_M

    def find_anchor(start_i, direction):
        start_dist = cum_dists[start_i]
        i = start_i
        stable_run = []
        while 0 < i < n - 1 and abs(cum_dists[i] - start_dist) <= MAX_ANCHOR_SEARCH_M:
            if is_anchor_candidate(i):
                stable_run.append(i)
                if len(stable_run) >= 2:
                    return stable_run[0]
            else:
                stable_run = []
            i += direction
        return None

    def is_left_boundary_anchor(i):
        return (
            0 < i < n - 1
            and cleaned[i] is not None
            and grades[i - 1] is not None
            and abs(grades[i - 1]) <= BOUNDARY_GRADE_LIMIT
        )

    def is_right_boundary_anchor(i):
        return (
            0 < i < n - 1
            and cleaned[i] is not None
            and grades[i] is not None
            and abs(grades[i]) <= BOUNDARY_GRADE_LIMIT
        )

    repair_ranges = []
    for cluster in clusters:
        start_pt = cluster["start_seg"]
        end_pt = cluster["end_seg"] + 1
        left_anchor = (start_pt if (is_left_boundary_anchor(start_pt) and is_anchor_candidate(start_pt))
                       else find_anchor(start_pt - 1, -1))
        right_anchor = (end_pt if (is_right_boundary_anchor(end_pt) and is_anchor_candidate(end_pt))
                        else find_anchor(end_pt + 1, 1))
        if left_anchor is None or right_anchor is None or left_anchor >= right_anchor:
            continue
        dist_m = cum_dists[right_anchor] - cum_dists[left_anchor]
        if dist_m <= 0 or cleaned[left_anchor] is None or cleaned[right_anchor] is None:
            continue
        net_grade = (cleaned[right_anchor] - cleaned[left_anchor]) / dist_m * 100
        if abs(net_grade) > MAX_ANCHOR_GRADE:
            continue
        repair_ranges.append({
            "left": left_anchor,
            "right": right_anchor,
            "bad_start": start_pt,
            "bad_end": end_pt,
        })

    if not repair_ranges:
        return cleaned, {"clusters": 0, "points": 0, "max_grade_before": max_grade_before, "max_grade_after": max_grade_before}

    repair_ranges.sort(key=lambda r: r["left"])
    merged = [repair_ranges[0]]
    for r in repair_ranges[1:]:
        prev = merged[-1]
        gap_m = cum_dists[r["left"]] - cum_dists[prev["right"]]
        if r["left"] <= prev["right"] or gap_m <= MERGE_GAP_M:
            prev["right"] = max(prev["right"], r["right"])
            prev["bad_start"] = min(prev["bad_start"], r["bad_start"])
            prev["bad_end"] = max(prev["bad_end"], r["bad_end"])
        else:
            merged.append(r)

    corrected_points = set()
    for r in merged:
        left = r["left"]
        right = r["right"]
        if right - left < 2:
            continue
        dist_m = cum_dists[right] - cum_dists[left]
        if dist_m <= 0 or cleaned[left] is None or cleaned[right] is None:
            continue
        net_grade = (cleaned[right] - cleaned[left]) / dist_m * 100
        if abs(net_grade) > MAX_ANCHOR_GRADE:
            continue
        for i in range(left + 1, right):
            if cleaned[i] is None:
                continue
            ratio = (cum_dists[i] - cum_dists[left]) / dist_m
            new_ele = cleaned[left] + (cleaned[right] - cleaned[left]) * ratio
            min_adj = min(cum_dists[i] - cum_dists[i - 1], cum_dists[i + 1] - cum_dists[i])
            if abs(cleaned[i] - new_ele) >= MIN_CORRECTION_GRADE_PCT / 100 * min_adj:
                cleaned[i] = round(new_ele, 1)
                corrected_points.add(i)

    grades_after = _elevation_grades(points, cleaned, cum_dists)
    max_grade_after = max((abs(g) for g in grades_after if g is not None), default=0.0)
    return cleaned, {
        "clusters": len(merged),
        "points": len(corrected_points),
        "max_grade_before": max_grade_before,
        "max_grade_after": max_grade_after,
    }

def compute_grade_stats(points, elevations):
    """勾配リストから上り最大・下り最大（符号付き）を返す。データなしは None。"""
    grades = _elevation_grades(points, elevations)
    valid = [g for g in grades if g is not None]
    if not valid:
        return None
    return {
        "max": max((g for g in valid if g > 0), default=0.0),
        "min": min((g for g in valid if g < 0), default=0.0),
    }

# ─────────────────────────────────────────────
# GPX ビルダー（マッチング・標高補正対応）
# ─────────────────────────────────────────────

def build_enhanced_gpx(gpx_content_str, route_points, elev_choice="org"):
    coords = [(p["lat"], p["lon"]) for p in route_points]
    elevs  = [p["ele_fix"] if elev_choice == "fix" else p["ele_org"] for p in route_points]

    if gpx_content_str:
        enhanced = gpxpy.parse(gpx_content_str)
    else:
        enhanced = gpxpy.gpx.GPX()
        enhanced.tracks.append(gpxpy.gpx.GPXTrack())

    # trkpt をroute_pointsの点数で完全に置き換える
    # （RDP間引き・マップマッチングで点数が変わっている場合に対応）
    _new_seg = gpxpy.gpx.GPXTrackSegment()
    for i, (lat, lon) in enumerate(coords):
        pt = gpxpy.gpx.GPXTrackPoint(lat, lon)
        if i < len(elevs) and elevs[i] is not None:
            pt.elevation = elevs[i]
        _new_seg.points.append(pt)
    if enhanced.tracks:
        enhanced.tracks[0].segments = [_new_seg]

    # ターンポイントを再構築
    enhanced.waypoints = []
    for p in route_points:
        if p["wpt"] is None:
            continue
        winfo = p["wpt"]
        name  = winfo.get("name") or ""
        delta = winfo.get("delta")
        if not name and delta is not None:
            name = turn_label(delta)[0]
        desc  = f"bearing_change:{delta:.1f}" if delta is not None else "manually added"
        enhanced.waypoints.append(gpxpy.gpx.GPXWaypoint(
            latitude=p["lat"], longitude=p["lon"], name=name, description=desc,
        ))
    return enhanced.to_xml()

# ─────────────────────────────────────────────
# 標高デフォルト選択
# ─────────────────────────────────────────────

def _set_default_elev_choice():
    rp = st.session_state.get("route_points", [])
    _org_ok = bool(rp) and all(p["ele_org"] is not None for p in rp)
    _fix_ok = bool(rp) and all(p["ele_fix"] is not None for p in rp)
    if _org_ok and _fix_ok:
        _go = st.session_state.get("_grade_org")
        _gf = st.session_state.get("_grade_fix")
        _so = (_go["max"] + abs(_go["min"])) if _go else float("inf")
        _sf = (_gf["max"] + abs(_gf["min"])) if _gf else float("inf")
        st.session_state["_elev_choice"] = "fix" if _sf < _so else "org"
    elif _fix_ok:
        st.session_state["_elev_choice"] = "fix"
    else:
        st.session_state["_elev_choice"] = "org"

# ─────────────────────────────────────────────
# ファイルアップロード / 新規ルートモード
# ─────────────────────────────────────────────

_STATE_KEYS = [
    "route_points",
    "_map_center", "_map_zoom",
    "_matched_points", "_mm_base_points", "_mm_kept_indices", "_mm_status", "_mm_n_snapped", "_mm_error",
    "_mm_chunk_idx", "_mm_matched_partial", "_mm_n_snapped_partial", "_mm_errors_partial",
    "_proc_status",
    "_grade_org", "_grade_fix", "_elev_choice",
    "_elev_batch_idx", "_elev_partial", "_elev_cancel_requested",
    "_iname_status",  "_iname_n_found",
    "_focus_wpt_idx",
    "_map_event_ts",
    "_undo_state",
    "route_modified",
    "_rdp_done",
    "_save_dialog",
    "_confirm_back",
    "_raw_gpx",
    "_gpx_filename",
]

# 編集モード判定：ファイル読込済み または 新規ルートモード
_in_editing = (st.session_state.get("_file_key") is not None
               or st.session_state.get("_new_route_mode"))

if not _in_editing:
    # ── スタート画面（ファイル未読込・非新規モードのときのみ表示）──
    st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div:first-child {
    background: white;
    border-radius: 12px;
    padding: 24px;
    border: 2px solid #e5e7eb;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    min-height: 340px;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) > div:first-child {
    border-color: #dbeafe;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) > div:first-child {
    border-color: #d1fae5;
}
.feat-list      { list-style: none; padding: 0; margin: 0 0 20px 0; }
.feat-list li   { font-size: 13px; color: #374151; padding: 5px 0;
                  border-bottom: 1px solid #f3f4f6; }
.feat-list li:last-child { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

    _col_gpx, _col_new = st.columns(2, gap="large")

    with _col_gpx:
        st.markdown("#### 📂 GPXファイルを読み込む")
        st.markdown("Stravaや他のアプリで作成・記録したGPXにターンポイントを追加して強化します")
        st.radio(
            "データの種類",
            ["🗺️ ルートデータ（Stravaルート作成など）",
             "🏃 実走行データ（GPSで記録した走行ログ）"],
            key="_data_type_radio",
            help="実走行データはマップマッチング・間引きを自動実行します",
        )
        _uploaded_start = st.file_uploader("GPXファイルをアップロード", type=["gpx", "xml"],
                                            label_visibility="collapsed")

    with _col_new:
        st.markdown("#### ✏️ 新規ルートを作成する")
        st.markdown("地図上をクリックしてゼロから自転車ルートを作成し、ナビ情報を付けてGPXに出力します")
        st.markdown("""
<ul class="feat-list">
  <li>📍 地図クリックでルートを延伸</li>
  <li>⚓ アンカーポイントのドラッグでルート編集</li>
  <li>🔍 交差点ターンポイントを自動検出</li>
  <li>⛰️ 国土地理院による標高補正（日本国内）</li>
</ul>
""", unsafe_allow_html=True)
        if st.button("🗺️ 新規ルートを作成する", use_container_width=True, type="primary"):
            for k in _STATE_KEYS:
                st.session_state.pop(k, None)
            st.session_state.pop("_file_key", None)
            st.session_state["_new_route_mode"] = True
            st.rerun()

    if _uploaded_start is not None:
        _is_actual_ride_s = "実走行データ" in st.session_state.get("_data_type_radio", "")
        _fk = f"{_uploaded_start.name}_{_is_actual_ride_s}"
        if st.session_state.get("_file_key") != _fk:
            for k in _STATE_KEYS:
                st.session_state.pop(k, None)
            st.session_state["_file_key"] = _fk
        st.session_state["_raw_gpx"]       = _uploaded_start.read().decode("utf-8")
        st.session_state["_gpx_filename"]  = _uploaded_start.name
        st.rerun()

    st.stop()

# ── 編集モード（以降は両パス共通） ────────────────
_data_type = st.session_state.get("_data_type_radio",
                                   "🗺️ ルートデータ（Stravaルート作成など）")
_is_actual_ride = "実走行データ" in _data_type

raw_content = st.session_state.get("_raw_gpx")
points = []
gpx_parsed = None
_has_wpts = False

if raw_content:
    try:
        gpx_parsed = gpxpy.parse(raw_content)
    except Exception as e:
        st.error(f"GPXの解析に失敗しました: {e}")
        st.stop()

    for track in gpx_parsed.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append((pt.latitude, pt.longitude))

    if len(points) < 6:
        st.error("トラックポイントが少なすぎます。")
        st.stop()

    _has_wpts = len(gpx_parsed.waypoints) > 0

_skip_map_center_save = st.session_state.pop("_skip_map_center_save", False)

# ─────────────────────────────────────────────
# 自動処理（マップマッチング）
# ─────────────────────────────────────────────

if st.session_state.get("_new_route_mode"):
    st.session_state.setdefault("_mm_status", "スキップ")
    st.session_state.setdefault("_matched_points", [])
    st.session_state.setdefault("_proc_status", "done")

@st.dialog("🗺️ マップマッチング中", width="large")
def _mm_progress_dialog():
    _mm_pts   = st.session_state.get("_mm_base_points", [])
    _MM_CHUNK = 50
    _ci       = st.session_state.get("_mm_chunk_idx", 0)
    _n_chunks = math.ceil(len(_mm_pts) / _MM_CHUNK) if _mm_pts else 1
    _cancelled = st.session_state.pop("_mm_cancel_requested", False)

    _prog_area = st.empty()
    if _cancelled:
        _prog_area.progress(_ci / _n_chunks, text="⏱️ キャンセル待ち中…")
    else:
        _prog_area.progress((_ci + 1) / _n_chunks,
                            text=f"🗺️ マップマッチング中… {_ci + 1}/{_n_chunks} チャンク")
    if st.button("⏹ キャンセル", key="mm_cancel_btn"):
        st.session_state["_mm_cancel_requested"] = True
        st.rerun()

    if _cancelled:
        _errs = st.session_state.get("_mm_errors_partial", [])
        st.session_state["_matched_points"] = st.session_state.get("_mm_matched_partial", list(_mm_pts))
        st.session_state["_mm_n_snapped"]   = st.session_state.get("_mm_n_snapped_partial", 0)
        st.session_state["_mm_error"]       = "キャンセルされました" + ("; " + "; ".join(_errs) if _errs else "")
        st.session_state["_mm_status"]      = "キャンセル"
        for _k in ["_mm_chunk_idx", "_mm_matched_partial", "_mm_n_snapped_partial", "_mm_errors_partial"]:
            st.session_state.pop(_k, None)
        _prog_area.progress(1.0, text="✅ キャンセルしました")
        st.rerun()
    else:
        _s = _ci * _MM_CHUNK
        _e = min(_s + _MM_CHUNK, len(_mm_pts))
        _auto_cancel = False
        try:
            _mp_list = st.session_state["_mm_matched_partial"]
            _data = _valhalla_match_chunk(_mm_pts[_s:_e])
            for _j, _mp in enumerate(_data.get("matched_points", [])):
                if _mp.get("type") in ("matched", "interpolated") and _s + _j < len(_mp_list):
                    _mp_list[_s + _j] = (_mp["lat"], _mp["lon"])
                    st.session_state["_mm_n_snapped_partial"] += 1
        except requests.exceptions.Timeout:
            if _ci == 0:
                _auto_cancel = True
            else:
                st.session_state["_mm_errors_partial"].append(f"chunk {_ci}: timeout")
        except Exception as _ex:
            st.session_state["_mm_errors_partial"].append(f"chunk {_ci}: {_ex}")

        if _auto_cancel:
            st.session_state["_matched_points"] = list(_mm_pts)
            st.session_state["_mm_n_snapped"]   = 0
            st.session_state["_mm_error"]       = "1チャンク目タイムアウトにより自動キャンセル"
            st.session_state["_mm_status"]      = "キャンセル"
            for _k in ["_mm_chunk_idx", "_mm_matched_partial", "_mm_n_snapped_partial", "_mm_errors_partial"]:
                st.session_state.pop(_k, None)
            _prog_area.progress(1.0, text="✅ タイムアウトによりキャンセルしました")
            st.rerun()
        elif _ci + 1 >= _n_chunks:
            _errs = st.session_state.get("_mm_errors_partial", [])
            _n_sn = st.session_state.get("_mm_n_snapped_partial", 0)
            st.session_state["_matched_points"] = st.session_state.get("_mm_matched_partial", list(_mm_pts))
            st.session_state["_mm_n_snapped"]   = _n_sn
            st.session_state["_mm_error"]       = "; ".join(_errs) if _errs else None
            st.session_state["_mm_status"]      = "完了" if _n_sn > 0 else "エラー"
            for _k in ["_mm_chunk_idx", "_mm_matched_partial", "_mm_n_snapped_partial", "_mm_errors_partial"]:
                st.session_state.pop(_k, None)
            st.rerun()
        else:
            st.session_state["_mm_chunk_idx"] = _ci + 1
            st.rerun()


if st.session_state.get("_mm_status") is None:
    if not _is_actual_ride:
        # ルートデータ：マップマッチングスキップ
        st.session_state["_matched_points"] = list(points)
        st.session_state["_mm_status"]      = "スキップ"
    else:
        # 実走行データ：RDP先行間引き → マップマッチング開始
        # epsilon は度単位（1度≈111km）。5m相当 = 5/111000 ≈ 0.000045
        _rdp_mask    = rdp_simplify([[p[0], p[1]] for p in points], epsilon=0.00005, return_mask=True)
        _mm_pts_init = [tuple(points[i]) for i, keep in enumerate(_rdp_mask) if keep]
        _mm_kept_idx = [i for i, keep in enumerate(_rdp_mask) if keep]
        st.session_state["_mm_base_points"]         = _mm_pts_init
        st.session_state["_mm_kept_indices"]        = _mm_kept_idx
        st.session_state["_rdp_done"]               = True
        st.session_state["_mm_status"]              = "running"
        st.session_state["_mm_chunk_idx"]           = 0
        st.session_state["_mm_matched_partial"]     = list(_mm_pts_init)
        st.session_state["_mm_n_snapped_partial"]   = 0
        st.session_state["_mm_errors_partial"]      = []

if st.session_state.get("_mm_status") == "running":
    _mm_progress_dialog()

# ─────────────────────────────────────────────
# route_points 初期化（初回のみ）
# ─────────────────────────────────────────────

if st.session_state.get("_proc_status") is None and st.session_state.get("_mm_status") != "running":
    _base_coords = list(st.session_state.get("_matched_points") or points or [])
    rp = [make_route_point(lat, lon, changed=not _has_wpts) for lat, lon in _base_coords]
    if rp:
        rp[0]["is_acpt"] = True
        rp[-1]["is_acpt"] = True
        # org 標高処理
        if gpx_parsed is not None:
            _all_orig_elevs = [p.elevation for tr in gpx_parsed.tracks
                               for seg in tr.segments for p in seg.points]
            # RDP間引き済みの場合は元インデックスで標高をマッピング
            _kept_idx = st.session_state.get("_mm_kept_indices")
            if _kept_idx and len(_kept_idx) == len(_base_coords):
                _orig_elevs = [_all_orig_elevs[i] if i < len(_all_orig_elevs) else None
                               for i in _kept_idx]
            else:
                _orig_elevs = _all_orig_elevs
            if not all(e is None for e in _orig_elevs):
                _org_cleaned, _ = clean_elevation_spikes(_base_coords, _orig_elevs)
                for i, v in enumerate(_org_cleaned):
                    if i < len(rp):
                        rp[i]["ele_org"] = v
                st.session_state["_grade_org"] = compute_grade_stats(_base_coords, _org_cleaned)
            else:
                st.session_state["_grade_org"] = None
        else:
            st.session_state["_grade_org"] = None
        # wpt 初期化
        if _has_wpts and gpx_parsed:
            for wpt in gpx_parsed.waypoints:
                delta = None
                desc  = wpt.description or ""
                if desc.startswith("bearing_change:"):
                    try:
                        delta = float(desc.split(":")[1])
                    except ValueError:
                        pass
                idx = nearest_trkpt_index(wpt.latitude, wpt.longitude, _base_coords)
                rp[idx]["wpt"] = {"name": wpt.name or "ターンポイント", "delta": delta}
            st.session_state["_iname_status"] = "スキップ"
        else:
            _raw_turns = detect_turns(_base_coords, min_turn_angle=45, min_dist=100, smooth=1)
            _inames = fetch_intersection_names(_raw_turns)
            st.session_state["_iname_status"]  = "完了"
            st.session_state["_iname_n_found"] = len(_inames)
            for t in _raw_turns:
                idx = t["index"]
                rp[idx]["wpt"] = with_name({"name": "", "delta": t["delta"]}, _inames.get(idx))
            if rp:
                rp[0]["wpt"]  = {"name": "スタート", "delta": None}
                rp[-1]["wpt"] = {"name": "目的地",   "delta": None}
        for p in rp:
            p["changed"] = False
    st.session_state["route_points"] = rp
    st.session_state["_grade_fix"]   = None
    st.session_state["_proc_status"] = "done"
    _set_default_elev_choice()

# 新規ルートモード（最初のクリック前）
if st.session_state.get("_new_route_mode") and "route_points" not in st.session_state:
    st.session_state["route_points"] = []
    st.session_state["_grade_org"]   = None
    st.session_state["_grade_fix"]   = None
    st.session_state["_proc_status"] = "done"

route_points = st.session_state.get("route_points", [])

if _has_wpts:
    st.info("📂 GPX内のターンポイントを読み込みました。マップマッチング・標高補正はスキップされています。")

# ─── ルート情報 ───────────────────────────────
coords = [(p["lat"], p["lon"]) for p in route_points]
dists_all = [haversine(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
             for i in range(len(coords) - 1)]
avg_spacing   = np.mean(dists_all) if dists_all else 0.0
total_dist_km = sum(dists_all) / 1000
route_name    = next((t.name for t in gpx_parsed.tracks if t.name), "（名称なし）") if gpx_parsed else "新規ルート"

# ── 「編集を破棄して戻る」ボタン ──────────────────
if st.session_state.get("_confirm_back"):
    st.warning("編集中の内容は破棄されます。スタート画面に戻りますか？")
    _cb1, _cb2, _ = st.columns([1, 1, 4])
    with _cb1:
        if st.button("🏠 スタート画面に戻る", type="primary"):
            for k in _STATE_KEYS:
                st.session_state.pop(k, None)
            st.session_state.pop("_file_key", None)
            st.session_state.pop("_new_route_mode", None)
            st.session_state.pop("_confirm_back", None)
            st.rerun()
    with _cb2:
        if st.button("✏️ 編集を続ける"):
            st.session_state.pop("_confirm_back", None)
            st.rerun()
else:
    if st.button("↩ 編集を破棄して戻る", help="スタート画面に戻ります"):
        st.session_state["_confirm_back"] = True
        st.rerun()

_elev_key = "ele_fix" if st.session_state.get("_elev_choice") == "fix" else "ele_org"
_elevs_for_gain = [p[_elev_key] for p in route_points]
if any(e is not None for e in _elevs_for_gain):
    _gain = sum(
        max(0, _elevs_for_gain[i+1] - _elevs_for_gain[i])
        for i in range(len(_elevs_for_gain) - 1)
        if _elevs_for_gain[i] is not None and _elevs_for_gain[i+1] is not None
    )
    _gain_str = f"{_gain:.0f} m"
else:
    _gain_str = "--- m"

c1, c2, c3 = st.columns(3)
c1.metric("ルート名", route_name)
c2.metric("総距離", f"{total_dist_km:.1f} km")
c3.metric("獲得標高", _gain_str)


# ─────────────────────────────────────────────
# 地図 + リストパネル
# ─────────────────────────────────────────────

col_map, col_list = st.columns([2, 1])
with col_map:
    st.subheader("🗺️ 地図プレビュー")
    _saved_center = st.session_state.get("_map_center")
    if _saved_center:
        _map_center = _saved_center
        _force_center = _skip_map_center_save
    elif route_points:
        _q = len(route_points) // 4
        _map_center = {
            "lat": route_points[_q]["lat"],
            "lng": route_points[_q]["lon"],
        }
        _force_center = True
    else:
        _map_center = {"lat": 35.681, "lng": 139.767}  # デフォルト（東京）
        _force_center = True
    _map_zoom = st.session_state.get("_map_zoom", 13)

    _acpts_for_map = [
        {"lat": p["lat"], "lng": p["lon"], "trkpt_idx": i}
        for i, p in enumerate(route_points) if p["is_acpt"]
    ]
    _wpts_for_map = [
        {"lat": p["lat"], "lng": p["lon"], "trkpt_idx": i,
         "name": p["wpt"]["name"], "color": wpt_style(p["wpt"])[1]}
        for i, p in enumerate(route_points) if p["wpt"] is not None
    ]
    _map_event = render_map(
        data={
            "trkpts": coords,
            "acpts": _acpts_for_map,
            "wpts": _wpts_for_map,
            "center": _map_center,
            "zoom": _map_zoom,
            "force_center": _force_center,
        },
        height=520,
        key="gpx_map",
    )

    _elev_fig = _render_elevation_profile(route_points)
    if _elev_fig is not None:
        st.plotly_chart(_elev_fig, width="stretch", config={"displayModeBar": False})

# ── イベント処理 ─────────────────────────────────
if isinstance(_map_event, dict) and _map_event.get("ts", 0) != st.session_state.get("_map_event_ts", 0):
    st.session_state["_map_event_ts"] = _map_event["ts"]
    if "center" in _map_event:
        st.session_state["_map_center"] = _map_event["center"]
    if "zoom" in _map_event:
        st.session_state["_map_zoom"] = _map_event["zoom"]

    _evt_type = _map_event.get("type")

    if _evt_type == "click_empty" or (
        _evt_type == "dialog_result" and _map_event.get("action") == "extend"
    ):
        _clat, _clng = _map_event["lat"], _map_event["lng"]
        rp = list(st.session_state.get("route_points", []))
        _all_acpts = [(i, p) for i, p in enumerate(rp) if p["is_acpt"]]

        if not _all_acpts:
            # 1点目クリック: スタートwptを作成してルート計算なし
            new_pt = make_route_point(
                _clat, _clng,
                is_acpt=True,
                wpt={"name": "スタート", "delta": None},
                changed=False,
            )
            st.session_state["route_points"] = [new_pt]
        else:
            st.session_state["_undo_state"] = {"route_points": _deep_copy_rp(rp)}
            _prev_pt = (rp[-1]["lat"], rp[-1]["lon"])
            _seg = calc_route_segment([_prev_pt, (_clat, _clng)])
            # 旧目的地wptを外す（そのtrkptはルート内点として残る）
            if rp and rp[-1]["wpt"] is not None and rp[-1]["wpt"].get("name") == "目的地":
                rp[-1]["wpt"] = None
            _seg_tail = _seg[1:]
            new_seg_pts = [
                make_route_point(lat, lon, is_acpt=(j == len(_seg_tail) - 1), changed=True)
                for j, (lat, lon) in enumerate(_seg_tail)
            ]
            if new_seg_pts:
                new_seg_pts[-1]["wpt"]    = {"name": "目的地", "delta": None}
                new_seg_pts[-1]["changed"] = False
            st.session_state["route_points"] = rp + new_seg_pts
        st.session_state["route_modified"] = True
        st.session_state["_skip_map_center_save"] = True
        st.rerun()

    elif _evt_type == "acpt_drag_end":
        _ai      = _map_event.get("acpt_idx")
        _new_lat = _map_event["lat"]
        _new_lng = _map_event["lng"]
        rp = list(st.session_state.get("route_points", []))
        _all_acpts = [(i, p) for i, p in enumerate(rp) if p["is_acpt"]]
        if _ai is not None and 0 <= _ai < len(_all_acpts):
            st.session_state["_undo_state"] = {"route_points": _deep_copy_rp(rp)}
            _trkpt_idx = _all_acpts[_ai][0]
            _is_first = (_ai == 0)
            _is_last  = (_ai == len(_all_acpts) - 1)
            _new_pos  = (_new_lat, _new_lng)

            if _is_first:
                _nxt_idx = _next_boundary(_trkpt_idx, rp)
                _seg = calc_route_segment([_new_pos, (rp[_nxt_idx]["lat"], rp[_nxt_idx]["lon"])])
                new_pts = [
                    make_route_point(lat, lon, is_acpt=(j == 0), changed=True)
                    for j, (lat, lon) in enumerate(_seg[:-1])
                ]
                if new_pts:
                    new_pts[0]["wpt"]    = {"name": "スタート", "delta": None}
                    new_pts[0]["changed"] = False
                rp[0:_nxt_idx] = new_pts

            elif _is_last:
                _prev_idx = _prev_boundary(_trkpt_idx, rp)
                _seg = calc_route_segment([(rp[_prev_idx]["lat"], rp[_prev_idx]["lon"]), _new_pos])
                _seg_tail = _seg[1:]
                new_pts = [
                    make_route_point(lat, lon, is_acpt=(j == len(_seg_tail) - 1), changed=True)
                    for j, (lat, lon) in enumerate(_seg_tail)
                ]
                if new_pts:
                    new_pts[-1]["wpt"]    = {"name": "目的地", "delta": None}
                    new_pts[-1]["changed"] = False
                rp[_prev_idx+1:] = new_pts

            else:
                _prev_idx = _prev_boundary(_trkpt_idx, rp)
                _nxt_idx  = _next_boundary(_trkpt_idx, rp)
                _bwd = calc_route_segment([(rp[_prev_idx]["lat"], rp[_prev_idx]["lon"]), _new_pos])
                _fwd = calc_route_segment([_new_pos, (rp[_nxt_idx]["lat"], rp[_nxt_idx]["lon"])])
                new_pts = (
                    [make_route_point(lat, lon, changed=True) for lat, lon in _bwd[1:]]
                    + [make_route_point(lat, lon, changed=True) for lat, lon in _fwd[1:-1]]
                )
                # mark the dragged acpt position (_bwd[1:]の最後 = index len(_bwd)-2)
                _acpt_pos_in_new = len(_bwd) - 2
                if 0 <= _acpt_pos_in_new < len(new_pts):
                    new_pts[_acpt_pos_in_new]["is_acpt"] = True
                rp[_prev_idx+1:_nxt_idx] = new_pts

            st.session_state["route_points"] = rp
            st.session_state["route_modified"] = True
            st.session_state["_skip_map_center_save"] = True
            st.rerun()

    elif _evt_type == "acpt_delete":
        _ai  = _map_event.get("acpt_idx")
        rp = list(st.session_state.get("route_points", []))
        _all_acpts = [(i, p) for i, p in enumerate(rp) if p["is_acpt"]]
        if _ai is not None and 0 <= _ai < len(_all_acpts):
            st.session_state["_undo_state"] = {"route_points": _deep_copy_rp(rp)}
            _trkpt_idx = _all_acpts[_ai][0]
            _is_first  = (_ai == 0)
            _is_last   = (_ai == len(_all_acpts) - 1)

            if _is_first:
                # 次の境界まで切り落とす
                if len(_all_acpts) <= 1:
                    # acptが1つだけ → ルート全消去
                    st.session_state["route_points"] = []
                else:
                    _nxt_idx = _all_acpts[1][0]
                    rp_new = rp[_nxt_idx:]
                    # 新スタートにacptとwptを付ける
                    if rp_new:
                        rp_new[0]["is_acpt"] = True
                        if rp_new[0]["wpt"] is None or rp_new[0]["wpt"].get("name") != "スタート":
                            rp_new[0]["wpt"] = {"name": "スタート", "delta": None}
                    st.session_state["route_points"] = rp_new

            elif _is_last:
                # 前の境界まで切り落とす
                if len(_all_acpts) <= 1:
                    st.session_state["route_points"] = []
                else:
                    _prev_idx = _all_acpts[-2][0]
                    rp_new = rp[:_prev_idx + 1]
                    if rp_new:
                        rp_new[-1]["is_acpt"] = True
                        if rp_new[-1]["wpt"] is None or rp_new[-1]["wpt"].get("name") != "目的地":
                            rp_new[-1]["wpt"] = {"name": "目的地", "delta": None}
                    st.session_state["route_points"] = rp_new

            else:
                # 中間acptを削除してセグメントを再計算
                _prev_idx = _prev_boundary(_trkpt_idx, rp)
                _nxt_idx  = _next_boundary(_trkpt_idx, rp)
                _seg = calc_route_segment([(rp[_prev_idx]["lat"], rp[_prev_idx]["lon"]),
                                           (rp[_nxt_idx]["lat"], rp[_nxt_idx]["lon"])])
                new_mid = [make_route_point(lat, lon, changed=True) for lat, lon in _seg[1:-1]]
                rp[_prev_idx+1:_nxt_idx] = new_mid
                st.session_state["route_points"] = rp

            st.session_state["route_modified"] = True
            st.session_state["_skip_map_center_save"] = True
            st.rerun()

    elif _evt_type == "dialog_result" and _map_event.get("action") == "acpt":
        _near_idx = _map_event.get("nearest_trkpt_idx", 0)
        rp = list(st.session_state.get("route_points", []))
        st.session_state["_undo_state"] = {"route_points": _deep_copy_rp(rp)}
        _near_pos = (rp[_near_idx]["lat"], rp[_near_idx]["lon"])
        _prev_idx = _prev_boundary(_near_idx, rp)
        _nxt_idx  = _next_boundary(_near_idx, rp)
        _seg1 = calc_route_segment([(rp[_prev_idx]["lat"], rp[_prev_idx]["lon"]), _near_pos])
        _seg2 = calc_route_segment([_near_pos, (rp[_nxt_idx]["lat"], rp[_nxt_idx]["lon"])])
        new_pts = (
            [make_route_point(lat, lon, changed=True) for lat, lon in _seg1[1:]]
            + [make_route_point(lat, lon, changed=True) for lat, lon in _seg2[1:-1]]
        )
        # mark the new acpt position (_seg1[1:]の最後 = index len(_seg1)-2)
        _new_acpt_pos = len(_seg1) - 2
        if 0 <= _new_acpt_pos < len(new_pts):
            new_pts[_new_acpt_pos]["is_acpt"] = True
        rp[_prev_idx+1:_nxt_idx] = new_pts
        st.session_state["route_points"] = rp
        st.session_state["route_modified"] = True
        st.session_state["_skip_map_center_save"] = True
        st.rerun()

    elif _evt_type == "dialog_result" and _map_event.get("action") == "wpt":
        _near_idx = _map_event.get("nearest_trkpt_idx", 0)
        rp = list(st.session_state.get("route_points", []))
        if rp[_near_idx]["wpt"] is None:
            if 1 <= _near_idx < len(rp) - 1:
                _bi = calculate_bearing(rp[_near_idx-1]["lat"], rp[_near_idx-1]["lon"],
                                        rp[_near_idx]["lat"],   rp[_near_idx]["lon"])
                _bo = calculate_bearing(rp[_near_idx]["lat"],   rp[_near_idx]["lon"],
                                        rp[_near_idx+1]["lat"], rp[_near_idx+1]["lon"])
                _delta = angle_diff(_bi, _bo)
            else:
                _delta = None
            _tmp = {"lat": rp[_near_idx]["lat"], "lon": rp[_near_idx]["lon"],
                    "index": _near_idx, "delta": _delta}
            _inames = fetch_intersection_names([_tmp], radius=20)
            _iname  = _inames.get(_near_idx)
            if _delta is not None:
                _wpt_info = with_name({"name": "", "delta": _delta}, _iname)
            elif _iname:
                _wpt_info = {"name": _iname, "delta": None}
            else:
                _spot = fetch_spot_name(rp[_near_idx]["lat"], rp[_near_idx]["lon"], radius=20)
                _wpt_info = {"name": f"「{_spot}」" if _spot else "追加したターンポイント", "delta": None}
            rp[_near_idx]["wpt"] = _wpt_info
            st.session_state["route_points"] = rp
        st.session_state["_skip_map_center_save"] = True
        st.rerun()

    elif _evt_type == "wpt_click":
        _wpt_idx = _map_event.get("wpt_idx")
        current_wpts_list = [(i, p) for i, p in enumerate(route_points) if p["wpt"] is not None]
        if _wpt_idx is not None and 0 <= _wpt_idx < len(current_wpts_list):
            _wi, _wp = current_wpts_list[_wpt_idx]
            st.session_state["_focus_wpt_idx"] = _wpt_idx
            st.session_state["_map_center"] = {
                "lat": _wp["lat"],
                "lng": _wp["lon"],
            }
            st.session_state["_skip_map_center_save"] = True
        st.rerun()

# wpt 名称をwidget stateから同期
route_points = st.session_state.get("route_points", [])
for i, p in enumerate(route_points):
    if p["wpt"] is None:
        continue
    _sync_key = f"wpt_name_{i}"
    if _sync_key in st.session_state:
        p["wpt"]["name"] = st.session_state[_sync_key]


# ─── 右パネル（リスト） ───────────────────────
with col_list:
    current_wpts = [(i, p) for i, p in enumerate(route_points) if p["wpt"] is not None]
    st.subheader(f"📋 ターンポイント一覧　({len(current_wpts)}件)")

    _wpt_detect_col, _undo_col = st.columns([2, 1])
    with _wpt_detect_col:
        if st.button("🔍 ターンポイント検出", use_container_width=True,
                     help="ルート上のターンを検出してナビ案内点を設定します"):
            _changed_idx = [i for i, p in enumerate(route_points) if p["changed"]]
            if _changed_idx:
                for _rstart, _rend in _contiguous_ranges(_changed_idx):
                    _sub = [(p["lat"], p["lon"]) for p in route_points[_rstart:_rend+1]]
                    _raw = detect_turns(_sub, min_turn_angle=45, min_dist=100, smooth=1)
                    _cands = [
                        {"lat": route_points[_rstart + t["index"]]["lat"],
                         "lon": route_points[_rstart + t["index"]]["lon"],
                         "index": _rstart + t["index"],
                         "delta": t["delta"]}
                        for t in _raw
                    ]
                    _inames_det = fetch_intersection_names(_cands)
                    for _ct in _cands:
                        _cidx = _ct["index"]
                        if route_points[_cidx]["wpt"] is None:
                            route_points[_cidx]["wpt"] = with_name(
                                {"name": "", "delta": _ct["delta"]},
                                _inames_det.get(_cidx)
                            )
                for p in route_points:
                    p["changed"] = False
                st.session_state["route_points"] = route_points
                st.session_state["route_modified"] = False
            st.rerun()
    with _undo_col:
        if st.button("↩ 戻す", use_container_width=True,
                     disabled=("_undo_state" not in st.session_state),
                     help="直前の操作を元に戻す（1回のみ）"):
            _us = st.session_state.pop("_undo_state")
            st.session_state["route_points"] = _us["route_points"]
            st.rerun()
    if st.session_state.get("route_modified") and current_wpts:
        st.warning("ルートが変更されています。ターンポイント検出を実行してください。", icon="⚠️")

    st.markdown("""<style>
    [data-testid="stButton"] button {
        font-size: 0.72rem !important;
        padding: 2px 6px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    [data-testid="stButton"] button p {
        font-size: 0.72rem !important;
        white-space: nowrap !important;
    }
    [data-testid="stTextInput"] input {
        font-size: 0.72rem !important;
        padding: 2px 6px !important;
    }
    </style>""", unsafe_allow_html=True)

    with st.container(height=520):
        if not current_wpts:
            st.info("ターンポイントがありません。")

        for list_idx, (trkpt_idx, p) in enumerate(current_wpts):
            winfo = p["wpt"]
            arrow, hex_color = wpt_style(winfo)
            delta = winfo.get("delta")
            badge = f"{delta:+.1f}°" if delta is not None else "手動"
            col_c, col_n, col_d = st.columns([3, 5, 1])
            with col_c:
                if st.button(
                    f"{list_idx+1} | {arrow} | trkpt: {trkpt_idx}",
                    key=f"center_{list_idx}",
                    use_container_width=True,
                    help="地図を移動",
                ):
                    st.session_state["_map_center"] = {"lat": p["lat"], "lng": p["lon"]}
                    st.session_state["_focus_wpt_idx"] = list_idx
                    st.session_state["_skip_map_center_save"] = True
                    st.rerun()
            with col_n:
                _wkey = f"wpt_name_{trkpt_idx}"
                if _wkey not in st.session_state:
                    st.session_state[_wkey] = winfo["name"]
                st.text_input(
                    "名前",
                    key=_wkey,
                    label_visibility="collapsed",
                )
            with col_d:
                if st.button("🗑", key=f"del_{list_idx}", help="削除"):
                    route_points[trkpt_idx]["wpt"] = None
                    st.session_state["route_points"] = route_points
                    st.rerun()

        _focus_wpt_idx = st.session_state.pop("_focus_wpt_idx", None)
        if _focus_wpt_idx is not None:
            components.html(
                f"""<script>
                setTimeout(function() {{
                    var inputs = window.parent.document.querySelectorAll('[data-testid="stTextInput"] input');
                    var el = inputs[{_focus_wpt_idx}];
                    if (el) {{
                        el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                        el.focus();
                    }}
                }}, 300);
                </script>""",
                height=0,
            )

    st.caption("💡 地図をクリックして新しいポイントを追加。ナビゲーションの内容は、「左折」「やや左」「直進」「やや右」「右折」を推奨しますが、フリーワードです。「左」、「右」の文字を入れておくと逆走時に正しく変換されます")

# ─────────────────────────────────────────────
# ダウンロード
# ─────────────────────────────────────────────

st.divider()
@st.dialog("💾 GPXを保存")
def _save_gpx_dialog(route_points, elev_choice):
    # wpt名称を最新のwidget stateで同期
    for i, p in enumerate(route_points):
        if p["wpt"] is not None:
            key = f"wpt_name_{i}"
            if key in st.session_state:
                p["wpt"]["name"] = st.session_state[key]

    org_ok = bool(route_points) and all(p["ele_org"] is not None for p in route_points)
    fix_ok = bool(route_points) and all(p["ele_fix"] is not None for p in route_points)
    current_wpts_count = sum(1 for p in route_points if p["wpt"] is not None)
    route_modified = st.session_state.get("route_modified")

    # ── ステータス表示 ──────────────────────────────
    if route_modified:
        st.markdown("**ターンポイント** ⚠️ 未確定（ルート変更後に再検出を推奨）")
    else:
        st.markdown(f"**ターンポイント** ✅ 設定済み（{current_wpts_count} 個）")

    if org_ok or fix_ok:
        st.markdown("**標高** ✅ 設定済み")
    else:
        st.markdown("**標高** ⚠️ データなし")

    _wpt_issue  = bool(route_modified)
    _elev_issue = not org_ok and not fix_ok
    if _wpt_issue and _elev_issue:
        st.warning(
            "ターンポイントが未確定で、標高不明のセグメントが存在します。"
            "「ターンポイント検出」と「国土地理院で標高補正」を行ってから保存することを推奨します。"
        )
    elif _wpt_issue:
        st.warning(
            "ターンポイントが未確定です。"
            "ルート変更後に「ターンポイント検出」を行ってから保存することを推奨します。"
        )
    elif _elev_issue:
        st.warning(
            "標高不明のセグメントが存在します。"
            "「国土地理院で標高補正」を行ってから保存することを推奨します。"
        )

    st.divider()

    # ── ファイル名入力 ──────────────────────────────
    _default = (st.session_state.get("_gpx_filename", "new_route")
                .replace(".gpx", "") or "new_route") + "_gne"
    _fname = st.text_input("ファイル名", value=_default, key="_dialog_fname")

    # ── GPX生成 ────────────────────────────────────
    _xml = build_enhanced_gpx(
        st.session_state.get("_raw_gpx"),
        route_points,
        elev_choice,
    )

    # ── ボタン行 ───────────────────────────────────
    _dc1, _dc2 = st.columns(2)
    with _dc1:
        if st.button("キャンセル", use_container_width=True, key="_dialog_cancel"):
            st.session_state.pop("_save_dialog", None)
            st.rerun()
    with _dc2:
        if st.download_button(
            "⬇️ ダウンロード",
            data=_xml,
            file_name=f"{_fname}.gpx",
            mime="application/gpx+xml",
            type="primary",
            use_container_width=True,
            key="_dialog_dl",
        ):
            st.session_state.pop("_save_dialog", None)
            st.rerun()


@st.dialog("⛰️ 国土地理院 標高補正中", width="large")
def _gsi_progress_dialog():
    _rp         = st.session_state.get("route_points", [])
    _en         = len(_rp)
    _E_BATCH    = 50
    _en_batches = math.ceil(_en / _E_BATCH)
    _ebi        = st.session_state.get("_elev_batch_idx", 0)
    _ecancelled = st.session_state.pop("_elev_cancel_requested", False)

    _elev_prog_area = st.empty()
    _pending_retry = st.session_state.get("_elev_retry_idxs")
    if _pending_retry:
        _es_init = _ebi * _E_BATCH
        _ee_init = min(_es_init + _E_BATCH, _en)
        _n_confirmed = (_ee_init - _es_init) - len(_pending_retry)
        _elev_prog_area.progress(
            (_es_init + _n_confirmed) / _en,
            text=f"⚠️ 国土地理院が遅いです（{len(_pending_retry)}点再試行中）… {_es_init + _n_confirmed}/{_en} 点確定",
        )
    else:
        _elev_prog_area.progress(
            _ebi / _en_batches,
            text=f"⛰️ 標高補正中（国土地理院）… {min(_ebi * _E_BATCH, _en)}/{_en} 点",
        )
    if st.button("⏹ キャンセル", key="elev_cancel_btn"):
        st.session_state["_elev_cancel_requested"] = True
        st.rerun()

    def _finalize_fix(partial_list, cancelled=False):
        rp = st.session_state.get("route_points", [])
        if not cancelled and partial_list:
            _coords_fix = [(p["lat"], p["lon"]) for p in rp]
            _fix_cleaned, _ = clean_elevation_spikes(_coords_fix, partial_list)
            for i, v in enumerate(_fix_cleaned):
                if i < len(rp):
                    rp[i]["ele_fix"] = v
            st.session_state["_grade_fix"] = compute_grade_stats(_coords_fix, _fix_cleaned)
        else:
            for p in rp:
                p["ele_fix"] = None
            st.session_state["_grade_fix"] = None
        st.session_state["route_points"] = rp
        for _ek in ["_elev_batch_idx", "_elev_partial", "_elev_retry_idxs"]:
            st.session_state.pop(_ek, None)
        st.session_state["_proc_status"] = "done"
        _set_default_elev_choice()

    if _ecancelled:
        _ep = st.session_state.get("_elev_partial", [None] * _en)
        _finalize_fix(_ep, cancelled=True)
        _elev_prog_area.progress(1.0, text="✅ キャンセルしました")
        st.rerun()
    else:
        _es = _ebi * _E_BATCH
        _ee = min(_es + _E_BATCH, _en)
        _ep = st.session_state.get("_elev_partial", [None] * _en)

        _etls = threading.local()
        def _efetch(args):
            _ei, _elat, _elon = args
            if not hasattr(_etls, "session"):
                _etls.session = requests.Session()
            try:
                _er = _etls.session.get(
                    "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php",
                    params={"lat": _elat, "lon": _elon, "outtype": "JSON"},
                    timeout=10,
                )
                _er.raise_for_status()
                _ev = _er.json().get("elevation")
                val = None if (_ev is None or _ev == -9999 or _ev == "-----") else float(_ev)
                return _ei, val, False
            except Exception:
                return _ei, None, True

        _retry_idxs = st.session_state.pop("_elev_retry_idxs", None)
        _is_retry   = _retry_idxs is not None
        _etasks = (
            [(_ei, _rp[_ei]["lat"], _rp[_ei]["lon"]) for _ei in _retry_idxs]
            if _is_retry
            else [(_es + i, _rp[_es + i]["lat"], _rp[_es + i]["lon"]) for i in range(_ee - _es)]
        )
        _edone      = [0]
        _econfirmed = [0]
        _bar_prev   = [0]
        _err_idxs   = []
        with ThreadPoolExecutor(max_workers=10) as _eex:
            for _ef in as_completed({_eex.submit(_efetch, t): t[0] for t in _etasks}):
                _ei2, _ev2, _err = _ef.result()
                if _err:
                    _err_idxs.append(_ei2)
                else:
                    _ep[_ei2] = _ev2
                    _econfirmed[0] += 1
                _edone[0] += 1
                if not _is_retry and (_econfirmed[0] - _bar_prev[0] >= 10 or _edone[0] == len(_etasks)):
                    _bar_prev[0] = _econfirmed[0]
                    _pts_confirmed = _es + _econfirmed[0]
                    _elev_prog_area.progress(
                        _pts_confirmed / _en,
                        text=f"⛰️ 標高補正中（国土地理院）… {_pts_confirmed}/{_en} 点",
                    )
        if _err_idxs:
            st.session_state["_elev_retry_idxs"] = _err_idxs
            _n_confirmed = (_ee - _es) - len(_err_idxs)
            _elev_prog_area.progress(
                (_es + _n_confirmed) / _en,
                text=f"⚠️ 国土地理院が遅いです（{len(_err_idxs)}点再試行中）… {_es + _n_confirmed}/{_en} 点確定",
            )
            st.rerun()

        st.session_state["_elev_partial"] = _ep
        if _ebi + 1 >= _en_batches:
            _finalize_fix(_ep)
            _elev_prog_area.progress(1.0, text="✅ 標高補正完了")
        else:
            st.session_state["_elev_batch_idx"] = _ebi + 1
        st.rerun()


st.markdown("#### 💾 GPXの出力")

# ── 標高設定（面取り矩形） ─────────────────────────
_org_ok = bool(route_points) and all(p["ele_org"] is not None for p in route_points)
_fix_ok = bool(route_points) and all(p["ele_fix"] is not None for p in route_points)

with st.container():
    st.markdown('<span class="elev-section-marker" style="display:none"></span>', unsafe_allow_html=True)
    st.markdown("""<style>
    div[data-testid="stVerticalBlock"]:has(span.elev-section-marker):not(:has(div[data-testid="stVerticalBlock"] span.elev-section-marker)) {
        border: 1.5px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        background-color: #f9fafb;
    }
    </style>""", unsafe_allow_html=True)
    if st.session_state.get("_proc_status") == "running_fix":
        _gsi_progress_dialog()
    else:
        _gsi_disabled = not (route_points and _is_in_japan(route_points[0]["lat"], route_points[0]["lon"]))
        _n_pts = len(route_points)

        _go = st.session_state.get("_grade_org")
        _gf = st.session_state.get("_grade_fix")
        _candidates = ([k for k, ok in [("org", _org_ok), ("fix", _fix_ok)] if ok])

        if len(_candidates) == 0:
            _rec = None
            _btn_star = "　★推奨" if not _gsi_disabled else ""
        elif len(_candidates) == 1:
            _rec = _candidates[0]
            _btn_star = ""
        else:
            _so = (_go["max"] + abs(_go["min"])) if _go else float("inf")
            _sf = (_gf["max"] + abs(_gf["min"])) if _gf else float("inf")
            _rec = "fix" if _sf < _so else "org"
            _btn_star = ""

        if st.button(f"⛰️ 国土地理院で標高補正{_btn_star}", disabled=_gsi_disabled,
                     help="日本国内のルートのみ対応。実行後にGSI補正データを選択できます。"):
            st.session_state["_proc_status"]    = "running_fix"
            st.session_state["_elev_batch_idx"] = 0
            st.session_state["_elev_partial"]   = [None] * len(route_points)
            # ele_fix をリセット
            for p in route_points:
                p["ele_fix"] = None
            st.session_state["route_points"] = route_points
            st.session_state.pop("_grade_fix", None)
            st.rerun()

        if st.session_state.get("_proc_status") == "done" and route_points:
            def _grade_label(key, grade):
                if key == "fix" and not _fix_ok:
                    return "国土地理院補正（実施前）"
                if key == "org" and not _org_ok:
                    return "元データ（標高不明のセグメントあり）"
                name = "元データ（スパイク補正済み）" if key == "org" else "国土地理院補正（スパイク補正済み）"
                star = "　★推奨" if key == _rec else ""
                return f"{name}{star}　上り {grade['max']:+.1f}%  下り {grade['min']:+.1f}%"

            _opts = ["org", "fix"]
            _lbls = [_grade_label("org", _go), _grade_label("fix", _gf)]
            _cur_choice = st.session_state.get("_elev_choice", _rec or "org")
            if _cur_choice == "fix" and not _fix_ok:
                _cur_choice = "org"
                st.session_state["_elev_choice"] = "org"
            _sel = st.radio("標高データ選択", _lbls, index=_opts.index(_cur_choice))
            _chosen = _opts[_lbls.index(_sel)]
            if _chosen == "fix" and not _fix_ok:
                _chosen = "org"
            if _chosen != _cur_choice:
                st.session_state["_elev_choice"] = _chosen
                st.rerun()
            else:
                st.session_state["_elev_choice"] = _chosen
        else:
            st.session_state.setdefault("_elev_choice", "org")

_choice = st.session_state.get("_elev_choice", "org")

# ── 適用済み情報 ─────────────────────────────────
_applied = []
if st.session_state.get("_mm_status") == "完了":
    _applied.append("🗺️ マップマッチング済み")
if _choice == "fix" and _fix_ok:
    _applied.append("⛰️ 標高補正済み（国土地理院）")
elif _choice == "org" and _org_ok:
    _applied.append("⛰️ 標高補正済み（元データ）")

_col_applied, _col_btn = st.columns([3, 1])
with _col_applied:
    if _applied:
        st.caption("出力に適用: " + " ／ ".join(_applied))
with _col_btn:
    if st.button("💾 GPXを保存", type="primary",
                 use_container_width=True, disabled=(len(current_wpts) == 0)):
        st.session_state["_save_dialog"] = True

if st.session_state.get("_save_dialog"):
    _save_gpx_dialog(route_points, _choice)
