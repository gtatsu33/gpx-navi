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
import folium
from streamlit_folium import st_folium
import requests
import urllib.parse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="GPX ターン検出ツール", layout="wide", page_icon="🚴")
st.title("🚴 GPX ターン検出・強化ツール")

_is_admin = (st.query_params.get("admin", "") == st.secrets.get("ADMIN_TOKEN", "__unset__"))
st.caption("Stravaなどのターン情報なしGPXにナビ用ターンポイントを追加します")

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

def with_name(t, intersection_name=None):
    """nameフィールドを持つターン辞書を返す。
    intersection_name がある場合は「△△交差点を右折」形式、
    ない場合は従来通り「右折」形式。
    """
    d = dict(t)
    dir_label = turn_label(t["delta"])[0]
    if intersection_name:
        d["name"] = f"{intersection_name}を{dir_label}"
    else:
        d["name"] = dir_label
    return d

def wpt_style(t):
    """(arrow, hex_color) を返す"""
    delta = t.get("delta")
    if delta is not None:
        _, arrow, color = turn_label(delta)
    else:
        arrow, color = "📍", "#27ae60"
    return arrow, color

# ─────────────────────────────────────────────
# 交差点名取得（Overpass API / OSM直接）
# ─────────────────────────────────────────────

_OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
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
    prog = st.progress(0, text=f"交差点名を取得中…（{n} 件）")

    # ^ $ で完全一致させ bus_stop などの部分一致を防ぐ
    _HW = '"^(traffic_signals|crossing|give_way|stop|mini_roundabout|motorway_junction)$"'
    union_parts = "".join(
        f'node(around:{radius},{t["lat"]},{t["lon"]})[name][highway~{_HW}];'
        for t in turns
    )
    query = f"[out:json][timeout:25];({union_parts});out body;"

    elements = None
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
        prog.empty()
        return {}

    # 各ターンポイントに最近傍ノードの名前を対応付ける
    prog.progress(0.8, text="交差点名を処理中…")
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
        prog.progress(0.8 + 0.2 * (i + 1) / n, text="交差点名を処理中…")

    prog.empty()
    return result

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
        # 境界アンカー候補がスパイク隣接点の場合は is_anchor_candidate で弾く
        # （隣接勾配チェックが入るため、スパイク端点は自動的に除外される）
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

def build_enhanced_gpx(gpx_content_str, turns, matched_points=None, elevations=None):
    enhanced = gpxpy.parse(gpx_content_str)

    # trkpt の座標・標高を更新
    all_pts = [pt for tr in enhanced.tracks
               for seg in tr.segments for pt in seg.points]
    if matched_points:
        for i, pt in enumerate(all_pts):
            if i < len(matched_points):
                pt.latitude  = matched_points[i][0]
                pt.longitude = matched_points[i][1]
    if elevations:
        for i, pt in enumerate(all_pts):
            if i < len(elevations) and elevations[i] is not None:
                pt.elevation = elevations[i]

    # ターンポイントを再構築
    enhanced.waypoints = []
    for t in turns:
        name  = t.get("name") or turn_label(t["delta"])[0]
        delta = t.get("delta")
        desc  = f"bearing_change:{delta:.1f}" if delta is not None else "manually added"
        enhanced.waypoints.append(gpxpy.gpx.GPXWaypoint(
            latitude=t["lat"], longitude=t["lon"], name=name, description=desc,
        ))
    return enhanced.to_xml()

# ─────────────────────────────────────────────
# ファイルアップロード
# ─────────────────────────────────────────────

uploaded = st.file_uploader("GPXファイルをアップロード", type=["gpx"])

if uploaded is None:
    st.info("GPXファイルをアップロードしてください（Stravaなどのルートエクスポートが対象）")
    st.stop()

raw_content = uploaded.read().decode("utf-8")
try:
    gpx_parsed = gpxpy.parse(raw_content)
except Exception as e:
    st.error(f"GPXの解析に失敗しました: {e}")
    st.stop()

points = []
for track in gpx_parsed.tracks:
    for segment in track.segments:
        for pt in segment.points:
            points.append((pt.latitude, pt.longitude))

if len(points) < 6:
    st.error("トラックポイントが少なすぎます。")
    st.stop()

_has_wpts = len(gpx_parsed.waypoints) > 0

# ファイルが変わったらセッション状態をリセット
_STATE_KEYS = [
    "edit_turns", "pending_wpt", "_handled_click", "_handled_tooltip",
    "_map_center", "_map_zoom",
    "_matched_points", "_mm_status", "_mm_n_snapped", "_mm_error",
    "_mm_chunk_idx", "_mm_matched_partial", "_mm_n_snapped_partial", "_mm_errors_partial",
    "_proc_status",
    "_trkpt_org_elevs", "_trkpt_fix_elevs", "_grade_org", "_grade_fix", "_elev_choice",
    "_elev_batch_idx", "_elev_partial", "_elev_cancel_requested",
    "_iname_status",  "_iname_n_found",
]
if st.session_state.get("_file_name") != uploaded.name:
    st.session_state["_file_name"] = uploaded.name
    for k in _STATE_KEYS:
        st.session_state.pop(k, None)

_skip_map_center_save = st.session_state.pop("_skip_map_center_save", False)

# ─────────────────────────────────────────────
# 自動処理（マップマッチング・標高補正）
# ─────────────────────────────────────────────
_needs_rerun = False

if st.session_state.get("_mm_status") is None:
    if _has_wpts:
        st.session_state["_matched_points"] = list(points)
        st.session_state["_mm_status"]      = "スキップ"
    else:
        st.session_state["_mm_status"]              = "running"
        st.session_state["_mm_chunk_idx"]           = 0
        st.session_state["_mm_matched_partial"]     = list(points)
        st.session_state["_mm_n_snapped_partial"]   = 0
        st.session_state["_mm_errors_partial"]      = []

if st.session_state.get("_mm_status") == "running":
    _MM_CHUNK  = 50
    _ci        = st.session_state["_mm_chunk_idx"]
    _n_chunks  = math.ceil(len(points) / _MM_CHUNK)
    _cancelled = st.session_state.pop("_mm_cancel_requested", False)

    _col_prog, _col_btn = st.columns([5, 1])
    with _col_prog:
        _prog_area = st.empty()
        if _cancelled:
            _prog_area.progress(_ci / _n_chunks, text="⏱️ キャンセル待ち中…")
        else:
            _prog_area.progress((_ci + 1) / _n_chunks,
                                text=f"🗺️ マップマッチング中… {_ci + 1}/{_n_chunks} チャンク")
    with _col_btn:
        if st.button("⏹ キャンセル", key="mm_cancel_btn"):
            st.session_state["_mm_cancel_requested"] = True
            st.rerun()

    if _cancelled:
        _errs = st.session_state.get("_mm_errors_partial", [])
        st.session_state["_matched_points"] = st.session_state.get("_mm_matched_partial", list(points))
        st.session_state["_mm_n_snapped"]   = st.session_state.get("_mm_n_snapped_partial", 0)
        st.session_state["_mm_error"]       = "キャンセルされました" + ("; " + "; ".join(_errs) if _errs else "")
        st.session_state["_mm_status"]      = "キャンセル"
        for _k in ["_mm_chunk_idx", "_mm_matched_partial", "_mm_n_snapped_partial", "_mm_errors_partial"]:
            st.session_state.pop(_k, None)
        _prog_area.progress(1.0, text="✅ キャンセルしました")
        _needs_rerun = True
    else:
        _s = _ci * _MM_CHUNK
        _e = min(_s + _MM_CHUNK, len(points))
        _auto_cancel = False

        try:
            _mp_list = st.session_state["_mm_matched_partial"]
            _data = _valhalla_match_chunk(points[_s:_e])
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
            st.session_state["_matched_points"] = list(points)
            st.session_state["_mm_n_snapped"]   = 0
            st.session_state["_mm_error"]       = "1チャンク目タイムアウトにより自動キャンセル"
            st.session_state["_mm_status"]      = "キャンセル"
            for _k in ["_mm_chunk_idx", "_mm_matched_partial", "_mm_n_snapped_partial", "_mm_errors_partial"]:
                st.session_state.pop(_k, None)
            _prog_area.progress(1.0, text="✅ タイムアウトによりキャンセルしました")
            _needs_rerun = True
        elif _ci + 1 >= _n_chunks:
            _errs  = st.session_state.get("_mm_errors_partial", [])
            _n_sn  = st.session_state.get("_mm_n_snapped_partial", 0)
            st.session_state["_matched_points"] = st.session_state.get("_mm_matched_partial", list(points))
            st.session_state["_mm_n_snapped"]   = _n_sn
            st.session_state["_mm_error"]       = "; ".join(_errs) if _errs else None
            st.session_state["_mm_status"]      = "完了" if _n_sn > 0 else "エラー"
            for _k in ["_mm_chunk_idx", "_mm_matched_partial", "_mm_n_snapped_partial", "_mm_errors_partial"]:
                st.session_state.pop(_k, None)
            _active_mm = st.session_state["_matched_points"]
            if "edit_turns" in st.session_state:
                for _t in st.session_state["edit_turns"]:
                    _tidx = _t.get("index", 0)
                    if _tidx < len(_active_mm):
                        _t["lat"] = _active_mm[_tidx][0]
                        _t["lon"] = _active_mm[_tidx][1]
            st.rerun()
        else:
            st.session_state["_mm_chunk_idx"] = _ci + 1
            st.rerun()

active_points = st.session_state.get("_matched_points", points)

# ─────────────────────────────────────────────
# 標高処理（org: 元GPX＋スパイク補正 / fix: GSI＋スパイク補正）
# ─────────────────────────────────────────────

def _set_default_elev_choice():
    if "_elev_choice" in st.session_state:
        return
    _go = st.session_state.get("_grade_org")
    _gf = st.session_state.get("_grade_fix")
    _so = (_go["max"] + abs(_go["min"])) if _go else float("inf")
    _sf = (_gf["max"] + abs(_gf["min"])) if _gf else float("inf")
    st.session_state["_elev_choice"] = "fix" if _sf < _so else "org"

if st.session_state.get("_proc_status") is None:
    # trkpt_org: 元GPX ele値 + スパイク補正
    _orig_elevs = [p.elevation for tr in gpx_parsed.tracks
                   for seg in tr.segments for p in seg.points]
    if all(e is None for e in _orig_elevs):
        st.session_state["_trkpt_org_elevs"] = None
        st.session_state["_grade_org"]       = None
    else:
        _org_cleaned, _ = clean_elevation_spikes(active_points, _orig_elevs)
        st.session_state["_trkpt_org_elevs"] = _org_cleaned
        st.session_state["_grade_org"]       = compute_grade_stats(active_points, _org_cleaned)

    # trkpt_fix: GSI標高取得 → スパイク補正
    if _has_wpts or not (active_points and _is_in_japan(active_points[0][0], active_points[0][1])):
        st.session_state["_trkpt_fix_elevs"] = None
        st.session_state["_grade_fix"]       = None
        st.session_state["_proc_status"]     = "done"
        _set_default_elev_choice()
    else:
        st.session_state["_proc_status"]    = "running_fix"
        st.session_state["_elev_batch_idx"] = 0
        st.session_state["_elev_partial"]   = [None] * len(active_points)
        st.rerun()

# GSI標高取得ループ（1 rerun = 1 バッチ）
if st.session_state.get("_proc_status") == "running_fix":
    _en         = len(active_points)
    _E_BATCH    = 50
    _en_batches = math.ceil(_en / _E_BATCH)
    _ebi        = st.session_state.get("_elev_batch_idx", 0)
    _ecancelled = st.session_state.pop("_elev_cancel_requested", False)

    _ecol_prog, _ecol_btn = st.columns([5, 1])
    with _ecol_prog:
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
    with _ecol_btn:
        if st.button("⏹ キャンセル", key="elev_cancel_btn"):
            st.session_state["_elev_cancel_requested"] = True
            st.rerun()

    def _finalize_fix(gsi_elevs, cancelled=False):
        if cancelled:
            st.session_state["_trkpt_fix_elevs"] = None
            st.session_state["_grade_fix"]       = None
        else:
            _fix_cleaned, _ = clean_elevation_spikes(active_points, gsi_elevs)
            st.session_state["_trkpt_fix_elevs"] = _fix_cleaned
            st.session_state["_grade_fix"]       = compute_grade_stats(active_points, _fix_cleaned)
        for _ek in ["_elev_batch_idx", "_elev_partial", "_elev_retry_idxs"]:
            st.session_state.pop(_ek, None)
        st.session_state["_proc_status"] = "done"
        _set_default_elev_choice()

    if _ecancelled:
        _ep = st.session_state.get("_elev_partial", [None] * _en)
        _finalize_fix(_ep, cancelled=True)
        _elev_prog_area.progress(1.0, text="✅ キャンセルしました")
        _needs_rerun = True
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
            [(_ei, active_points[_ei][0], active_points[_ei][1]) for _ei in _retry_idxs]
            if _is_retry
            else [(_es + i, lat, lon) for i, (lat, lon) in enumerate(active_points[_es:_ee])]
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
            _needs_rerun = True
        else:
            st.session_state["_elev_batch_idx"] = _ebi + 1
            st.rerun()

if _needs_rerun:
    st.rerun()

# ─── ターン初期化（初回のみ）──────────────────
if "edit_turns" not in st.session_state:
    if _has_wpts:
        turns = []
        for wpt in gpx_parsed.waypoints:
            delta = None
            desc  = wpt.description or ""
            if desc.startswith("bearing_change:"):
                try:
                    delta = float(desc.split(":")[1])
                except ValueError:
                    pass
            idx = nearest_trkpt_index(wpt.latitude, wpt.longitude, points)
            turns.append({
                "lat":   wpt.latitude,
                "lon":   wpt.longitude,
                "delta": delta,
                "index": idx,
                "name":  wpt.name or "ターンポイント",
            })
        st.session_state["edit_turns"]    = turns
        st.session_state["_iname_status"] = "スキップ"
    else:
        raw_turns = detect_turns(active_points, min_turn_angle=45, min_dist=100, smooth=1)
        intersection_names = fetch_intersection_names(raw_turns)
        st.session_state["_iname_status"]  = "完了"
        st.session_state["_iname_n_found"] = len(intersection_names)
        st.session_state["edit_turns"] = [
            with_name(t, intersection_names.get(t["index"]))
            for t in raw_turns
        ]

current_turns = st.session_state["edit_turns"]

if _has_wpts:
    st.info("📂 GPX内のターンポイントを読み込みました。マップマッチング・標高補正はスキップされています。")

# ─── ルート情報 ───────────────────────────────
dists_all = [haversine(active_points[i][0], active_points[i][1],
                       active_points[i+1][0], active_points[i+1][1])
             for i in range(len(active_points) - 1)]
avg_spacing   = np.mean(dists_all)
total_dist_km = sum(dists_all) / 1000
route_name    = next((t.name for t in gpx_parsed.tracks if t.name), "（名称なし）")

c1, c2, c3 = st.columns(3)
c1.metric("ルート名", route_name)
c2.metric("総距離", f"{total_dist_km:.1f} km")
c3.metric("GPSポイント間隔（平均）", f"{avg_spacing:.0f} m")

# ─────────────────────────────────────────────
# 標高データ選択UI
# ─────────────────────────────────────────────
if st.session_state.get("_proc_status") == "done":
    _go = st.session_state.get("_grade_org")
    _gf = st.session_state.get("_grade_fix")

    if _go is not None or _gf is not None:
        st.divider()
        _so = (_go["max"] + abs(_go["min"])) if _go else float("inf")
        _sf = (_gf["max"] + abs(_gf["min"])) if _gf else float("inf")
        _rec = "fix" if _sf < _so else "org"

        def _grade_label(key, grade):
            name = "元データ（スパイク補正済み）" if key == "org" else "国土地理院補正（スパイク補正済み）"
            if grade is None:
                return f"{name}　—— データなし"
            star = "　★推奨" if key == _rec else ""
            return f"{name}{star}　上り {grade['max']:+.1f}%  下り {grade['min']:+.1f}%"

        _options = ["org", "fix"]
        _labels  = [_grade_label("org", _go), _grade_label("fix", _gf)]

        _cur = st.session_state.get("_elev_choice", _rec)
        if _cur == "org" and _go is None:
            _cur = "fix"
        elif _cur == "fix" and _gf is None:
            _cur = "org"

        _sel = st.radio(
            "⛰️ 標高データ選択",
            _labels,
            index=_options.index(_cur),
        )
        _chosen = _options[_labels.index(_sel)]
        if _chosen == "org" and _go is None:
            _chosen = "fix"
        elif _chosen == "fix" and _gf is None:
            _chosen = "org"
        if _chosen != _cur:
            st.session_state["_elev_choice"] = _chosen
            st.rerun()
        else:
            st.session_state["_elev_choice"] = _chosen

# ─────────────────────────────────────────────
# 地図 + リストパネル
# ─────────────────────────────────────────────

st.info("✏️ 地図をクリックしてターンポイントを追加できます。右パネルで削除・名前（ナビゲーション内容）変更もできます。")

col_map, col_list = st.columns([2, 1])
pending = st.session_state.get("pending_wpt")

with col_map:
    st.subheader("🗺️ 地図プレビュー")
    _saved_center = st.session_state.get("_map_center")
    _map_init_loc = ([_saved_center["lat"], _saved_center["lng"]] if _saved_center
                     else active_points[len(active_points)//4])
    _map_init_zoom = st.session_state.get("_map_zoom", 13)
    m = folium.Map(location=_map_init_loc, zoom_start=_map_init_zoom)
    folium.PolyLine(active_points, color="#3498db", weight=4, opacity=0.8).add_to(m)
    folium.Marker(active_points[0],  tooltip="スタート",
                  icon=folium.Icon(color="green",   icon="play", prefix="fa")).add_to(m)
    folium.Marker(active_points[-1], tooltip="ゴール",
                  icon=folium.Icon(color="darkred", icon="flag", prefix="fa")).add_to(m)

    for i, t in enumerate(current_turns):
        arrow, hex_color = wpt_style(t)
        delta = t.get("delta")
        popup_html = (f"<b>{arrow} {t['name']}</b>"
                      + (f"<br>ターン角: {delta:+.1f}°" if delta is not None else "")
                      + f"<br>trkpt: {t['index']}")
        tooltip_str = f"wpt:{i+1} / trkpt:{t['index']} {arrow} {t['name']}"
        folium.CircleMarker(
            location=[t["lat"], t["lon"]], radius=9,
            color=hex_color, fill=True, fill_color=hex_color, fill_opacity=0.9,
            tooltip=tooltip_str,
            popup=folium.Popup(popup_html, max_width=200),
        ).add_to(m)

    if pending:
        folium.Marker(
            [pending["lat"], pending["lon"]],
            tooltip=f"追加予定 trkpt#{pending['index']}",
            icon=folium.Icon(color="orange", icon="star", prefix="fa"),
        ).add_to(m)

    map_data = st_folium(
        m, height=520, use_container_width=True,
        key="gpx_map",
        center=_map_init_loc,
        zoom=_map_init_zoom,
        returned_objects=["last_clicked", "last_object_clicked_tooltip"],
    )

# 地図の表示位置を記憶
if map_data and not _skip_map_center_save:
    if map_data.get("center"):
        st.session_state["_map_center"] = map_data["center"]
    if map_data.get("zoom") is not None:
        st.session_state["_map_zoom"] = map_data["zoom"]

# 手動編集した名前を edit_turns に同期（rerun で widget state が消える前に保持）
for _sync_t in st.session_state.get("edit_turns", []):
    _sync_key = f"wpt_name_{_sync_t['index']}"
    if _sync_key in st.session_state:
        _sync_t["name"] = st.session_state[_sync_key]

# ─── マップクリック → pending_wpt 更新 ─────────
if map_data:
    tooltip_val = map_data.get("last_object_clicked_tooltip") or ""

    if tooltip_val.startswith("wpt:") and tooltip_val != st.session_state.get("_handled_tooltip"):
        st.session_state["_handled_tooltip"] = tooltip_val
        trkpt_idx = int(tooltip_val.split(" / trkpt:")[1].split(" ")[0])
        if map_data.get("last_clicked"):
            click = map_data["last_clicked"]
            st.session_state["_handled_click"] = (round(click["lat"], 7), round(click["lng"], 7))
        st.session_state["pending_wpt"] = {
            "index": trkpt_idx,
            "lat":   active_points[trkpt_idx][0],
            "lon":   active_points[trkpt_idx][1],
        }
        st.session_state["_skip_map_center_save"] = True
        st.rerun()

    elif map_data.get("last_clicked"):
        click     = map_data["last_clicked"]
        click_key = (round(click["lat"], 7), round(click["lng"], 7))
        if click_key != st.session_state.get("_handled_click"):
            st.session_state["_handled_click"] = click_key
            idx = nearest_trkpt_index(click["lat"], click["lng"], active_points)
            existing_idx = next(
                (j for j, t in enumerate(current_turns) if t["index"] == idx),
                None,
            )
            if existing_idx is not None:
                st.session_state["pending_wpt"] = {
                    "index": idx,
                    "lat":   active_points[idx][0],
                    "lon":   active_points[idx][1],
                }
            elif idx == 0 or idx == len(active_points) - 1:
                st.session_state["pending_wpt"] = {
                    "index": idx,
                    "lat":   active_points[idx][0],
                    "lon":   active_points[idx][1],
                    "is_start_goal": True,
                }
            else:
                sm = 1
                if sm <= idx < len(active_points) - sm:
                    b_in  = calculate_bearing(
                        active_points[idx - sm][0], active_points[idx - sm][1],
                        active_points[idx][0],      active_points[idx][1],
                    )
                    b_out = calculate_bearing(
                        active_points[idx][0],      active_points[idx][1],
                        active_points[idx + sm][0], active_points[idx + sm][1],
                    )
                    wpt_delta = angle_diff(b_in, b_out)
                else:
                    wpt_delta = None
                _temp = {
                    "lat":   active_points[idx][0],
                    "lon":   active_points[idx][1],
                    "delta": wpt_delta,
                    "index": idx,
                }
                _inames = fetch_intersection_names([_temp], radius=20)
                _iname = _inames.get(idx)
                if wpt_delta is not None:
                    wpt_name = with_name(_temp, _iname)["name"]
                elif _iname:
                    wpt_name = _iname
                else:
                    wpt_name = "追加したターンポイント"
                turns_list = st.session_state["edit_turns"]
                insert_at = next(
                    (j for j, t in enumerate(turns_list) if t["index"] > idx),
                    len(turns_list),
                )
                turns_list.insert(insert_at, {**_temp, "name": wpt_name})
                st.session_state.pop("pending_wpt", None)
            st.session_state["_skip_map_center_save"] = True
            st.rerun()

# ─── 右パネル（リスト） ───────────────────────
with col_list:
    st.subheader(f"📋 ターンポイント一覧　({len(current_turns)}件)")

    if not current_turns and not pending:
        st.warning("ターンポイントが検出されませんでした。\nターン角閾値を下げてみてください。")

    for i, t in enumerate(current_turns):
        arrow, hex_color = wpt_style(t)
        delta = t.get("delta")
        badge = f"{delta:+.1f}°" if delta is not None else "手動"
        st.markdown(
            f'<div style="border-left:4px solid {hex_color};padding:3px 8px;'
            f'background:#f8f9fa;border-radius:3px;margin-bottom:2px;">'
            f'<b>{i+1}. {arrow}</b> <small style="color:#888">{badge} | trkpt:{t["index"]}</small></div>',
            unsafe_allow_html=True,
        )
        col_n, col_d = st.columns([5, 1])
        with col_n:
            _wkey = f"wpt_name_{t['index']}"
            if _wkey not in st.session_state:
                st.session_state[_wkey] = t["name"]
            st.text_input(
                "名前",
                key=_wkey,
                label_visibility="collapsed",
            )
        with col_d:
            if st.button("🗑", key=f"del_{t['index']}", help="削除"):
                st.session_state["edit_turns"].pop(i)
                st.session_state.pop("pending_wpt", None)
                st.rerun()

    # ─── 保留ターンポイント追加／削除UI ──────────
    if pending:
        st.divider()

        existing_idx = next(
            (j for j, t in enumerate(current_turns) if t["index"] == pending["index"]),
            None,
        )

        if pending.get("is_start_goal"):
            label = "スタート" if pending["index"] == 0 else "ゴール"
            st.error(f"{label}地点は追加できません")
        elif existing_idx is not None:
            existing = current_turns[existing_idx]
            ex_arrow, ex_color = wpt_style(existing)
            st.markdown(
                f'<div style="border-left:4px solid {ex_color};padding:6px 10px;'
                f'border-radius:4px;background:#fff3cd;">'
                f'<b>{ex_arrow} {existing["name"]}</b><br>'
                f'<small>trkpt #{pending["index"]}</small></div>',
                unsafe_allow_html=True,
            )
            st.warning("既存のターンポイントを選択しています")

    st.caption("💡 地図をクリックして新しいポイントを追加。ナビゲーションの内容は、「左折」「やや左」「直進」「やや右」「右折」を推奨しますが、フリーワードです。「左」、「右」の文字を入れておくと逆走時に正しく変換されます")

# ─────────────────────────────────────────────
# ダウンロード
# ─────────────────────────────────────────────

st.divider()
st.subheader("💾 強化GPXの出力")

_applied = []
if st.session_state.get("_mm_status") == "完了":
    _applied.append("🗺️ マップマッチング済み")
_choice = st.session_state.get("_elev_choice")
if _choice == "fix" and st.session_state.get("_trkpt_fix_elevs") is not None:
    _applied.append("⛰️ 標高補正済み（国土地理院）")
elif _choice == "org" and st.session_state.get("_trkpt_org_elevs") is not None:
    _applied.append("⛰️ 標高補正済み（元データ）")
if _applied:
    st.info("出力GPXに適用: " + " ／ ".join(_applied))

col_dl1, col_dl2, _ = st.columns([1, 1, 2])
col_dl1.metric("ターンポイント数", len(current_turns))

with col_dl2:
    if st.button("📥 強化GPXを生成", type="primary", disabled=(len(current_turns) == 0)):
        turns_for_build = []
        for t in current_turns:
            tc = dict(t)
            widget_key = f"wpt_name_{t['index']}"
            tc["name"] = st.session_state.get(widget_key, t.get("name", "ターンポイント"))
            turns_for_build.append(tc)

        _out_elevs = (
            st.session_state.get("_trkpt_fix_elevs") if _choice == "fix"
            else st.session_state.get("_trkpt_org_elevs")
        )
        xml_output = build_enhanced_gpx(
            raw_content,
            turns_for_build,
            matched_points=st.session_state.get("_matched_points"),
            elevations=_out_elevs,
        )
        base_name = uploaded.name.replace(".gpx", "")
        st.download_button(
            label=f"⬇️ {base_name}_turns.gpx をダウンロード",
            data=xml_output,
            file_name=f"{base_name}_turns.gpx",
            mime="application/gpx+xml",
        )
        st.success(f"✅ {len(turns_for_build)} 個のターンポイントを埋め込みました")
