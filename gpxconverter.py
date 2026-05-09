"""
GPX ターン検出・強化ツール
点X の前後の点 A,B のベアリング差でコーナーを検出する
手動編集機能付き（追加・削除・名前変更）
マップマッチング（Valhalla）・標高補正（国土地理院 / Open-Meteo）対応
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
# マップマッチング（Valhalla）
# OSM公式インスタンス。自転車・徒歩・車すべて1サーバーで対応。
# ─────────────────────────────────────────────

_VALHALLA_URL = "https://valhalla1.openstreetmap.de/trace_attributes"

_VALHALLA_COSTING = {
    "cycling": "bicycle",
    "foot":    "pedestrian",
    "driving": "auto",
}

def _valhalla_match_chunk(chunk, costing, search_radius):
    """Valhalla trace_attributes で1チャンクをスナップ"""
    resp = requests.post(
        _VALHALLA_URL,
        json={
            "shape":         [{"lat": lat, "lon": lon} for lat, lon in chunk],
            "costing":       costing,
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
# マップマッチング（Google Maps Roads API）
# ─────────────────────────────────────────────

_GOOGLE_ROADS_URL = "https://roads.googleapis.com/v1/snapToRoads"

def _google_roads_match_chunk(chunk, api_key):
    """Google Roads API snapToRoads で1チャンクをスナップ（最大100点）"""
    path = "|".join(f"{lat},{lon}" for lat, lon in chunk)
    resp = requests.get(
        _GOOGLE_ROADS_URL,
        params={"path": path, "interpolate": "false", "key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ─────────────────────────────────────────────
# 標高補正（国土地理院 / Open-Meteo）
# ─────────────────────────────────────────────

def _is_in_japan(lat, lon):
    return 24.0 <= lat <= 46.0 and 122.0 <= lon <= 154.0

def _fetch_gsi_elevation(lat, lon):
    """国土地理院 標高API（1点）。取得失敗・海洋部(-9999)はNoneを返す"""
    resp = requests.get(
        "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php",
        params={"lat": lat, "lon": lon, "outtype": "JSON"},
        timeout=12,
    )
    resp.raise_for_status()
    data = resp.json()
    elev = data.get("elevation")
    return None if (elev is None or elev == -9999) else float(elev)

def _fetch_openmeteo_batch(batch):
    """Open-Meteo elevation API（最大100点バッチ）"""
    resp = requests.get(
        "https://api.open-meteo.com/v1/elevation",
        params={
            "latitude":  ",".join(f"{lat:.6f}" for lat, lon in batch),
            "longitude": ",".join(f"{lon:.6f}" for lat, lon in batch),
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("elevation", [None] * len(batch))

def fetch_all_elevations(points, source="auto"):
    """
    全trkptの標高を取得する。
    source: "auto" | "gsi" | "openmeteo"
    戻り値: (elevations: list[float|None], source_used: str, n_ok: int)
    """
    n          = len(points)
    elevations = [None] * n
    in_japan   = _is_in_japan(points[0][0], points[0][1]) if points else False
    use_gsi    = (source == "gsi") or (source == "auto" and in_japan)
    src_label  = "国土地理院" if use_gsi else "Open-Meteo"

    prog = st.progress(0, text=f"標高データ取得中（{src_label}）…")

    if use_gsi:
        # 国土地理院: スレッドローカルSessionでTCP接続を再利用しながら並列リクエスト
        _tls = threading.local()
        done = [0]

        def _fetch_one(args):
            i, lat, lon = args
            if not hasattr(_tls, "session"):
                _tls.session = requests.Session()
            try:
                resp = _tls.session.get(
                    "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php",
                    params={"lat": lat, "lon": lon, "outtype": "JSON"},
                    timeout=12,
                )
                resp.raise_for_status()
                data = resp.json()
                elev = data.get("elevation")
                return i, (None if (elev is None or elev == -9999) else float(elev))
            except Exception:
                return i, None

        tasks = [(i, lat, lon) for i, (lat, lon) in enumerate(points)]
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(_fetch_one, t): t[0] for t in tasks}
            for future in as_completed(futures):
                i, elev = future.result()
                elevations[i] = elev
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == n:
                    prog.progress(done[0] / n,
                                  text=f"標高取得中（国土地理院）… {done[0]}/{n} 点")

        # GSIで取れなかった点は元のGPX標高を保持（None のまま = build_enhanced_gpx が元値を使う）
    else:
        # Open-Meteo: バッチ処理（最大100点/リクエスト）
        BATCH = 100
        for b in range(0, n, BATCH):
            batch = points[b:b + BATCH]
            try:
                batch_e = _fetch_openmeteo_batch(batch)
                for j, elev in enumerate(batch_e):
                    elevations[b + j] = elev
            except Exception:
                pass
            prog.progress(min(b + BATCH, n) / n,
                          text=f"標高取得中（Open-Meteo）… {min(b+BATCH, n)}/{n} 点")

    prog.empty()
    n_ok = sum(1 for e in elevations if e is not None)
    return elevations, src_label, n_ok

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
    MIN_ELEVATION_JUMP_M = 2.0  # 距離を加味しない高低差の条件は不要かもしれない
    NEAR_ELEVATION_JUMP_M = 3.0
    CLUSTER_GAP_M = cluster_gap_m
    MERGE_GAP_M = 50.0
    MAX_ANCHOR_SEARCH_M = 600.0
    ANCHOR_GRADE_LIMIT = 12.0
    BOUNDARY_GRADE_LIMIT = 13.0
    MEDIAN_WINDOW_M = 150.0
    ANCHOR_MEDIAN_DEV_M = 5.0
    MAX_ANCHOR_GRADE = 15.0

    cleaned = list(elevations)
    cum_dists = _cumulative_distances(points)
    grades = _elevation_grades(points, cleaned, cum_dists)
    max_grade_before = max((abs(g) for g in grades if g is not None), default=0.0)

    bad_segments = []
    for i, grade in enumerate(grades):
        if grade is None or cleaned[i] is None or cleaned[i + 1] is None:
            continue
        dz = cleaned[i + 1] - cleaned[i]
        if (
            abs(grade) >= BAD_GRADE_THRESHOLD and abs(dz) >= MIN_ELEVATION_JUMP_M
        ) or (
            abs(grade) >= HARD_SPIKE_THRESHOLD and abs(dz) >= NEAR_ELEVATION_JUMP_M
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
            if (
                abs(seg_center - center) <= CLUSTER_GAP_M
                and abs(grade) >= NEAR_BAD_THRESHOLD
                and abs(dz) >= NEAR_ELEVATION_JUMP_M
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
        left_anchor = start_pt if is_left_boundary_anchor(start_pt) else find_anchor(start_pt - 1, -1)
        right_anchor = end_pt if is_right_boundary_anchor(end_pt) else find_anchor(end_pt + 1, 1)
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
            if abs(cleaned[i] - new_ele) >= 0.5:
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
    _gkey_pre      = st.secrets.get("GOOGLE_ROADS_API_KEY", None)
    _plabels_pre   = ["Valhalla（OSM公開API）", "Google Maps Roads API"]
    _pprov_pre     = st.session_state.get("_mm_provider", "valhalla")
    _pidx_pre      = 1 if _pprov_pre == "google" else 0
    if _gkey_pre and _is_admin:
        _sel_pre = st.radio(
            "🗺️ マップマッチング プロバイダ",
            _plabels_pre, index=_pidx_pre, horizontal=True,
        )
        st.session_state["_mm_provider"] = "google" if _sel_pre == _plabels_pre[1] else "valhalla"
    else:
        st.session_state["_mm_provider"] = "valhalla"
        st.caption("マップマッチング: Valhalla（OSM公開API）")
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
    "_elevations",    "_elev_status", "_elev_source", "_elev_n_ok", "_elev_clean_stats",
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
_mm_ran      = False

if st.session_state.get("_mm_status") is None:
    if _has_wpts and not st.session_state.pop("_force_mm", False):
        st.session_state["_matched_points"]         = list(points)
        st.session_state["_mm_status"]              = "スキップ"
    else:
        st.session_state["_mm_status"]              = "running"
        st.session_state["_mm_chunk_idx"]           = 0
        st.session_state["_mm_matched_partial"]     = list(points)
        st.session_state["_mm_n_snapped_partial"]   = 0
        st.session_state["_mm_errors_partial"]      = []

if st.session_state.get("_mm_status") == "running":
    _cur_provider = st.session_state.get("_mm_provider", "valhalla")
    _MM_CHUNK   = 100 if _cur_provider == "google" else 50
    _profile    = st.session_state.get("_mm_profile", "cycling")
    _radius     = st.session_state.get("_mm_radius", 50)
    _costing    = _VALHALLA_COSTING.get(_profile, "bicycle")
    _ci         = st.session_state["_mm_chunk_idx"]
    _n_chunks   = math.ceil(len(points) / _MM_CHUNK)
    _cancelled  = st.session_state.pop("_mm_cancel_requested", False)

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
            if _cur_provider == "google":
                _gkey = st.secrets.get("GOOGLE_ROADS_API_KEY", "")
                _data = _google_roads_match_chunk(points[_s:_e], _gkey)
                for _sp in _data.get("snappedPoints", []):
                    _orig = _sp.get("originalIndex")
                    if _orig is not None and _s + _orig < len(_mp_list):
                        _loc = _sp["location"]
                        _mp_list[_s + _orig] = (_loc["latitude"], _loc["longitude"])
                        st.session_state["_mm_n_snapped_partial"] += 1
            else:
                _data = _valhalla_match_chunk(points[_s:_e], _costing, _radius)
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
            _mm_ran      = True
            _needs_rerun = True
        else:
            st.session_state["_mm_chunk_idx"] = _ci + 1
            st.rerun()

active_points = st.session_state.get("_matched_points", points)

# MM 再実行時、edit_turns の wpt 座標を新しい trkpt 座標に同期する
if _mm_ran and "edit_turns" in st.session_state:
    for t in st.session_state["edit_turns"]:
        idx = t.get("index", 0)
        if idx < len(active_points):
            t["lat"] = active_points[idx][0]
            t["lon"] = active_points[idx][1]

if st.session_state.get("_elev_status") is None:
    if _has_wpts and not st.session_state.pop("_force_elev", False):
        st.session_state["_elev_status"] = "スキップ"
    else:
        _src = st.session_state.get("_elev_src", "auto")
        with st.spinner("⛰️ 標高補正処理中…"):
            elevs, src_used, n_ok = fetch_all_elevations(active_points, source=_src)
            elevs, clean_stats = clean_elevation_spikes(
                active_points, elevs,
                bad_grade_threshold=st.session_state.get("_elev_bad_grade", 15.0),
                cluster_gap_m=st.session_state.get("_elev_cluster_gap", 250.0),
            )
        st.session_state["_elevations"]  = elevs
        st.session_state["_elev_source"] = src_used
        st.session_state["_elev_n_ok"]   = n_ok
        st.session_state["_elev_clean_stats"] = clean_stats
        st.session_state["_elev_status"] = "完了" if n_ok > 0 else "エラー"
        _needs_rerun = True

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
        st.session_state["edit_turns"]   = turns
        st.session_state["_iname_status"] = "スキップ"
    else:
        _mta = st.session_state.get("_mta", 45)
        _md  = st.session_state.get("_md",  100)
        _sm  = st.session_state.get("_sm",  1)
        raw_turns = detect_turns(active_points, min_turn_angle=_mta, min_dist=_md, smooth=_sm)
        intersection_names = fetch_intersection_names(raw_turns)
        st.session_state["edit_turns"] = [
            with_name(t, intersection_names.get(t["index"]))
            for t in raw_turns
        ]
        st.session_state["_iname_status"]  = "完了"
        st.session_state["_iname_n_found"] = len(intersection_names)

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
# サイドバー ─ 交差点名取得
# ─────────────────────────────────────────────

st.sidebar.header("🏷️ 交差点名取得")

iname_status = st.session_state.get("_iname_status")
if iname_status == "完了":
    n_found = st.session_state.get("_iname_n_found", 0)
    st.sidebar.success(f"✅ {n_found} 件取得済み")
elif iname_status == "エラー":
    st.sidebar.error("❌ 取得失敗")
elif iname_status == "スキップ":
    st.sidebar.info("⏭️ スキップ（wpt読み込みモード）")
else:
    st.sidebar.info("⏳ 未取得")

_cur_iname_radius = st.session_state.get("_iname_radius", 20)
_iname_radius = st.sidebar.slider("検索半径（m）", 10, 100, _cur_iname_radius, 5,
                                   help="交差点ノードを探索する半径。20m推奨")
st.session_state["_iname_radius"] = _iname_radius

if iname_status is not None:
    st.sidebar.caption("ターンポイント付近の交差点名をOSMから取得します")
    if st.sidebar.button("🔄 再取得", key="iname_reset"):
        _new_inames = fetch_intersection_names(
            st.session_state["edit_turns"], radius=_iname_radius
        )
        for t in st.session_state["edit_turns"]:
            if t["index"] in _new_inames:
                new_name = with_name(t, _new_inames[t["index"]])["name"]
                t["name"] = new_name
                st.session_state[f"wpt_name_{t['index']}"] = new_name
        st.session_state["_iname_status"]         = "完了"
        st.session_state["_iname_n_found"]        = len(_new_inames)
        st.session_state["_skip_map_center_save"] = True
        st.rerun()

# ─────────────────────────────────────────────
# サイドバー ─ ターン検出パラメータ
# ─────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.header("⚙️ ターン検出パラメータ")
st.sidebar.markdown("""
**アルゴリズム**: 角度法
点 X の前後の点 A, B のベアリング差でコーナーを判定します。
""")

min_turn_angle = st.sidebar.slider(
    "ターン角閾値（度）", 20, 120, st.session_state.get("_mta", 45), 5,
    help="進入・離脱方向の差がこの角度以上ならコーナーとみなす。\n"
         "45°=やや曲がりも検出、60°=交差点のみ、90°=ほぼ直角以上のみ")
min_dist_val = st.sidebar.slider(
    "ターン間最小距離（m）", 30, 500, st.session_state.get("_md", 100), 10,
    help="同一交差点での重複検出を防ぐ")
smooth_val = st.sidebar.slider(
    "スムージング（前後N点参照）", 1, 5, st.session_state.get("_sm", 1), 1,
    help="1=隣接点のみ（推奨）、2以上=ノイズに強いが精度低下の可能性あり")
st.session_state["_mta"] = min_turn_angle
st.session_state["_md"]  = min_dist_val
st.session_state["_sm"]  = smooth_val

if st.sidebar.button("🔄 自動検出を再実行（現在のターンポイントは破棄されます）", type="primary"):
    raw_turns = detect_turns(active_points, min_turn_angle=min_turn_angle,
                             min_dist=min_dist_val, smooth=smooth_val)
    intersection_names = fetch_intersection_names(raw_turns)
    st.session_state["edit_turns"] = [
        with_name(t, intersection_names.get(t["index"]))
        for t in raw_turns
    ]
    st.session_state["_iname_status"]         = "完了"
    st.session_state["_iname_n_found"]        = len(intersection_names)
    st.session_state.pop("pending_wpt", None)
    st.session_state["_skip_map_center_save"] = True
    st.rerun()

# ─────────────────────────────────────────────
# サイドバー ─ マップマッチング（Valhalla）
# ─────────────────────────────────────────────

_MM_PROFILES = {
    "自転車 (bicycle)": "cycling",
    "徒歩・ハイキング":  "foot",
    "車 (auto)":       "driving",
}
_mm_labels = list(_MM_PROFILES.keys())
_mm_codes  = list(_MM_PROFILES.values())
_cur_mm_code = st.session_state.get("_mm_profile", "cycling")
_cur_mm_idx  = _mm_codes.index(_cur_mm_code) if _cur_mm_code in _mm_codes else 0

st.sidebar.divider()
st.sidebar.header("🗺️ マップマッチング")

mm_status = st.session_state.get("_mm_status")
if mm_status == "完了":
    n_snapped = st.session_state.get("_mm_n_snapped", 0)
    st.sidebar.success(f"✅ {n_snapped}/{len(points)} 点スナップ済み")
    if st.session_state.get("_mm_error"):
        st.sidebar.caption(f"⚠️ {st.session_state['_mm_error'][:120]}")
elif mm_status == "エラー":
    st.sidebar.error(f"❌ マッチング失敗\n{st.session_state.get('_mm_error','')[:200]}")
elif mm_status == "スキップ":
    st.sidebar.info("⏭️ スキップ（wpt読み込みモード）")
elif mm_status == "キャンセル":
    n_snapped = st.session_state.get("_mm_n_snapped", 0)
    st.sidebar.warning(f"⚠️ キャンセル済み（{n_snapped} 点スナップ）")
    if st.session_state.get("_mm_error"):
        st.sidebar.caption(st.session_state["_mm_error"][:120])
else:
    st.sidebar.info("⏳ 処理中…")

# プロバイダ選択
_google_key       = st.secrets.get("GOOGLE_ROADS_API_KEY", None)
_google_available = bool(_google_key)
_provider_labels  = ["Valhalla（OSM公開API）", "Google Maps Roads API"]
_cur_provider     = st.session_state.get("_mm_provider", "valhalla")
_provider_idx     = 1 if _cur_provider == "google" else 0

if not _google_available or not _is_admin:
    st.sidebar.radio("プロバイダ", [_provider_labels[0]], index=0, disabled=True)
    st.session_state["_mm_provider"] = "valhalla"
else:
    _sel_provider = st.sidebar.radio("プロバイダ", _provider_labels, index=_provider_idx)
    _new_provider = "google" if _sel_provider == _provider_labels[1] else "valhalla"
    if _new_provider != _cur_provider:
        for _k in ["_matched_points", "_mm_status", "_mm_n_snapped", "_mm_error",
                   "_mm_chunk_idx", "_mm_matched_partial", "_mm_n_snapped_partial",
                   "_mm_errors_partial", "_mm_cancel_requested"]:
            st.session_state.pop(_k, None)
    st.session_state["_mm_provider"] = _new_provider
    _cur_provider = _new_provider

if _cur_provider == "valhalla":
    _sel_mm = st.sidebar.selectbox("プロファイル", _mm_labels, index=_cur_mm_idx)
    st.session_state["_mm_profile"] = _MM_PROFILES[_sel_mm]
    _cur_radius = st.session_state.get("_mm_radius", 50)
    _mm_radius  = st.sidebar.slider("サーチ半径（m）", 10, 100, _cur_radius, 10,
                                     help="道路を探索する半径。GPS誤差が大きいルートは大きくする")
    st.session_state["_mm_radius"] = _mm_radius
else:
    st.sidebar.caption(f"APIキー: ✅ 設定済み")

if mm_status is not None:
    st.sidebar.caption("設定を変えた後は再処理ボタンを押してください")
    if st.sidebar.button("🔄 再処理", key="mm_reset"):
        for k in ["_matched_points", "_mm_status", "_mm_n_snapped", "_mm_error",
                  "_mm_chunk_idx", "_mm_matched_partial", "_mm_n_snapped_partial", "_mm_errors_partial",
                  "_mm_cancel_requested", "pending_wpt"]:
            st.session_state.pop(k, None)
        st.session_state["_force_mm"]             = True
        st.session_state["_skip_map_center_save"] = True
        st.rerun()

# ─────────────────────────────────────────────
# サイドバー ─ 標高補正
# ─────────────────────────────────────────────

_ELEV_SOURCES = {
    "自動（日本→国土地理院、海外→Open-Meteo）": "auto",
    "国土地理院（日本専用・高精度）":             "gsi",
    "Open-Meteo（全世界）":                    "openmeteo",
}
_elev_labels = list(_ELEV_SOURCES.keys())
_elev_codes  = list(_ELEV_SOURCES.values())
_cur_src     = st.session_state.get("_elev_src", "auto")
_cur_src_idx = _elev_codes.index(_cur_src) if _cur_src in _elev_codes else 0

st.sidebar.divider()
st.sidebar.header("⛰️ 標高補正")

elev_status = st.session_state.get("_elev_status")
if elev_status == "完了":
    n_ok = st.session_state.get("_elev_n_ok", 0)
    src  = st.session_state.get("_elev_source", "")
    st.sidebar.success(f"✅ {n_ok}/{len(active_points)} 点取得（{src}）")
    clean_stats = st.session_state.get("_elev_clean_stats", {})
    clean_points = clean_stats.get("points", 0)
    if clean_points:
        st.sidebar.caption(
            f"スパイク除去: {clean_stats.get('clusters', 0)} 箇所 / {clean_points} 点補正 "
            f"（最大勾配 {clean_stats.get('max_grade_before', 0):.1f}% → "
            f"{clean_stats.get('max_grade_after', 0):.1f}%）"
        )
    else:
        st.sidebar.caption("スパイク除去: 補正対象なし")
elif elev_status == "エラー":
    st.sidebar.warning("⚠️ 一部取得失敗")
elif elev_status == "スキップ":
    st.sidebar.info("⏭️ スキップ（wpt読み込みモード）")
else:
    st.sidebar.info("⏳ 処理中…")

in_japan_hint = _is_in_japan(active_points[0][0], active_points[0][1])
st.sidebar.caption(f"ルート判定: {'🇯🇵 日本' if in_japan_hint else '🌍 海外'}")

_sel_src = st.sidebar.selectbox("データソース", _elev_labels, index=_cur_src_idx)
st.session_state["_elev_src"] = _ELEV_SOURCES[_sel_src]

_elev_bad_grade = st.sidebar.slider(
    "スパイク判定 勾配閾値（%）", 5, 25, int(st.session_state.get("_elev_bad_grade", 15)), 1,
    help="この勾配以上 かつ 高度変化6m以上のセグメントをスパイク候補とみなす")
st.session_state["_elev_bad_grade"] = float(_elev_bad_grade)

_elev_cluster_gap = st.sidebar.slider(
    "スパイク判定 クラスターギャップ（m）", 50, 500, int(st.session_state.get("_elev_cluster_gap", 250)), 50,
    help="スパイク候補同士をまとめる最大距離。大きくすると広範囲のスパイクをまとめて修正")
st.session_state["_elev_cluster_gap"] = float(_elev_cluster_gap)

if elev_status is not None:
    st.sidebar.caption("設定を変えた後は再処理ボタンを押してください")
    if st.sidebar.button("🔄 再処理", key="elev_reset"):
        for k in ["_elevations", "_elev_status", "_elev_source", "_elev_n_ok", "_elev_clean_stats"]:
            st.session_state.pop(k, None)
        st.session_state["_force_elev"]           = True
        st.session_state["_skip_map_center_save"] = True
        st.rerun()

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
    _cur_provider  = st.session_state.get("_mm_provider", "valhalla")
    if _cur_provider == "google":
        m = folium.Map(location=_map_init_loc, zoom_start=_map_init_zoom, tiles=None)
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            attr='© <a href="https://www.google.com/maps">Google Maps</a>',
            name="Google Maps", max_zoom=20,
        ).add_to(m)
    else:
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
                sm = smooth_val
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
                _iname_radius = st.session_state.get("_iname_radius", 20)
                _inames = fetch_intersection_names([_temp], radius=_iname_radius)
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

applied = []
if st.session_state.get("_mm_status") == "完了":
    applied.append("🗺️ マップマッチング済み")
if st.session_state.get("_elev_status") == "完了":
    applied.append(f"⛰️ 標高補正済み（{st.session_state.get('_elev_source', '')}）")
if applied:
    st.info("出力GPXに適用: " + " ／ ".join(applied))

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

        xml_output = build_enhanced_gpx(
            raw_content,
            turns_for_build,
            matched_points=st.session_state.get("_matched_points"),
            elevations=st.session_state.get("_elevations"),
        )
        base_name = uploaded.name.replace(".gpx", "")
        st.download_button(
            label=f"⬇️ {base_name}_turns.gpx をダウンロード",
            data=xml_output,
            file_name=f"{base_name}_turns.gpx",
            mime="application/gpx+xml",
        )
        st.success(f"✅ {len(turns_for_build)} 個のターンポイントを埋め込みました")
