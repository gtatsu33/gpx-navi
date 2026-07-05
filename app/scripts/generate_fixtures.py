"""
ゴールデンファイル生成スクリプト（Phase 1用）

gpxconverter.py の対象関数の「本体をそのままコピー」して実行し、
JS移植（app/src/lib/*.js）の突き合わせ用フィクスチャをJSON出力する。
gpxconverter.py 自体は import せず、Streamlitへの依存を持ち込まない。

実行方法: python generate_fixtures.py
出力先:   app/src/__fixtures__/*.json
"""
import json
import math
import os

import numpy as np
from rdp import rdp as rdp_simplify

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "src", "__fixtures__")


# ─────────────────────────────────────────────
# gpxconverter.py からのコピー（geo.js 対応）
# ─────────────────────────────────────────────

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def angle_diff(a, b):
    return (b - a + 180) % 360 - 180

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return R * 2 * math.asin(math.sqrt(max(0, a)))


# ─────────────────────────────────────────────
# gpxconverter.py からのコピー（turns.js 対応）
# ─────────────────────────────────────────────

def detect_turns(points, min_turn_angle=45, min_dist=100, smooth=1):
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


# ─────────────────────────────────────────────
# gpxconverter.py からのコピー（elevation.js 対応）
# ─────────────────────────────────────────────

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
    n = len(points)
    if n < 4 or not elevations or len(elevations) != n:
        return elevations, {"clusters": 0, "points": 0, "max_grade_before": 0.0, "max_grade_after": 0.0}

    BAD_GRADE_THRESHOLD = bad_grade_threshold
    HARD_SPIKE_THRESHOLD = 35.0
    NEAR_BAD_THRESHOLD = 15.0
    MIN_ELEVATION_JUMP_M = 2.0
    NEAR_ELEVATION_JUMP_M = 3.0
    SHORT_SEG_M = 10.0
    CLUSTER_GAP_M = cluster_gap_m
    MERGE_GAP_M = 50.0
    MAX_ANCHOR_SEARCH_M = 600.0
    ANCHOR_GRADE_LIMIT = 12.0
    BOUNDARY_GRADE_LIMIT = 13.0
    MEDIAN_WINDOW_M = 150.0
    ANCHOR_MEDIAN_DEV_M = 5.0
    MAX_ANCHOR_GRADE = 15.0
    MIN_CORRECTION_GRADE_PCT = 1.0

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
    grades = _elevation_grades(points, elevations)
    valid = [g for g in grades if g is not None]
    if not valid:
        return None
    return {
        "max": max((g for g in valid if g > 0), default=0.0),
        "min": min((g for g in valid if g < 0), default=0.0),
    }


# ─────────────────────────────────────────────
# サンプルデータ
# ─────────────────────────────────────────────

def make_geo_fixture():
    pairs = [
        (35.6812, 139.7671, 35.6586, 139.7454),  # 東京駅→東京タワー付近
        (35.1709, 136.8815, 35.1709, 136.9000),  # 名古屋、東方向
        (43.0687, 141.3508, 42.9849, 141.3936),  # 札幌→南東
        (35.6586, 139.7454, 35.6586, 139.7454),  # 同一点（距離0）
    ]
    cases = []
    for lat1, lon1, lat2, lon2 in pairs:
        cases.append({
            "input": {"lat1": lat1, "lon1": lon1, "lat2": lat2, "lon2": lon2},
            "bearing": calculate_bearing(lat1, lon1, lat2, lon2),
            "haversine": haversine(lat1, lon1, lat2, lon2),
        })
    angle_pairs = [(10, 350), (350, 10), (0, 180), (179, -179), (-45, 45)]
    angle_cases = [{"input": {"a": a, "b": b}, "output": angle_diff(a, b)} for a, b in angle_pairs]
    return {"bearingHaversine": cases, "angleDiff": angle_cases}


def _rect_route(lat0=35.0, lon0=139.0, step=0.001, n_side=8):
    """矩形に近い経路（直角ターンを複数含む）を生成する。"""
    pts = []
    lat, lon = lat0, lon0
    for _ in range(n_side):
        lat += step
        pts.append((lat, lon))
    for _ in range(n_side):
        lon += step
        pts.append((lat, lon))
    for _ in range(n_side):
        lat -= step
        pts.append((lat, lon))
    for _ in range(n_side):
        lon -= step
        pts.append((lat, lon))
    return pts


def make_turns_fixture():
    points = _rect_route()
    turns = detect_turns(points, min_turn_angle=45, min_dist=100, smooth=1)
    label_inputs = [70, 40, 10, -10, -40, -70]
    return {
        "detectTurns": {
            "input": {"points": [list(p) for p in points], "min_turn_angle": 45, "min_dist": 100, "smooth": 1},
            "output": turns,
        },
        "turnLabel": [{"input": d, "output": list(turn_label(d))} for d in label_inputs],
    }


def make_rdp_fixture():
    # ほぼ直線上の点＋明らかな外れ点を混ぜて間引き対象を作る
    points = [[35.0 + i * 0.0001, 139.0] for i in range(20)]
    points[10][1] += 0.01  # 大きく外れた点（残るはず）
    mask = rdp_simplify(points, epsilon=0.00005, return_mask=True)
    return {
        "input": {"points": points, "epsilon": 0.00005},
        "mask": [bool(m) for m in mask],
    }


def make_elevation_fixture():
    n = 30
    points = [(35.0 + i * 0.001, 139.0) for i in range(n)]
    elevations = [100.0 + i * 0.5 for i in range(n)]
    # 局所スパイクを注入（周囲となじまない急激な値）
    elevations[15] = elevations[15] + 80.0
    cleaned, stats = clean_elevation_spikes(points, elevations)
    grade_stats = compute_grade_stats(points, cleaned)
    return {
        "input": {"points": [list(p) for p in points], "elevations": elevations},
        "cleaned": cleaned,
        "stats": stats,
        "gradeStats": grade_stats,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fixtures = {
        "geo.json": make_geo_fixture(),
        "turns.json": make_turns_fixture(),
        "rdp.json": make_rdp_fixture(),
        "elevation.json": make_elevation_fixture(),
    }
    for filename, data in fixtures.items():
        path = os.path.join(OUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
