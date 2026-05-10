"""
SpikeFixer.py  -  スパイク補正スタンドアロンツール

gpxconverter.py の clean_elevation_spikes を単独で実行する。
調整・デバッグ後、最終的に gpxconverter.py へ書き戻す想定。

使い方:
  python SpikeFixer.py input.gpx output.gpx compare.txt
  python SpikeFixer.py input.gpx output.gpx compare.txt --threshold 12.0 --gap 200.0

オプション:
  --threshold  バッドセグメント判定の勾配閾値(%) デフォルト 15.0
  --gap        クラスタ分割距離(m)               デフォルト 250.0
"""

import sys
import math
import time
import argparse
import numpy as np
import gpxpy
import gpxpy.gpx

# ═══════════════════════════════════════════════════════════════════
#  gpxconverter.py からそのままコピーした関数群
#  ※ この区間は gpxconverter.py と完全に同一に保つこと
# ═══════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return R * 2 * math.asin(math.sqrt(max(0, a)))

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

# ═══════════════════════════════════════════════════════════════════
#  ここから下は SpikeFixer.py 独自のコード
# ═══════════════════════════════════════════════════════════════════

def _analyze_spikes(points, elevations, bad_grade_threshold=15.0, cluster_gap_m=250.0):
    """
    clean_elevation_spikes と同じロジックを再実行し、内部状態を診断データとして返す。
    compare ファイル生成専用。補正は行わない。
    """
    n = len(points)

    BAD_GRADE_THRESHOLD  = bad_grade_threshold
    HARD_SPIKE_THRESHOLD = 35.0
    NEAR_BAD_THRESHOLD   = 15.0
    MIN_ELEVATION_JUMP_M = 2.0
    NEAR_ELEVATION_JUMP_M = 3.0
    SHORT_SEG_M           = 10.0
    CLUSTER_GAP_M         = cluster_gap_m
    MERGE_GAP_M           = 50.0
    MAX_ANCHOR_SEARCH_M   = 600.0
    ANCHOR_GRADE_LIMIT    = 12.0
    BOUNDARY_GRADE_LIMIT  = 13.0
    MEDIAN_WINDOW_M       = 150.0
    ANCHOR_MEDIAN_DEV_M   = 5.0
    MAX_ANCHOR_GRADE      = 15.0

    cleaned   = list(elevations)
    cum_dists = _cumulative_distances(points)
    grades    = _elevation_grades(points, cleaned, cum_dists)

    # --- フェーズ1: バッドセグメント検出 ---
    bad_segs     = []   # セグメントインデックスのリスト
    seg_type     = {}   # seg_idx -> "通常スパイク" / "通常スパイク(短距離)" / "ハードスパイク" / "ハードスパイク(短距離)"
    for i, grade in enumerate(grades):
        if grade is None or cleaned[i] is None or cleaned[i+1] is None:
            continue
        dz    = cleaned[i+1] - cleaned[i]
        dist  = cum_dists[i+1] - cum_dists[i]
        short = dist < SHORT_SEG_M
        is_hard = abs(grade) >= HARD_SPIKE_THRESHOLD and (short or abs(dz) >= NEAR_ELEVATION_JUMP_M)
        is_norm = abs(grade) >= BAD_GRADE_THRESHOLD  and (short or abs(dz) >= MIN_ELEVATION_JUMP_M)
        if is_hard or is_norm:
            bad_segs.append(i)
            if is_hard:
                seg_type[i] = "ハードスパイク(短距離)" if short else "ハードスパイク"
            else:
                seg_type[i] = "通常スパイク(短距離)"  if short else "通常スパイク"

    # --- フェーズ2: 近傍セグメント取り込み ---
    near_segs = set(bad_segs)
    for bad_idx in bad_segs:
        center = (cum_dists[bad_idx] + cum_dists[bad_idx+1]) / 2
        for i, grade in enumerate(grades):
            if grade is None or cleaned[i] is None or cleaned[i+1] is None:
                continue
            seg_center = (cum_dists[i] + cum_dists[i+1]) / 2
            dz    = cleaned[i+1] - cleaned[i]
            short = (cum_dists[i+1] - cum_dists[i]) < SHORT_SEG_M
            if (abs(seg_center - center) <= CLUSTER_GAP_M
                    and abs(grade) >= NEAR_BAD_THRESHOLD
                    and (short or abs(dz) >= NEAR_ELEVATION_JUMP_M)):
                near_segs.add(i)
                if i not in seg_type:
                    seg_type[i] = "近傍"

    # --- フェーズ3: クラスタリング ---
    raw_clusters = _cluster_segments(sorted(near_segs), cum_dists, CLUSTER_GAP_M)

    def is_anchor_candidate(i):
        if i <= 0 or i >= n-1 or cleaned[i] is None:
            return False
        pg, ng = grades[i-1], grades[i]
        if pg is None or ng is None:
            return False
        if abs(pg) > ANCHOR_GRADE_LIMIT or abs(ng) > ANCHOR_GRADE_LIMIT:
            return False
        m = _local_median_elevation(i, cum_dists, cleaned, MEDIAN_WINDOW_M)
        return m is not None and abs(cleaned[i] - m) <= ANCHOR_MEDIAN_DEV_M

    def find_anchor(start_i, direction):
        start_dist = cum_dists[start_i]
        i = start_i
        run = []
        while 0 < i < n-1 and abs(cum_dists[i] - start_dist) <= MAX_ANCHOR_SEARCH_M:
            if is_anchor_candidate(i):
                run.append(i)
                if len(run) >= 2:
                    return run[0]
            else:
                run = []
            i += direction
        return None

    def is_left_boundary_anchor(i):
        return (0 < i < n-1 and cleaned[i] is not None
                and grades[i-1] is not None and abs(grades[i-1]) <= BOUNDARY_GRADE_LIMIT)

    def is_right_boundary_anchor(i):
        return (0 < i < n-1 and cleaned[i] is not None
                and grades[i] is not None and abs(grades[i]) <= BOUNDARY_GRADE_LIMIT)

    # --- フェーズ4: アンカー選定 + 修復範囲 ---
    repair_ranges = []
    for cluster in raw_clusters:
        start_pt = cluster["start_seg"]
        end_pt   = cluster["end_seg"] + 1
        la = (start_pt if (is_left_boundary_anchor(start_pt) and is_anchor_candidate(start_pt))
              else find_anchor(start_pt - 1, -1))
        ra = (end_pt if (is_right_boundary_anchor(end_pt) and is_anchor_candidate(end_pt))
              else find_anchor(end_pt + 1, 1))

        skip_reason = None
        net_grade   = None
        if la is None or ra is None or la >= ra:
            skip_reason = "アンカー見つからず"
        else:
            dist_m = cum_dists[ra] - cum_dists[la]
            if dist_m > 0 and cleaned[la] is not None and cleaned[ra] is not None:
                net_grade = (cleaned[ra] - cleaned[la]) / dist_m * 100
                if abs(net_grade) > MAX_ANCHOR_GRADE:
                    skip_reason = f"正味勾配過大 ({net_grade:+.2f}%)"

        repair_ranges.append({
            "cluster":     cluster,
            "left":        la,
            "right":       ra,
            "net_grade":   net_grade,
            "skip_reason": skip_reason,
        })

    # --- フェーズ5: 修復範囲マージ ---
    valid = [r for r in repair_ranges if r["skip_reason"] is None and r["left"] is not None]
    valid.sort(key=lambda r: r["left"])
    merged = []
    if valid:
        merged = [dict(valid[0])]
        for r in valid[1:]:
            prev    = merged[-1]
            gap_m   = cum_dists[r["left"]] - cum_dists[prev["right"]]
            if r["left"] <= prev["right"] or gap_m <= MERGE_GAP_M:
                prev["right"] = max(prev["right"], r["right"])
            else:
                merged.append(dict(r))

    # seg→クラスタID 逆引き
    seg_to_cl = {}
    for ci, r in enumerate(repair_ranges):
        for si in range(r["cluster"]["start_seg"], r["cluster"]["end_seg"] + 1):
            seg_to_cl[si] = ci

    return {
        "cum_dists":     cum_dists,
        "grades":        grades,
        "bad_segs":      set(bad_segs),
        "near_segs":     near_segs,
        "seg_type":      seg_type,
        "repair_ranges": repair_ranges,
        "merged":        merged,
        "seg_to_cl":     seg_to_cl,
    }


def _fmt_grade(g):
    if g is None:
        return "    -   "
    return f"{g:+8.2f}%"

def _fmt_ele(e):
    if e is None:
        return "    -  "
    return f"{e:7.1f}"

def _fmt_dist(d):
    if d is None or d < 0.01:
        return "      -  "
    return f"{d:8.1f}m"


def write_compare(path, input_path, output_path, points, elevs_before, elevs_after,
                  stats, analysis, threshold, gap):
    n = len(points)
    cum   = analysis["cum_dists"]
    gb    = analysis["grades"]
    ga    = _elevation_grades(points, elevs_after, cum)
    bad   = analysis["bad_segs"]
    near  = analysis["near_segs"]
    stype = analysis["seg_type"]
    rrs   = analysis["repair_ranges"]
    seg2cl= analysis["seg_to_cl"]

    # アンカーポイントの逆引き (pt_idx -> list of "左:C{n}" or "右:C{n}")
    pt_anchor = {}
    for ci, r in enumerate(rrs):
        if r["skip_reason"]:
            continue
        la, ra = r["left"], r["right"]
        if la is not None:
            pt_anchor.setdefault(la, []).append(f"左アンカー:C{ci+1}")
        if ra is not None:
            pt_anchor.setdefault(ra, []).append(f"右アンカー:C{ci+1}")

    corrected = set()
    for i in range(n):
        if elevs_before[i] != elevs_after[i] and elevs_after[i] is not None:
            corrected.add(i)

    SEP  = "=" * 78
    SEP2 = "-" * 78

    with open(path, "w", encoding="utf-8") as f:
        def w(s=""):
            f.write(s + "\n")

        w(SEP)
        w("  スパイク補正レポート")
        w(f"  実行日時  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"  入力      : {input_path}")
        w(f"  出力      : {output_path}")
        w(f"  総点数    : {n}")
        w(f"  閾値設定  : 勾配閾値={threshold}%  クラスタ距離={gap}m")
        w(SEP)

        w()
        w("【結果サマリー】")
        w(f"  補正前 最大勾配  : {stats['max_grade_before']:+.2f}%")
        w(f"  補正後 最大勾配  : {stats['max_grade_after']:+.2f}%")
        w(f"  補正クラスタ数   : {stats['clusters']}")
        w(f"  補正点数         : {stats['points']}")

        w()
        w("【バッドセグメント一覧】")
        if not bad:
            w("  (なし)")
        else:
            w(f"  {'seg':>4}  {'pt始→終':^11}  {'距離':>7}  {'高低差':>7}  {'勾配':>9}  分類")
            w(f"  {'-'*4}  {'-'*11}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*20}")
            for i in sorted(bad):
                if elevs_before[i] is None or elevs_before[i+1] is None:
                    continue
                d  = cum[i+1] - cum[i]
                dz = elevs_before[i+1] - elevs_before[i]
                g  = gb[i]
                w(f"  {i:>4}  pt{i:>4}→{i+1:<4}  {d:>6.1f}m  {dz:>+7.1f}m  {g:>+8.2f}%  {stype.get(i,'')}")

        w()
        w("【クラスタ・修復範囲】")
        if not rrs:
            w("  (クラスタなし)")
        for ci, r in enumerate(rrs):
            cl  = r["cluster"]
            la  = r["left"]
            ra  = r["right"]
            ng  = r["net_grade"]
            skp = r["skip_reason"]
            w(f"  クラスタ {ci+1}:")
            w(f"    バッドセグメント区間 : seg {cl['start_seg']} → {cl['end_seg']}"
              f"  (pt{cl['start_seg']} → pt{cl['end_seg']+1})")
            if skp:
                w(f"    修復                : スキップ ({skp})")
                if la is not None:
                    w(f"    左アンカー          : pt{la}  ele={_fmt_ele(elevs_before[la]).strip()}")
                else:
                    w(f"    左アンカー          : 見つからず")
                if ra is not None:
                    w(f"    右アンカー          : pt{ra}  ele={_fmt_ele(elevs_before[ra]).strip()}")
                else:
                    w(f"    右アンカー          : 見つからず")
            else:
                dist_m = cum[ra] - cum[la]
                w(f"    左アンカー          : pt{la}  ele={_fmt_ele(elevs_before[la]).strip()}")
                w(f"    右アンカー          : pt{ra}  ele={_fmt_ele(elevs_before[ra]).strip()}")
                w(f"    アンカー間距離      : {dist_m:.1f}m")
                w(f"    アンカー間正味勾配  : {ng:+.2f}%")
                pts_fixed = [i for i in range(la+1, ra) if i in corrected]
                w(f"    補正点              : {len(pts_fixed)}点  {pts_fixed}")

        w()
        w(SEP2)
        w("【点別データ】")
        w()
        w("  凡例:")
        w("    seg分類 : [ハード]=ハードスパイク  [ハード短]=ハードスパイク(短距離)")
        w("             [通常]=通常スパイク        [通常短]=通常スパイク(短距離)")
        w("             [近傍]=近傍セグメント      (空白)=正常")
        w("    pt属性  : [LA:Cn]=クラスタnの左アンカー  [RA:Cn]=クラスタnの右アンカー")
        w("             [補正]=標高値が変更された点")
        w()

        hdr = (f"  {'pt':>4}  {'前ele':>7}  {'後ele':>7}  {'変化':>6}  "
               f"{'区間距離':>8}  {'前勾配':>9}  {'後勾配':>9}  "
               f"{'seg分類':<18}  pt属性")
        w(hdr)
        w(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*6}  "
          f"{'-'*8}  {'-'*9}  {'-'*9}  "
          f"{'-'*18}  {'-'*20}")

        for i in range(n):
            eb   = elevs_before[i]
            ea   = elevs_after[i]
            delt = "" if (eb is None or ea is None) else f"{ea-eb:+.1f}"

            # 次点との区間情報（最終点はなし）
            if i < n - 1:
                d    = cum[i+1] - cum[i]
                d_s  = f"{d:8.1f}m"
                gb_s = _fmt_grade(gb[i])
                ga_s = _fmt_grade(ga[i])
                # seg分類
                if i in bad:
                    t = stype.get(i, "")
                    if "ハード" in t and "短距離" in t:
                        scls = "[ハード短]"
                    elif "ハード" in t:
                        scls = "[ハード]  "
                    elif "短距離" in t:
                        scls = "[通常短]  "
                    else:
                        scls = "[通常]    "
                elif i in near:
                    scls = "[近傍]    "
                else:
                    scls = "          "
                cl_tag = f" C{seg2cl[i]+1}" if i in seg2cl else ""
                scls = (scls + cl_tag).ljust(18)
            else:
                d_s  = "         -"
                gb_s = "         -"
                ga_s = "         -"
                scls = "".ljust(18)

            # pt属性
            flags = []
            if i in pt_anchor:
                flags.extend(pt_anchor[i])
            if i in corrected:
                flags.append("[補正]")
            flags_s = "  ".join(flags)

            w(f"  {i:>4}  {_fmt_ele(eb)}  {_fmt_ele(ea)}  {delt:>6}  "
              f"{d_s}  {gb_s}  {ga_s}  "
              f"{scls}  {flags_s}")

        w()
        w(SEP)
        w("  レポート終了")
        w(SEP)


def main():
    parser = argparse.ArgumentParser(
        description="GPX標高スパイク補正ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input",   help="補正前GPXファイル")
    parser.add_argument("output",  help="補正後GPXファイル")
    parser.add_argument("compare", help="比較テキストファイル")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="バッドセグメント判定の勾配閾値(%%) (デフォルト: 15.0)")
    parser.add_argument("--gap",       type=float, default=250.0,
                        help="クラスタ分割距離(m) (デフォルト: 250.0)")
    args = parser.parse_args()

    # --- 入力 ---
    print(f"読み込み: {args.input}")
    with open(args.input, encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    all_pts = [pt for tr in gpx.tracks for seg in tr.segments for pt in seg.points]
    points       = [(pt.latitude, pt.longitude) for pt in all_pts]
    elevs_before = [pt.elevation for pt in all_pts]
    n = len(points)
    print(f"  {n} 点")

    # --- 補正 ---
    print(f"スパイク補正中... (勾配閾値={args.threshold}%, クラスタ距離={args.gap}m)")
    elevs_after, stats = clean_elevation_spikes(
        points, elevs_before,
        bad_grade_threshold=args.threshold,
        cluster_gap_m=args.gap,
    )
    print(f"  補正前最大勾配: {stats['max_grade_before']:+.2f}%")
    print(f"  補正後最大勾配: {stats['max_grade_after']:+.2f}%")
    print(f"  クラスタ数: {stats['clusters']}  補正点数: {stats['points']}")

    # --- 出力GPX ---
    print(f"書き込み: {args.output}")
    for i, pt in enumerate(all_pts):
        pt.elevation = elevs_after[i]
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(gpx.to_xml())

    # --- compare ---
    print(f"レポート: {args.compare}")
    analysis = _analyze_spikes(
        points, elevs_before,
        bad_grade_threshold=args.threshold,
        cluster_gap_m=args.gap,
    )
    write_compare(
        args.compare, args.input, args.output,
        points, elevs_before, elevs_after,
        stats, analysis, args.threshold, args.gap,
    )

    print("完了")


if __name__ == "__main__":
    main()
