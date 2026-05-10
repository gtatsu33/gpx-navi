import sys
import math
import xml.etree.ElementTree as ET
from statistics import median
from datetime import datetime

# ====================================================================
#  定数一覧 (SpikeFix.txt より)
# ====================================================================
BAD_GRADE_THRESHOLD = 15.0      # 通常スパイク判定の勾配閾値 (%)
HARD_SPIKE_THRESHOLD = 35.0     # ハードスパイク判定の勾配閾値 (%)
NEAR_BAD_THRESHOLD = 15.0       # クラスタ拡張時の勾配閾値 (%)
MIN_ELEVATION_JUMP_M = 2.0      # 通常スパイク判定の高低差閾値 (m)
NEAR_ELEVATION_JUMP_M = 3.0     # ハード・拡張時の高低差閾値 (m)
SHORT_SEG_M = 10.0              # 高低差条件を免除する点間距離 (m)
CLUSTER_GAP_M = 250.0           # クラスタ拡張の走査距離 (m)
MERGE_GAP_M = 50.0              # 修復範囲をマージする距離 (m)
MAX_ANCHOR_SEARCH_M = 600.0     # アンカー探索の最大距離 (m)
ANCHOR_GRADE_LIMIT = 12.0       # アンカー候補の隣接勾配上限 (%)
BOUNDARY_GRADE_LIMIT = 13.0     # 境界アンカー高速判定の勾配上限 (%)
MEDIAN_WINDOW_M = 150.0         # 局所中央値の計算ウィンドウ (m)
ANCHOR_MEDIAN_DEV_M = 5.0       # アンカー候補の中央値からの許容偏差 (m)
MAX_ANCHOR_GRADE = 15.0         # アンカー間の正味勾配上限 (%)

def haversine_distance(lat1, lon1, lat2, lon2):
    """2点間の距離(m)を計算する"""
    R = 6371000  # 地球の半径
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def clean_elevation_spikes(points):
    """
    スパイク補正アルゴリズム
    引数 points: [{'lat':, 'lon':, 'ele':, 'dist':, 'cum_dist':}, ...] のリスト
    戻り値: 補正後の points, ログ用データ
    """
    n = len(points)
    if n < 2:
        return points, []

    # ログ用：各点の属性を記録する
    point_attr = [ {
        'grade_prev': 0.0, 'ele_diff': 0.0, 'spike_type': '', 
        'cluster_id': -1, 'is_anchor': False, 'new_ele': p['ele']
    } for p in points ]

    # --- フェーズ1：バッドセグメントの検出 ---
    bad_segments = []
    for i in range(1, n):
        p1, p2 = points[i-1], points[i]
        dist = p2['cum_dist'] - p1['cum_dist']
        if dist <= 0: continue
        
        ele_diff = abs(p2['ele'] - p1['ele'])
        grade = (ele_diff / dist) * 100
        point_attr[i]['grade_prev'] = grade
        point_attr[i]['ele_diff'] = ele_diff

        is_bad = False
        # 通常スパイク
        if grade >= BAD_GRADE_THRESHOLD and (dist < SHORT_SEG_M or ele_diff >= MIN_ELEVATION_JUMP_M):
            is_bad = True
            point_attr[i]['spike_type'] = 'Normal'
        # ハードスパイク
        if grade >= HARD_SPIKE_THRESHOLD and (dist < SHORT_SEG_M or ele_diff >= NEAR_ELEVATION_JUMP_M):
            is_bad = True
            point_attr[i]['spike_type'] = 'Hard'
        
        if is_bad:
            bad_segments.append(i)

    if not bad_segments:
        return points, point_attr

    # --- フェーズ2：周辺セグメントの取り込み（クラスタリング） ---
    clusters = []
    used_indices = set()
    for start_idx in bad_segments:
        if start_idx in used_indices: continue
        
        # クラスタ初期化
        current_cluster = {start_idx}
        used_indices.add(start_idx)
        
        # 前後 CLUSTER_GAP_M 以内をスキャン
        for i in range(1, n):
            if i in used_indices: continue
            p_target = points[i]
            # クラスタ内のいずれかの点から距離内かチェック
            in_range = any(abs(p_target['cum_dist'] - points[c]['cum_dist']) <= CLUSTER_GAP_M for c in current_cluster)
            if in_range:
                p1, p2 = points[i-1], points[i]
                d = p2['cum_dist'] - p1['cum_dist']
                if d > 0:
                    diff = abs(p2['ele'] - p1['ele'])
                    g = (diff / d) * 100
                    if g >= NEAR_BAD_THRESHOLD and (d < SHORT_SEG_M or diff >= NEAR_ELEVATION_JUMP_M):
                        current_cluster.add(i)
                        used_indices.add(i)
        
        cluster_list = sorted(list(current_cluster))
        clusters.append({'indices': cluster_list, 'start': cluster_list[0], 'end': cluster_list[-1]})

    for idx, c in enumerate(clusters):
        for i in c['indices']: point_attr[i]['cluster_id'] = idx

    # --- フェーズ3：アンカーの選定 ---
    repair_ranges = []

    def is_anchor_candidate(idx):
        if idx <= 0 or idx >= n - 1: return False
        # 1. 前後勾配が 12% 以下
        g_prev = point_attr[idx]['grade_prev']
        # 次のセグメント勾配を計算
        d_next = points[idx+1]['cum_dist'] - points[idx]['cum_dist']
        g_next = (abs(points[idx+1]['ele'] - points[idx]['ele']) / d_next * 100) if d_next > 0 else 0
        if g_prev > ANCHOR_GRADE_LIMIT or g_next > ANCHOR_GRADE_LIMIT: return False
        
        # 2. 前後150m以内の中央値との差が 5m 以内
        window = [p['ele'] for p in points if abs(p['cum_dist'] - points[idx]['cum_dist']) <= MEDIAN_WINDOW_M]
        if not window: return False
        if abs(points[idx]['ele'] - median(window)) > ANCHOR_MEDIAN_DEV_M: return False
        return True

    for c in clusters:
        left_anchor = None
        right_anchor = None
        
        # 左アンカー探し (外側方向)
        search_idx = c['start'] - 1
        found_candidates = 0
        while search_idx >= 0 and (points[c['start']]['cum_dist'] - points[search_idx]['cum_dist'] <= MAX_ANCHOR_SEARCH_M):
            # 境界アンカー高速判定
            if search_idx == c['start'] - 1:
                d_next = points[search_idx+1]['cum_dist'] - points[search_idx]['cum_dist']
                g_next = (abs(points[search_idx+1]['ele'] - points[search_idx]['ele']) / d_next * 100) if d_next > 0 else 0
                if g_next <= BOUNDARY_GRADE_LIMIT and is_anchor_candidate(search_idx):
                    left_anchor = search_idx
                    break
            
            if is_anchor_candidate(search_idx):
                found_candidates += 1
                if found_candidates >= 2:
                    left_anchor = search_idx
                    break
            else:
                found_candidates = 0
            search_idx -= 1
            
        # 右アンカー探し
        search_idx = c['end'] + 1
        found_candidates = 0
        while search_idx < n and (points[search_idx]['cum_dist'] - points[c['end']]['cum_dist'] <= MAX_ANCHOR_SEARCH_M):
            if search_idx == c['end'] + 1:
                d_prev = points[search_idx]['cum_dist'] - points[search_idx-1]['cum_dist']
                g_prev = (abs(points[search_idx]['ele'] - points[search_idx-1]['ele']) / d_prev * 100) if d_prev > 0 else 0
                if g_prev <= BOUNDARY_GRADE_LIMIT and is_anchor_candidate(search_idx):
                    right_anchor = search_idx
                    break

            if is_anchor_candidate(search_idx):
                found_candidates += 1
                if found_candidates >= 2:
                    right_anchor = search_idx
                    break
            else:
                found_candidates = 0
            search_idx += 1
            
        if left_anchor is not None and right_anchor is not None:
            # 正味勾配チェック
            dist = points[right_anchor]['cum_dist'] - points[left_anchor]['cum_dist']
            net_grade = (abs(points[right_anchor]['ele'] - points[left_anchor]['ele']) / dist * 100) if dist > 0 else 0
            if net_grade <= MAX_ANCHOR_GRADE:
                repair_ranges.append((left_anchor, right_anchor))

    # --- フェーズ4：修復範囲のマージ ---
    if not repair_ranges:
        return points, point_attr
        
    repair_ranges.sort()
    merged_ranges = []
    if repair_ranges:
        curr_start, curr_end = repair_ranges[0]
        for i in range(1, len(repair_ranges)):
            next_start, next_end = repair_ranges[i]
            # 重なるか、50m以内に隣接する場合
            gap = points[next_start]['cum_dist'] - points[curr_end]['cum_dist']
            if next_start <= curr_end or gap <= MERGE_GAP_M:
                curr_end = max(curr_end, next_end)
            else:
                merged_ranges.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_ranges.append((curr_start, curr_end))

    # --- フェーズ5：線形補間 ---
    for r_start, r_end in merged_ranges:
        point_attr[r_start]['is_anchor'] = True
        point_attr[r_end]['is_anchor'] = True
        ele_l = points[r_start]['ele']
        ele_r = points[r_end]['ele']
        dist_l = points[r_start]['cum_dist']
        dist_r = points[r_end]['cum_dist']
        total_dist = dist_r - dist_l
        
        for i in range(r_start + 1, r_end):
            ratio = (points[i]['cum_dist'] - dist_l) / total_dist
            new_ele = ele_l + (ele_r - ele_l) * ratio
            # 0.5m 未満の差は変更しない
            if abs(new_ele - points[i]['ele']) >= 0.5:
                points[i]['ele'] = round(new_ele, 2)
                point_attr[i]['new_ele'] = points[i]['ele']

    return points, point_attr

def main():
    if len(sys.argv) < 4:
        print("Usage: python SpikeFixer.py [input.gpx] [output.gpx] [compare.txt]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    compare_file = sys.argv[3]

    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing GPX: {e}")
        sys.exit(1)

    # 名前空間の処理
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    ET.register_namespace('', ns['gpx'])

    # 全トラックポイントをリスト化
    points_data = []
    xml_points = []
    cumulative_dist = 0.0
    last_pos = None

    for trkpt in root.findall('.//gpx:trkpt', ns):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        ele_elem = trkpt.find('gpx:ele', ns)
        ele = float(ele_elem.text) if ele_elem is not None else 0.0
        
        dist = 0.0
        if last_pos:
            dist = haversine_distance(last_pos[0], last_pos[1], lat, lon)
        cumulative_dist += dist
        
        points_data.append({
            'lat': lat, 'lon': lon, 'ele': ele, 
            'dist': dist, 'cum_dist': cumulative_dist
        })
        xml_points.append(trkpt)
        last_pos = (lat, lon)

    # スパイク補正実行
    # points_data はミュータブルなので ele が書き換わる
    original_eles = [p['ele'] for p in points_data]
    corrected_points, attributes = clean_elevation_spikes(points_data)

    # GPXツリーの更新
    for i, trkpt in enumerate(xml_points):
        ele_elem = trkpt.find('gpx:ele', ns)
        if ele_elem is not None:
            ele_elem.text = f"{corrected_points[i]['ele']:.2f}"
        else:
            new_ele = ET.SubElement(trkpt, '{http://www.topografix.com/GPX/1/1}ele')
            new_ele.text = f"{corrected_points[i]['ele']:.2f}"

    tree.write(output_file, encoding='utf-8', xml_declaration=True)

    # 比較レポート (compare.txt) の生成
    with open(compare_file, 'w', encoding='utf-8') as f:
        f.write("========================================================================================\n")
        f.write("  SpikeFixer Comparison Report\n")
        f.write(f"  Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Input: {input_file}\n")
        f.write("========================================================================================\n\n")
        
        header = (f"{'ID':>4} | {'Dist(m)':>8} | {'In-Ele':>8} | {'Out-Ele':>8} | {'Diff':>6} | "
                  f"{'Grade%':>7} | {'Spike':>7} | {'Clust':>5} | {'Anchor':>6}\n")
        f.write(header)
        f.write("-" * len(header) + "\n")

        for i in range(len(points_data)):
            p = points_data[i]
            attr = attributes[i]
            orig_e = original_eles[i]
            new_e = attr['new_ele']
            e_diff = new_e - orig_e
            
            spike_str = attr['spike_type'] if attr['spike_type'] else ""
            clust_str = str(attr['cluster_id']) if attr['cluster_id'] != -1 else ""
            anchor_str = "YES" if attr['is_anchor'] else ""
            
            line = (f"{i:4d} | {p['cum_dist']:8.1f} | {orig_e:8.2f} | {new_e:8.2f} | {e_diff:6.2f} | "
                    f"{attr['grade_prev']:7.1f} | {spike_str:7} | {clust_str:5} | {anchor_str:6}\n")
            f.write(line)

    print(f"Done.")
    print(f"Output GPX: {output_file}")
    print(f"Compare Report: {compare_file}")

if __name__ == "__main__":
    main()