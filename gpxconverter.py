"""
GPX ターン検出・強化ツール
点X の前後の点 A,B のベアリング差でコーナーを検出する
手動編集機能付き（追加・削除・名前変更）
マップマッチング（OSRM）・標高補正（国土地理院 / Open-Meteo）対応
"""

import streamlit as st
import gpxpy
import gpxpy.gpx
import math
import numpy as np
import folium
from streamlit_folium import st_folium
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="GPX ターン検出ツール", layout="wide", page_icon="🚴")
st.title("🚴 GPX ターン検出・強化ツール")
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

def with_name(t):
    """nameフィールドを持つターン辞書を返す（deltaから自動生成）"""
    if "name" in t:
        return t
    d = dict(t)
    d["name"] = turn_label(t["delta"])[0]
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

def map_match_points(points, profile="cycling", radius=50):
    """
    全trkptをValhalla trace_attributes でスナップする。
    戻り値: (matched_points, n_snapped, error_msg)
      - matched_points: 元と同じ長さのリスト。スナップできなかった点は元の座標のまま。
    """
    CHUNK     = 50
    matched   = list(points)
    n_snapped = 0
    errors    = []
    n_chunks  = math.ceil(len(points) / CHUNK)
    costing   = _VALHALLA_COSTING.get(profile, "bicycle")

    prog = st.progress(0, text="マップマッチング中…")
    for ci in range(n_chunks):
        s     = ci * CHUNK
        e     = min(s + CHUNK, len(points))
        chunk = points[s:e]
        try:
            data = _valhalla_match_chunk(chunk, costing, radius)
        except Exception as ex:
            errors.append(f"chunk {ci}: {ex}")
            prog.progress((ci + 1) / n_chunks,
                          text=f"マッチング中… {ci+1}/{n_chunks} チャンク (エラー)")
            continue

        for j, mp in enumerate(data.get("matched_points", [])):
            if mp.get("type") in ("matched", "interpolated") and s + j < len(matched):
                matched[s + j] = (mp["lat"], mp["lon"])
                n_snapped += 1

        prog.progress((ci + 1) / n_chunks,
                      text=f"マッチング中… {ci+1}/{n_chunks} チャンク")

    prog.empty()
    return matched, n_snapped, ("; ".join(errors) if errors else None)

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
    n         = len(points)
    elevations = [None] * n
    in_japan  = _is_in_japan(points[0][0], points[0][1]) if points else False
    use_gsi   = (source == "gsi") or (source == "auto" and in_japan)
    src_label  = "国土地理院" if use_gsi else "Open-Meteo"

    prog = st.progress(0, text=f"標高データ取得中（{src_label}）…")

    if use_gsi:
        # 国土地理院: 並列リクエスト（ThreadPoolExecutor + as_completed でプログレス更新）
        done = [0]

        def _fetch_one(args):
            i, lat, lon = args
            try:
                return i, _fetch_gsi_elevation(lat, lon)
            except Exception:
                return i, None

        tasks = [(i, lat, lon) for i, (lat, lon) in enumerate(points)]
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(_fetch_one, t): t[0] for t in tasks}
            for future in as_completed(futures):
                i, elev = future.result()
                elevations[i] = elev
                done[0] += 1
                prog.progress(done[0] / n,
                              text=f"標高取得中（国土地理院）… {done[0]}/{n} 点")

        # GSIで取れなかった点をOpen-Meteoでフォールバック
        failed = [i for i, e in enumerate(elevations) if e is None]
        if failed:
            BATCH = 100
            for b in range(0, len(failed), BATCH):
                idxs  = failed[b:b + BATCH]
                batch = [points[i] for i in idxs]
                try:
                    batch_e = _fetch_openmeteo_batch(batch)
                    for j, idx in enumerate(idxs):
                        elevations[idx] = batch_e[j]
                except Exception:
                    pass
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

    # ウェイポイントを再構築
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

# ファイルが変わったらセッション状態をリセット
_STATE_KEYS = [
    "edit_turns", "pending_wpt", "_handled_click", "_handled_tooltip",
    "_map_center", "_map_zoom",
    "_matched_points", "_mm_status", "_mm_n_snapped", "_mm_error",
    "_elevations",    "_elev_status", "_elev_source", "_elev_n_ok",
]
if st.session_state.get("_file_name") != uploaded.name:
    st.session_state["_file_name"] = uploaded.name
    for k in _STATE_KEYS:
        st.session_state.pop(k, None)

_skip_map_center_save = st.session_state.pop("_skip_map_center_save", False)
in_edit_mode = "edit_turns" in st.session_state

# マッチング済み座標があればそちらを使う（標高取得・ターン検出・地図描画すべてに適用）
active_points = st.session_state.get("_matched_points", points)

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
# サイドバー ─ パラメータ調整
# ─────────────────────────────────────────────

st.sidebar.header("⚙️ パラメータ調整")

if in_edit_mode:
    st.sidebar.info("✏️ 手動編集モード中\nパラメータは固定されています")
    if st.sidebar.button("🔄 自動検出に戻す（編集をリセット）"):
        for k in ["edit_turns", "pending_wpt", "_handled_click", "_handled_tooltip",
                  "_map_center", "_map_zoom"]:
            st.session_state.pop(k, None)
        st.session_state["_skip_map_center_save"] = True
        st.rerun()
    min_turn_angle = st.session_state.get("_mta", 45)
    min_dist_val   = st.session_state.get("_md",  100)
    smooth_val     = st.session_state.get("_sm",  1)
else:
    st.sidebar.markdown("""
**アルゴリズム**: 角度法
点 X の前後の点 A, B のベアリング差でコーナーを判定します。
""")
    min_turn_angle = st.sidebar.slider(
        "ターン角閾値（度）", 20, 120, 45, 5,
        help="進入・離脱方向の差がこの角度以上ならコーナーとみなす。\n"
             "45°=やや曲がりも検出、60°=交差点のみ、90°=ほぼ直角以上のみ")
    min_dist_val = st.sidebar.slider(
        "ターン間最小距離（m）", 30, 500, 100, 10,
        help="同一交差点での重複検出を防ぐ")
    smooth_val = st.sidebar.slider(
        "スムージング（前後N点参照）", 1, 5, 1, 1,
        help="1=隣接点のみ（推奨）、2以上=ノイズに強いが精度低下の可能性あり")
    st.session_state["_mta"] = min_turn_angle
    st.session_state["_md"]  = min_dist_val
    st.session_state["_sm"]  = smooth_val

# ─── ターン検出（active_pointsを使用） ────────
if not in_edit_mode:
    raw_turns    = detect_turns(active_points, min_turn_angle=min_turn_angle,
                                min_dist=min_dist_val, smooth=smooth_val)
    current_turns = [with_name(t) for t in raw_turns]
else:
    current_turns = st.session_state["edit_turns"]

# ─── 手動編集モード開始 ──────────────────────
if not in_edit_mode:
    st.sidebar.divider()
    if st.sidebar.button("✏️ 手動編集モードへ", type="primary"):
        st.session_state["edit_turns"] = [with_name(dict(t)) for t in current_turns]
        st.session_state["_skip_map_center_save"] = True
        st.rerun()

# ─────────────────────────────────────────────
# サイドバー ─ マップマッチング（Valhalla）
# ─────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.header("🗺️ マップマッチング")
st.sidebar.caption("OSM道路網に各trkptをスナップします（Valhalla API使用）")

mm_status = st.session_state.get("_mm_status", "未実行")

if mm_status == "完了":
    n_snapped = st.session_state.get("_mm_n_snapped", 0)
    st.sidebar.success(f"✅ 完了 — {n_snapped}/{len(points)} 点スナップ済み")
    if st.session_state.get("_mm_error"):
        st.sidebar.caption(f"⚠️ 一部エラー: {st.session_state['_mm_error'][:120]}")
    if st.sidebar.button("🔄 マッチングをリセット", key="mm_reset"):
        for k in ["_matched_points", "_mm_status", "_mm_n_snapped", "_mm_error",
                  "edit_turns", "pending_wpt"]:
            st.session_state.pop(k, None)
        st.session_state["_skip_map_center_save"] = True
        st.rerun()
else:
    if mm_status == "エラー":
        err_detail = st.session_state.get("_mm_error", "")
        st.sidebar.error(f"❌ マッチング失敗\n{err_detail[:200]}")
    else:
        st.sidebar.info("未実行")

    _mm_profiles = {
        "自転車 (bicycle)":   "cycling",
        "徒歩・ハイキング":    "foot",
        "車 (auto)":         "driving",
    }
    _mm_profile_key = st.sidebar.selectbox(
        "プロファイル", list(_mm_profiles.keys()), key="mm_profile_sel")
    _mm_radius = st.sidebar.slider(
        "サーチ半径（m）", 10, 100, 50, 10,
        help="道路を探索する半径。GPS誤差が大きいルートは大きくする")

    if st.sidebar.button("🗺️ マッチング実行", type="primary", key="run_mm"):
        try:
            matched, n_snapped, err = map_match_points(
                points,
                profile=_mm_profiles[_mm_profile_key],
                radius=_mm_radius,
            )
        except Exception as ex:
            st.sidebar.error(f"予期しないエラー:\n{ex}")
            st.stop()
        st.session_state["_matched_points"] = matched
        st.session_state["_mm_n_snapped"]   = n_snapped
        st.session_state["_mm_error"]       = err
        st.session_state["_mm_status"]      = "完了" if n_snapped > 0 else "エラー"
        # ポイントが変わるため編集モードとキャッシュをリセット
        for k in ["edit_turns", "pending_wpt",
                  "_elevations", "_elev_status", "_elev_source", "_elev_n_ok"]:
            st.session_state.pop(k, None)
        st.session_state["_skip_map_center_save"] = True
        st.rerun()

# ─────────────────────────────────────────────
# サイドバー ─ 標高補正
# ─────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.header("⛰️ 標高補正")
st.sidebar.caption("日本国内→国土地理院DEM（5m精度）、海外→Open-Meteo（90m）")

elev_status = st.session_state.get("_elev_status", "未実行")

if elev_status == "完了":
    n_ok = st.session_state.get("_elev_n_ok", 0)
    src  = st.session_state.get("_elev_source", "")
    st.sidebar.success(f"✅ 完了 — {n_ok}/{len(active_points)} 点取得（{src}）")
    if st.sidebar.button("🔄 標高データをリセット", key="elev_reset"):
        for k in ["_elevations", "_elev_status", "_elev_source", "_elev_n_ok"]:
            st.session_state.pop(k, None)
        st.session_state["_skip_map_center_save"] = True
        st.rerun()
else:
    if elev_status == "エラー":
        st.sidebar.warning("⚠️ 一部取得失敗")
    else:
        st.sidebar.info("未実行")

    _elev_sources = {
        "自動（日本→国土地理院、海外→Open-Meteo）": "auto",
        "国土地理院 API のみ（日本専用・高精度）":    "gsi",
        "Open-Meteo のみ（全世界・低精度）":         "openmeteo",
    }
    _elev_src_key = st.sidebar.selectbox(
        "データソース", list(_elev_sources.keys()), key="elev_src_sel")

    in_japan_hint = _is_in_japan(active_points[0][0], active_points[0][1])
    st.sidebar.caption(
        f"ルート判定: {'🇯🇵 日本（国土地理院が利用可能）' if in_japan_hint else '🌍 海外（Open-Meteo）'}")

    if st.sidebar.button("⛰️ 標高を取得", type="primary", key="run_elev"):
        elevs, src_used, n_ok = fetch_all_elevations(
            active_points, source=_elev_sources[_elev_src_key])
        st.session_state["_elevations"]   = elevs
        st.session_state["_elev_source"]  = src_used
        st.session_state["_elev_n_ok"]    = n_ok
        st.session_state["_elev_status"]  = "完了" if n_ok > 0 else "エラー"
        st.session_state["_skip_map_center_save"] = True
        st.rerun()

# ─────────────────────────────────────────────
# 地図 + リストパネル
# ─────────────────────────────────────────────

if in_edit_mode:
    st.info("✏️ **手動編集モード** — 地図をクリックして新しいウェイポイントを追加できます。"
            "右パネルで削除・名前変更もできます。")

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
                      + f"<br>idx: {t['index']}")
        if in_edit_mode:
            tooltip_str = f"wpt:{t['index']}|{i+1}. {arrow} {t['name']}"
        else:
            tooltip_str = f"{i+1}. {arrow} {t['name']}"
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
        returned_objects=(["last_clicked", "last_object_clicked_tooltip"]
                          if in_edit_mode else []),
    )

# 地図の表示位置を記憶
if map_data and not _skip_map_center_save:
    if map_data.get("center"):
        st.session_state["_map_center"] = map_data["center"]
    if map_data.get("zoom") is not None:
        st.session_state["_map_zoom"] = map_data["zoom"]

# ─── マップクリック → pending_wpt 更新 ─────────
if in_edit_mode and map_data:
    tooltip_val = map_data.get("last_object_clicked_tooltip") or ""

    if tooltip_val.startswith("wpt:") and tooltip_val != st.session_state.get("_handled_tooltip"):
        st.session_state["_handled_tooltip"] = tooltip_val
        trkpt_idx = int(tooltip_val.split("|")[0][4:])
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
            st.session_state["pending_wpt"] = {
                "index": idx,
                "lat":   active_points[idx][0],
                "lon":   active_points[idx][1],
            }
            st.session_state["_skip_map_center_save"] = True
            st.rerun()

# ─── 右パネル（リスト） ───────────────────────
with col_list:
    st.subheader(f"📋 ウェイポイント一覧　({len(current_turns)}件)")

    if not current_turns and not (in_edit_mode and pending):
        st.warning("ウェイポイントが検出されませんでした。\nターン角閾値を下げてみてください。")

    for i, t in enumerate(current_turns):
        arrow, hex_color = wpt_style(t)
        delta = t.get("delta")

        if in_edit_mode:
            badge = f"{delta:+.1f}°" if delta is not None else "手動"
            st.markdown(
                f'<div style="border-left:4px solid {hex_color};padding:3px 8px;'
                f'background:#f8f9fa;border-radius:3px;margin-bottom:2px;">'
                f'<b>{i+1}. {arrow}</b> <small style="color:#888">{badge} | idx:{t["index"]}</small></div>',
                unsafe_allow_html=True,
            )
            col_n, col_d = st.columns([5, 1])
            with col_n:
                st.text_input(
                    "名前", value=t["name"],
                    key=f"wpt_name_{t['index']}",
                    label_visibility="collapsed",
                )
            with col_d:
                if st.button("🗑", key=f"del_{t['index']}", help="削除"):
                    st.session_state["edit_turns"].pop(i)
                    st.session_state.pop("pending_wpt", None)
                    st.rerun()
        else:
            name = t.get("name", "?")
            st.markdown(
                f'<div style="border-left:4px solid {hex_color};padding:6px 10px;'
                f'margin-bottom:5px;border-radius:4px;background:#f8f9fa;">'
                f'<b>{i+1}. {arrow} {name}</b><br>'
                f'<small>{delta:+.1f}° &nbsp;|&nbsp; idx:{t["index"]}</small></div>',
                unsafe_allow_html=True,
            )

    # ─── 保留ウェイポイント追加／削除UI ──────────
    if in_edit_mode and pending:
        st.divider()

        existing_idx = next(
            (j for j, t in enumerate(current_turns) if t["index"] == pending["index"]),
            None,
        )

        if existing_idx is not None:
            existing = current_turns[existing_idx]
            ex_arrow, ex_color = wpt_style(existing)
            st.markdown(
                f'<div style="border-left:4px solid {ex_color};padding:6px 10px;'
                f'border-radius:4px;background:#fff3cd;">'
                f'<b>{ex_arrow} {existing["name"]}</b><br>'
                f'<small>trkpt #{pending["index"]}</small></div>',
                unsafe_allow_html=True,
            )
            st.warning("既存のウェイポイントを選択しています")
            col_a, col_c = st.columns(2)
            with col_a:
                if st.button("🗑 削除", type="primary", key="pending_del"):
                    st.session_state["edit_turns"].pop(existing_idx)
                    st.session_state.pop("pending_wpt", None)
                    st.session_state["_skip_map_center_save"] = True
                    st.rerun()
            with col_c:
                if st.button("✖ キャンセル", key="pending_cancel_ex"):
                    st.session_state.pop("pending_wpt", None)
                    st.session_state["_skip_map_center_save"] = True
                    st.rerun()
        else:
            st.markdown(
                f"**📍 追加予定のポイント**  \n"
                f"trkpt #{pending['index']}  \n"
                f"`{pending['lat']:.6f}, {pending['lon']:.6f}`"
            )
            new_name = st.text_input(
                "ウェイポイント名", key="pending_name",
                placeholder="例: 右折、信号など",
            )
            col_a, col_c = st.columns(2)
            with col_a:
                if st.button("➕ 追加", type="primary", disabled=not (new_name or "").strip()):
                    new_wpt = {
                        "lat":   pending["lat"],
                        "lon":   pending["lon"],
                        "delta": None,
                        "index": pending["index"],
                        "name":  new_name.strip(),
                    }
                    turns_list = st.session_state["edit_turns"]
                    insert_at = next(
                        (j for j, t in enumerate(turns_list) if t["index"] > pending["index"]),
                        len(turns_list),
                    )
                    turns_list.insert(insert_at, new_wpt)
                    st.session_state.pop("pending_wpt", None)
                    st.session_state["_skip_map_center_save"] = True
                    st.rerun()
            with col_c:
                if st.button("✖ キャンセル", key="pending_cancel_new"):
                    st.session_state.pop("pending_wpt", None)
                    st.session_state["_skip_map_center_save"] = True
                    st.rerun()

    if in_edit_mode:
        st.caption("💡 地図をクリックして新しいポイントを追加")

# ─────────────────────────────────────────────
# ダウンロード
# ─────────────────────────────────────────────

st.divider()
st.subheader("💾 強化GPXの出力")

# 適用済み処理をバッジ表示
applied = []
if st.session_state.get("_mm_status") == "完了":
    applied.append("🗺️ マップマッチング済み")
if st.session_state.get("_elev_status") == "完了":
    applied.append(f"⛰️ 標高補正済み（{st.session_state.get('_elev_source', '')}）")
if applied:
    st.info("出力GPXに適用: " + " ／ ".join(applied))

col_dl1, col_dl2, _ = st.columns([1, 1, 2])
col_dl1.metric("ウェイポイント数", len(current_turns))

with col_dl2:
    if st.button("📥 強化GPXを生成", type="primary", disabled=(len(current_turns) == 0)):
        turns_for_build = []
        for t in current_turns:
            tc = dict(t)
            if in_edit_mode:
                widget_key = f"wpt_name_{t['index']}"
                tc["name"] = st.session_state.get(widget_key, t.get("name", "ウェイポイント"))
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
        st.success(f"✅ {len(turns_for_build)} 個のウェイポイントを埋め込みました")
