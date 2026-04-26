"""
GPX ターン検出・強化ツール
点X の前後の点 A,B のベアリング差でコーナーを検出する
手動編集機能付き（追加・削除・名前変更）
"""

import streamlit as st
import gpxpy
import gpxpy.gpx
import math
import numpy as np
import folium
from streamlit_folium import st_folium

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
        bearing_in  = ベアリング(A → X)  ← 進入方向
        bearing_out = ベアリング(X → B)  ← 離脱方向
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

def build_enhanced_gpx(gpx_content_str, turns):
    enhanced = gpxpy.parse(gpx_content_str)
    enhanced.waypoints = []
    for t in turns:
        name = t.get("name") or turn_label(t["delta"])[0]
        delta = t.get("delta")
        desc = f"bearing_change:{delta:.1f}" if delta is not None else "manually added"
        wpt = gpxpy.gpx.GPXWaypoint(
            latitude=t["lat"], longitude=t["lon"],
            name=name, description=desc,
        )
        enhanced.waypoints.append(wpt)
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
if st.session_state.get("_file_name") != uploaded.name:
    st.session_state["_file_name"] = uploaded.name
    for k in ["edit_turns", "pending_wpt", "_handled_click", "_handled_tooltip",
              "_map_center", "_map_zoom"]:
        st.session_state.pop(k, None)
_skip_map_center_save = st.session_state.pop("_skip_map_center_save", False)
in_edit_mode = "edit_turns" in st.session_state

# ─── ルート情報 ───────────────────────────────
dists_all = [haversine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
             for i in range(len(points) - 1)]
avg_spacing = np.mean(dists_all)
total_dist_km = sum(dists_all) / 1000
route_name = next((t.name for t in gpx_parsed.tracks if t.name), "（名称なし）")

c1, c2, c3 = st.columns(3)
c1.metric("ルート名", route_name)
c2.metric("総距離", f"{total_dist_km:.1f} km")
c3.metric("GPSポイント間隔（平均）", f"{avg_spacing:.0f} m")

# ─────────────────────────────────────────────
# サイドバー
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

# ─── ターン検出 ───────────────────────────────
if not in_edit_mode:
    raw_turns = detect_turns(points, min_turn_angle=min_turn_angle,
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
                     else points[len(points)//4])
    _map_init_zoom = st.session_state.get("_map_zoom", 13)
    m = folium.Map(location=_map_init_loc, zoom_start=_map_init_zoom)
    folium.PolyLine(points, color="#3498db", weight=4, opacity=0.8).add_to(m)
    folium.Marker(points[0],  tooltip="スタート",
                  icon=folium.Icon(color="green",   icon="play", prefix="fa")).add_to(m)
    folium.Marker(points[-1], tooltip="ゴール",
                  icon=folium.Icon(color="darkred", icon="flag", prefix="fa")).add_to(m)

    for i, t in enumerate(current_turns):
        arrow, hex_color = wpt_style(t)
        delta = t.get("delta")
        popup_html = (f"<b>{arrow} {t['name']}</b>"
                      + (f"<br>ターン角: {delta:+.1f}°" if delta is not None else "")
                      + f"<br>idx: {t['index']}")
        # 編集モードではtooltipにtrkptインデックスを埋め込む（クリック検出用）
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

    # 選択中（保留）のtrkptをオレンジ星で表示
    if pending:
        folium.Marker(
            [pending["lat"], pending["lon"]],
            tooltip=f"追加予定 trkpt#{pending['index']}",
            icon=folium.Icon(color="orange", icon="star", prefix="fa"),
        ).add_to(m)

    map_data = st_folium(
        m, height=520, use_container_width=True,
        key="gpx_map",
        center=_map_init_loc,       # ← 追加
        zoom=_map_init_zoom,        # ← 追加
        returned_objects=(["last_clicked", "last_object_clicked_tooltip"]
                          if in_edit_mode else []),
    )

# 地図の表示位置を記憶（rerun後も同じ位置を維持するため）
if map_data and not _skip_map_center_save:
    if map_data.get("center"):
        st.session_state["_map_center"] = map_data["center"]
    if map_data.get("zoom") is not None:   # 0 の場合も拾えるよう is not None に
        st.session_state["_map_zoom"] = map_data["zoom"]

# ─── マップクリック → pending_wpt 更新 ─────────
if in_edit_mode and map_data:
    tooltip_val = map_data.get("last_object_clicked_tooltip") or ""

    if tooltip_val.startswith("wpt:") and tooltip_val != st.session_state.get("_handled_tooltip"):
        # 既存wptのマーカークリック → tooltipからtrkptインデックスを確実に取得
        st.session_state["_handled_tooltip"] = tooltip_val
        trkpt_idx = int(tooltip_val.split("|")[0][4:])
        # このクリックに対応するlast_clickedも処理済みとしてマーク
        if map_data.get("last_clicked"):
            click = map_data["last_clicked"]
            st.session_state["_handled_click"] = (round(click["lat"], 7), round(click["lng"], 7))
        st.session_state["pending_wpt"] = {
            "index": trkpt_idx,
            "lat":   points[trkpt_idx][0],
            "lon":   points[trkpt_idx][1],
        }
        st.session_state["_skip_map_center_save"] = True
        st.rerun()

    elif map_data.get("last_clicked"):
        # 空白地図クリック → 新規trkpt選択
        click = map_data["last_clicked"]
        click_key = (round(click["lat"], 7), round(click["lng"], 7))
        if click_key != st.session_state.get("_handled_click"):
            st.session_state["_handled_click"] = click_key
            idx = nearest_trkpt_index(click["lat"], click["lng"], points)
            st.session_state["pending_wpt"] = {
                "index": idx,
                "lat": points[idx][0],
                "lon": points[idx][1],
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
                # キーをtrkptインデックスにすることで削除後も他の名前がずれない
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

        # クリックしたtrkptが既存wptかどうかを確認
        existing_idx = next(
            (j for j, t in enumerate(current_turns) if t["index"] == pending["index"]),
            None,
        )

        if existing_idx is not None:
            # 既存wptのクリック → 削除ダイアログ
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
            # 新規trkpt → 追加フォーム
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
col_dl1, col_dl2, _ = st.columns([1, 1, 2])
col_dl1.metric("ウェイポイント数", len(current_turns))

with col_dl2:
    if st.button("📥 強化GPXを生成", type="primary", disabled=(len(current_turns) == 0)):
        # 編集モードの場合はテキスト入力widgetの最新値を名前に反映
        turns_for_build = []
        for t in current_turns:
            tc = dict(t)
            if in_edit_mode:
                widget_key = f"wpt_name_{t['index']}"
                tc["name"] = st.session_state.get(widget_key, t.get("name", "ウェイポイント"))
            turns_for_build.append(tc)

        xml_output = build_enhanced_gpx(raw_content, turns_for_build)
        base_name = uploaded.name.replace(".gpx", "")
        st.download_button(
            label=f"⬇️ {base_name}_turns.gpx をダウンロード",
            data=xml_output,
            file_name=f"{base_name}_turns.gpx",
            mime="application/gpx+xml",
        )
        st.success(f"✅ {len(turns_for_build)} 個のウェイポイントを埋め込みました")
