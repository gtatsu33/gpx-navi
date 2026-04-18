"""
GPX ターン検出・強化ツール
点X の前後の点 A,B のベアリング差でコーナーを検出する
シンプル・直感的なアルゴリズム
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

    Parameters
    ----------
    min_turn_angle : コーナーとみなす最小ターン角（度）
    min_dist       : 近接重複排除の距離（メートル）
    smooth         : 前後何点を参照するか（1=隣接点、2=2点先）
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

    # NMS: ターン角の大きい順に処理し、min_dist以内の重複を除去
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

def build_enhanced_gpx(gpx_content_str, turns):
    enhanced = gpxpy.parse(gpx_content_str)
    enhanced.waypoints = []
    for t in turns:
        label, arrow, _ = turn_label(t["delta"])
        wpt = gpxpy.gpx.GPXWaypoint(
            latitude=t["lat"], longitude=t["lon"],
            name=label,
            description=f"bearing_change:{t['delta']:.1f}",
        )
        enhanced.waypoints.append(wpt)
    return enhanced.to_xml()

# ─────────────────────────────────────────────
# UI
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

dists_all = [haversine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
             for i in range(len(points) - 1)]
avg_spacing = np.mean(dists_all)
total_dist_km = sum(dists_all) / 1000

route_name = next((t.name for t in gpx_parsed.tracks if t.name), "（名称なし）")

c1, c2, c3 = st.columns(3)
c1.metric("ルート名", route_name)
c2.metric("総距離", f"{total_dist_km:.1f} km")
c3.metric("GPSポイント間隔（平均）", f"{avg_spacing:.0f} m")

# ─── サイドバー ───────────────────────────────
st.sidebar.header("⚙️ パラメータ調整")
st.sidebar.markdown("""
**アルゴリズム**: 角度法  
点 X の前後の点 A, B のベアリング差でコーナーを判定します。
""")

min_turn_angle = st.sidebar.slider(
    "ターン角閾値（度）",
    min_value=20, max_value=120, value=45, step=5,
    help="進入・離脱方向の差がこの角度以上ならコーナーとみなす。\n"
         "45°=やや曲がりも検出、60°=交差点のみ、90°=ほぼ直角以上のみ"
)
min_dist = st.sidebar.slider(
    "ターン間最小距離（m）",
    min_value=30, max_value=500, value=100, step=10,
    help="同一交差点での重複検出を防ぐ"
)
smooth = st.sidebar.slider(
    "スムージング（前後N点参照）",
    min_value=1, max_value=5, value=1, step=1,
    help="1=隣接点のみ（推奨）、2以上=ノイズに強いが精度低下の可能性あり"
)

# ─── 検出・表示 ───────────────────────────────
turns = detect_turns(points, min_turn_angle=min_turn_angle,
                     min_dist=min_dist, smooth=smooth)

col_map, col_list = st.columns([2, 1])

with col_map:
    st.subheader("🗺️ 地図プレビュー")
    m = folium.Map(location=points[len(points)//4], zoom_start=13)
    folium.PolyLine(points, color="#3498db", weight=4, opacity=0.8).add_to(m)
    folium.Marker(points[0],  tooltip="スタート",
                  icon=folium.Icon(color="green",   icon="play", prefix="fa")).add_to(m)
    folium.Marker(points[-1], tooltip="ゴール",
                  icon=folium.Icon(color="darkred", icon="flag", prefix="fa")).add_to(m)

    for i, t in enumerate(turns):
        label, arrow, hex_color = turn_label(t["delta"])
        folium.CircleMarker(
            location=[t["lat"], t["lon"]], radius=9,
            color=hex_color, fill=True, fill_color=hex_color, fill_opacity=0.9,
            tooltip=f"{i+1}. {arrow} {label} ({t['delta']:+.1f}°)",
            popup=folium.Popup(
                f"<b>{arrow} {label}</b><br>ターン角: {t['delta']:+.1f}°<br>index: {t['index']}",
                max_width=200),
        ).add_to(m)

    st_folium(m, height=520, use_container_width=True)

with col_list:
    st.subheader(f"📋 ターン一覧　({len(turns)}件)")
    if not turns:
        st.warning("ターンが検出されませんでした。\nターン角閾値を下げてみてください。")
    else:
        for i, t in enumerate(turns):
            label, arrow, hex_color = turn_label(t["delta"])
            st.markdown(
                f"""<div style="border-left:4px solid {hex_color};
                    padding:6px 10px;margin-bottom:5px;border-radius:4px;background:#f8f9fa;">
                <b>{i+1}. {arrow} {label}</b><br>
                <small>{t['delta']:+.1f}° &nbsp;|&nbsp; idx:{t['index']}</small>
                </div>""",
                unsafe_allow_html=True)

# ─── ダウンロード ──────────────────────────────
st.divider()
st.subheader("💾 強化GPXの出力")

col_dl1, col_dl2, _ = st.columns([1, 1, 2])
col_dl1.metric("検出ターン数", len(turns))

with col_dl2:
    if st.button("📥 強化GPXを生成", type="primary", disabled=(len(turns) == 0)):
        xml_output = build_enhanced_gpx(raw_content, turns)
        base_name = uploaded.name.replace(".gpx", "")
        st.download_button(
            label=f"⬇️ {base_name}_turns.gpx をダウンロード",
            data=xml_output,
            file_name=f"{base_name}_turns.gpx",
            mime="application/gpx+xml",
        )
        st.success(f"✅ {len(turns)} 個のターンポイントを埋め込みました")
        