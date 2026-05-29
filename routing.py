import requests


def calc_route_segment(
    points: list[tuple[float, float]],
    costing: str = "bicycle",
) -> list[tuple[float, float]]:
    """
    複数点を経由する道路沿いtrkpt列を返す。
    OSRM public API を使用。失敗時は points をそのまま返す（直線フォールバック）。
    points は [(lat, lng), ...] 形式。
    """
    # OSRM は lon,lat 順でセミコロン区切り
    coords_str = ";".join(f"{lng},{lat}" for lat, lng in points)
    try:
        r = requests.get(
            f"http://router.project-osrm.org/route/v1/bike/{coords_str}",
            params={"overview": "full", "geometries": "geojson"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok":
            return points
        # GeoJSON は [lng, lat] 順 → (lat, lng) に変換
        return [(c[1], c[0]) for c in data["routes"][0]["geometry"]["coordinates"]]
    except Exception:
        return points


def find_segment_boundaries(
    acpt_trkpt_idx: int,
    acpts: list[dict],
    wpts: list[dict],
    trkpts: list,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    指定acptの前後境界点を (lat, lng) で返す。
    優先順: acpt → wpt → スタート/ゴール
    """
    prev = None
    for a in sorted(acpts, key=lambda x: x["trkpt_idx"], reverse=True):
        if a["trkpt_idx"] < acpt_trkpt_idx:
            prev = (a["lat"], a["lng"])
            break
    if prev is None:
        for w in sorted(wpts, key=lambda x: x["trkpt_idx"], reverse=True):
            if w["trkpt_idx"] < acpt_trkpt_idx:
                prev = (w["lat"], w["lng"])
                break
    if prev is None:
        prev = tuple(trkpts[0])

    nxt = None
    for a in sorted(acpts, key=lambda x: x["trkpt_idx"]):
        if a["trkpt_idx"] > acpt_trkpt_idx:
            nxt = (a["lat"], a["lng"])
            break
    if nxt is None:
        for w in sorted(wpts, key=lambda x: x["trkpt_idx"]):
            if w["trkpt_idx"] > acpt_trkpt_idx:
                nxt = (w["lat"], w["lng"])
                break
    if nxt is None:
        nxt = tuple(trkpts[-1])

    return prev, nxt
