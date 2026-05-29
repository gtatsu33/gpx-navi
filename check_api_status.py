# -*- coding: utf-8 -*-
"""
Valhalla / 国土地理院 サービス状態チェッカー
使い方: python check_api_status.py
"""

import sys
import time
import urllib.parse
import requests

# Windows cp932 ターミナル対応
if sys.stdout.encoding and sys.stdout.encoding.lower() in ("cp932", "mbcs"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VALHALLA_BASE = "https://valhalla1.openstreetmap.de"
VALHALLA_URL  = VALHALLA_BASE + "/trace_attributes"
GSI_URL       = "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php"
OVERPASS_URLS = [
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# 新宿西口交差点付近（traffic_signals + name タグ持ちノードが多い）
OVERPASS_QUERY = (
    "[out:json][timeout:15];"
    "node(around:50,35.6896,139.6917)[highway~\"^(traffic_signals|crossing)$\"][name];"
    "out body 1;"
)

# テスト用座標（千葉県富津市・浜金谷付近の山中）
TEST_COORDS = [
    (35.1650, 139.8320),
    (35.1612, 139.8301),
    (35.1700, 139.8350),
]

# Valhalla trace_attributes テスト: 新宿〜渋谷 の主要道路（約4km）
VALHALLA_PAYLOAD = {
    "shape": [
        {"lat": 35.6896, "lon": 139.6917},  # 新宿駅南口
        {"lat": 35.6830, "lon": 139.7025},  # 代々木公園
        {"lat": 35.6580, "lon": 139.7016},  # 渋谷駅
    ],
    "costing": "bicycle",
    "shape_match": "map_snap",
    "search_radius": 50,
    "filters": {
        "attributes": ["matched.point", "matched.type"],
        "action": "include",
    },
}

SEP = "-" * 60


def verdict(elapsed_s, timeout=False, error=False):
    if timeout or error:
        return "[NG]"
    if elapsed_s < 1.0:
        return "[OK  快適]"
    if elapsed_s < 5.0:
        return "[注意 やや遅い]"
    return "[NG  非常に遅い]"


def check_valhalla():
    print("\n【Valhalla マップマッチングAPI】")
    print(f"  URL: {VALHALLA_BASE}")

    # ① /status で死活確認
    t0 = time.perf_counter()
    try:
        rs = requests.get(VALHALLA_BASE + "/status", timeout=10)
        t_status = time.perf_counter() - t0
        rs.raise_for_status()
        tileset = rs.json().get("tileset_last_modified", "")
        print(f"  /status    成功  {t_status:.2f}s  tileset={tileset}  {verdict(t_status)}")
    except requests.exceptions.Timeout:
        t_status = time.perf_counter() - t0
        print(f"  /status    タイムアウト ({t_status:.1f}s 超)  {verdict(0, timeout=True)}")
        return
    except Exception as e:
        t_status = time.perf_counter() - t0
        print(f"  /status    エラー ({t_status:.2f}s): {e}  {verdict(0, error=True)}")
        return

    # ② trace_attributes で実際のマッチングテスト（東京市街地座標）
    t0 = time.perf_counter()
    try:
        r = requests.post(VALHALLA_URL, json=VALHALLA_PAYLOAD, timeout=20)
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        n = len(r.json().get("matched_points", []))
        v = verdict(elapsed)
        print(f"  /trace     成功  {elapsed:.2f}s  マッチ点数={n}  {v}")
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - t0
        print(f"  /trace     タイムアウト ({elapsed:.1f}s 超)  {verdict(0, timeout=True)}")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        body = ""
        try:
            body = e.response.text[:200] if hasattr(e, "response") and e.response else ""
        except Exception:
            pass
        print(f"  /trace     エラー ({elapsed:.2f}s): {e}  {body}  {verdict(0, error=True)}")


def check_overpass():
    print("\n【Overpass API (交差点名取得)】")
    payload = "data=" + urllib.parse.quote(OVERPASS_QUERY)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "GPXTurnDetector/1.0",
    }
    for url in OVERPASS_URLS:
        print(f"  URL: {url}")
        t0 = time.perf_counter()
        try:
            r = requests.post(url, data=payload, headers=headers, timeout=20)
            elapsed = time.perf_counter() - t0
            r.raise_for_status()
            elements = r.json().get("elements", [])
            names = [e.get("tags", {}).get("name", "") for e in elements if e.get("tags", {}).get("name")]
            sample = f'  サンプル名="{names[0]}"' if names else "  名前付きノードなし"
            print(f"  成功  {elapsed:.2f}s  ノード数={len(elements)}{sample}  {verdict(elapsed)}")
        except requests.exceptions.Timeout:
            elapsed = time.perf_counter() - t0
            print(f"  タイムアウト ({elapsed:.1f}s 超)  {verdict(0, timeout=True)}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            body = ""
            try:
                body = e.response.text[:200] if hasattr(e, "response") and e.response else ""
            except Exception:
                pass
            print(f"  エラー ({elapsed:.2f}s): {e}  {body}  {verdict(0, error=True)}")
        time.sleep(0.5)


def check_gsi():
    print("\n【国土地理院 標高API】")
    print(f"  URL: {GSI_URL}")
    elapsed_list = []
    for i, (lat, lon) in enumerate(TEST_COORDS):
        t0 = time.perf_counter()
        try:
            r = requests.get(
                GSI_URL,
                params={"lat": lat, "lon": lon, "outtype": "JSON"},
                timeout=30,
            )
            elapsed = time.perf_counter() - t0
            r.raise_for_status()
            elev = r.json().get("elevation", "N/A")
            print(f"  [{i+1}] ({lat}, {lon})  成功  {elapsed:.2f}s  標高={elev}m")
            elapsed_list.append(elapsed)
        except requests.exceptions.Timeout:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}] ({lat}, {lon})  タイムアウト ({elapsed:.1f}s 超)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}] ({lat}, {lon})  エラー ({elapsed:.2f}s): {e}")
        time.sleep(0.3)

    if elapsed_list:
        avg = sum(elapsed_list) / len(elapsed_list)
        v = verdict(avg)
        print(f"\n  平均応答時間: {avg:.2f}s  ({len(elapsed_list)}/{len(TEST_COORDS)} 件成功)  {v}")
        if avg >= 5.0:
            print("  → CDNキャッシュミスの可能性あり（ユニーク座標は毎回オリジンサーバーへ）")
    else:
        print(f"\n  全件失敗  {verdict(0, error=True)}")


if __name__ == "__main__":
    print(SEP)
    print("  API サービス状態チェック")
    print(f"  実行時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)

    check_valhalla()
    check_gsi()
    check_overpass()

    print(f"\n{SEP}")
    print("  チェック完了")
    print(SEP)
