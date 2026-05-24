# gpxconverter マップ刷新 & 新規ルート作成機能 — 要件仕様 / 実装プロンプト

## 概要

`gpxconverter.py` のマップコンポーネントを `st_folium` から `components.html` 製のフル Leaflet.js コンポーネントに置き換える。

**実現したいこと：**
- 新規ルートをマップ上でクリックして作成できる
- アンカーポイント（acpt）のドラッグでルートを自動再計算できる
- GPX読み込みによる既存ルートの編集も同一UIで行える
- フェーズ分けなし。常にルートとwptを同時編集できる

**変えないこと：**
- Valhalla マップマッチング・標高補正・wpt管理・GPX出力の Python ロジック
- 右パネルのwpt一覧UI（ほぼそのまま流用）

---

## 用語定義

| 用語 | 定義 |
|------|------|
| **trkpt** | ルートを構成する座標点列。Valhalla /route の計算結果 |
| **acpt（アンカーポイント）** | ルートの形を決める中継点。ユーザーが配置・ドラッグする。ナビ案内なし |
| **wpt（ターンポイント）** | ターンバイターンのナビ案内点。名前付き。acptとは独立 |
| **セグメント** | 隣接する2つのacpt間のtrkpt列 |
| **境界点** | セグメントの端点。acpt > wpt > スタート/ゴール の優先順で決まる |

---

## アプリ全体フロー

```
┌──────────────────────────────────────┐
│           開始画面（モード選択）        │
│  [新規にルートを作成]  [GPXを読み込む]  │
└──────────┬───────────────┬───────────┘
           │               │
   ┌───────▼──────┐  ┌─────▼──────────────────┐
   │ 新規ルートモード│  │ GPX読み込みモード          │
   │              │  │ GPX → マップマッチング(任意)│
   │              │  │ → 標高補正(任意)           │
   │              │  │ → trkpt確定・acptなし      │
   └───────┬──────┘  └─────┬──────────────────┘
           └───────┬────────┘
                   │
         ┌─────────▼──────────────────────┐
         │   ルート編集モード（常時）         │
         │  ・acpt追加/ドラッグ/削除         │
         │  ・wpt追加/名前編集/削除          │
         │  ・フェーズ分けなし、常に両方編集可 │
         └─────────┬──────────────────────┘
                   │
         ┌─────────▼──────────┐
         │     GPX 出力        │
         └────────────────────┘
```

---

## モード1：新規ルートモード

- 空のマップから開始
- **最初のacptを置いた時点でスタート地点として表示する**（緑マーカー＋acpt）
- 2点目のacptを置いた時点で Valhalla /route で1セグメント計算 → trkpt生成
- セグメント生成のたびにターン角でwptを自動検出し右パネルに追加
- 以降はルート編集モードの操作が全て使える（常時編集）

```
[acpt1=スタート] ──[/route]──▶ [acpt2] ──[/route]──▶ [acpt3=末尾]
                                 ↑ドラッグ可
```

---

## モード2：GPX読み込みモード

1. GPXファイルをアップロード
2. マップマッチング（Valhalla `/trace_attributes`）・標高補正（任意、現行UIを流用）
3. trkptを確定。**acptは初期配置しない**
4. ルート編集モードへ（acptは後からユーザーが追加して形状変更可能）

---

## ルート編集モード（両モード共通・常時）

### マップ表示

| 要素 | 表示 |
|------|------|
| ルートポリライン | 青いライン（trkpt列） |
| acpt | 小さな白抜き●（ドラッグ可、右クリックで削除） |
| wpt | 大きな●（現行と同じ色分け）（ドラッグ不可） |
| スタート | 緑マーカー（最初のacptと兼用） |
| ゴール | 赤マーカー（最後のacptに追従） |
| ドラッグ中の予告ライン | acptから前後境界点への点線 |

### クリック挙動

「trkptに近いか」の判定がacpt挿入か末尾追加かの分岐点になる。

| クリック位置 | 判断 | 動作 |
|------|------|------|
| trkptから遠い場所（>閾値） | ゴールの後ろへ延長 | acptを**末尾に追加** → 前のacpt/境界とのセグメントを計算 → wpt自動検出 |
| trkptに近い場所（≤閾値） | 既存セグメントを分割 | ダイアログ表示（下記） |
| 既存のacpt（右クリック） | acpt削除 | acptを削除 → 前後を再接続 → wpt再検出 |
| 既存のwpt | wpt削除 | wptを削除（現行と同じ） |

**ダイアログ（trkptに近い場合）：**
- 「ゴールを延長する」→ 現在の末尾acptからクリック地点へルートを追加（末尾追加）
- 「アンカーポイントを挿入する」→ 最寄りtrkptの位置にacptを**挿入**（セグメント分割）
- 「ターンポイントを追加する」→ 最寄りtrkptにwptを挿入（現行ロジック流用）

往復ルート・ループルートなど既存trkptに近い場所でゴールを延長したい場合も「ゴールを延長する」で対応できる。

**「trkptに近い」の閾値：** 画面上20px相当（メートル換算はズームレベルで動的計算）

**ダイアログUI：** Leaflet の `L.popup` を使ってマップ上にカスタムポップアップを表示。ブラウザのalert/confirmは使わない。

### acptドラッグ

- ドラッグ開始：前後境界点への点線プレビューを表示
- ドラッグ完了：`acpt_drag_end` イベントを Python へ送信
  1. `_undo_state` に現在状態を保存
  2. 前後セグメントを Valhalla /route で再計算
  3. 影響セグメント内のwptを全削除 → wpt自動再検出
  4. rerun

### リルート時のセグメント境界

```
前の境界 = 直前のacpt → なければ直前のwpt → なければスタート地点
後の境界 = 直後のacpt → なければ直後のwpt → なければゴール地点
```

Valhalla /route の呼び出し：`境界A（前） → acpt → 境界B（後）`

境界がwptやスタート/ゴールの場合もその座標をそのままValhalla に渡す。

### wpt自動検出タイミング

以下の操作後、**影響を受けたセグメント上でwptを自動検出**する：
- acpt追加（新セグメント生成時）
- acptドラッグ完了
- acpt削除（前後を再接続した新セグメント）

検出ロジックは現行の `calculate_bearing` / `angle_diff` / `fetch_intersection_names` / `with_name` をそのまま流用。

**セグメント内のwpt処理：**
1. 境界点の内側にある既存wptを全削除
2. 新trkpt列でwptを自動検出・挿入
3. 右パネルに通知：「セグメントを再計算しました。N件のwptを再検出しました」
4. ※ユーザーが手動で付けた名前は失われる（仕様）

### undo

- 操作前の `{acpts, active_points, edit_turns}` を `session_state["_undo_state"]` に1世代保存
- 「↩ 元に戻す」ボタンで復元・rerun
- undo後はundo不可（1回のみ）

---

## マップコンポーネント仕様（Leaflet カスタム）

### 実装方式

```python
# leaflet_map.py
def build_map_html(data: dict) -> str:
    """Leaflet.js マップHTMLを生成して返す"""

# gpxconverter.py での使い方
html = build_map_html(map_data)
event = components.html(html, height=520)
if isinstance(event, dict) and event.get("ts") != st.session_state.get("_map_event_ts"):
    st.session_state["_map_event_ts"] = event["ts"]
    # イベント処理 ...
```

**Leaflet CDN（HTML内で読み込む）：**
```html
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
```

### Python → JS（描画データ）

```python
map_data = {
    "trkpts": [[lat, lng], ...],
    "acpts": [
        {"lat": ..., "lng": ..., "trkpt_idx": 0},  # trkpt上のインデックスも保持
        ...
    ],
    "wpts": [
        {"lat": ..., "lng": ..., "trkpt_idx": 0, "name": "左折", "color": "#e74c3c"},
        ...
    ],
    "center": {"lat": ..., "lng": ...},
    "zoom": 13,
    "click_threshold_px": 20,  # trkptに「近い」と判定する画面上のピクセル距離
}
```

### JS → Python（イベント）

全イベントに `ts: Date.now()` を付与して重複処理を防ぐ。
**マップ移動（map_move）はイベントとして送らない。** 他のイベント送信時に現在の center / zoom を乗せて一緒に送る。

```javascript
// 全イベント共通フィールド
{ ts: Date.now(), center: {lat, lng}, zoom: N, ...イベント固有フィールド }

// trkptから遠い場所クリック → acpt追加
{ type: "click_empty", ts, center, zoom, lat, lng }

// trkptに近い場所クリック → ダイアログ表示（JS側でpopup、選択結果を送信）
{ type: "dialog_result", ts, center, zoom, action: "extend"|"acpt"|"wpt", lat, lng, nearest_trkpt_idx }

// acpt右クリック → 削除
{ type: "acpt_delete", ts, center, zoom, acpt_idx }

// wptクリック → 右パネルフォーカス
{ type: "wpt_click", ts, center, zoom, wpt_idx }

// acptドラッグ完了
{ type: "acpt_drag_end", ts, center, zoom, acpt_idx, lat, lng }
```

### イベント重複防止

```python
_last_ts = st.session_state.get("_map_event_ts", 0)
if isinstance(event, dict) and event.get("ts", 0) != _last_ts:
    st.session_state["_map_event_ts"] = event["ts"]
    # イベント処理
```

---

## Python側イベントハンドラ

```
click_empty
  → acptsの末尾に {lat, lng} を追加
  → 前の境界点との間を calc_route_segment() で計算
  → trkptを更新・acptにtrkpt_idxを付与
  → 新セグメントでwptを自動検出
  → rerun

dialog_result: action="extend"
  → click_empty と同じ処理（末尾に追加）

dialog_result: action="acpt"
  → 最寄りtrkptの位置にacptを挿入（セグメント分割）
  → 前後セグメントを再計算 → wpt再検出 → rerun

dialog_result: action="wpt"
  → 現行の「最寄りtrkptにwptを挿入」ロジックを流用
  → rerun

acpt_delete
  → _undo_state に保存
  → acptを削除
  → 前後境界を再接続（calc_route_segment()）
  → 削除acptが挟んでいたセグメントのwptを再検出
  → rerun

wpt_click
  → session_state["_focus_wpt_idx"] を設定
  → rerun（現行と同じ、テキストボックスにフォーカス）

acpt_drag_end
  → _undo_state に保存
  → acptの座標を更新
  → 前後セグメントを calc_route_segment() で再計算
  → 影響セグメントのwptを全削除 → 再検出
  → rerun

全イベント共通
  → event["center"] / event["zoom"] を _map_center / _map_zoom に保存
```

---

## Valhalla /route ラッパー（routing.py）

```python
import requests
import polyline  # pip install polyline

def calc_route_segment(
    points: list[tuple[float, float]],  # [(lat, lng), ...] 2点以上
    costing: str = "bicycle",
) -> list[tuple[float, float]]:
    """
    複数点を経由する道路沿いtrkpt列を返す。
    失敗時は points 間を直線補間してフォールバック。
    """
    payload = {
        "locations": [{"lat": lat, "lon": lng} for lat, lng in points],
        "costing": costing,
    }
    try:
        r = requests.post(
            "https://valhalla1.openstreetmap.de/route",
            json=payload, timeout=30,
        )
        r.raise_for_status()
        # Valhallaのデフォルトはencoded polyline（精度6）
        encoded = r.json()["trip"]["legs"][0]["shape"]
        coords = polyline.decode(encoded, geojson=False)  # [(lat, lng), ...]
        return coords
    except Exception:
        # フォールバック：直線補間
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
    # 前の境界
    prev = None
    for a in sorted(acpts, key=lambda x: x["trkpt_idx"], reverse=True):
        if a["trkpt_idx"] < acpt_trkpt_idx:
            prev = (a["lat"], a["lng"]); break
    if prev is None:
        for w in sorted(wpts, key=lambda x: x["trkpt_idx"], reverse=True):
            if w["trkpt_idx"] < acpt_trkpt_idx:
                prev = (w["lat"], w["lng"]); break
    if prev is None:
        prev = tuple(trkpts[0])

    # 後の境界
    nxt = None
    for a in sorted(acpts, key=lambda x: x["trkpt_idx"]):
        if a["trkpt_idx"] > acpt_trkpt_idx:
            nxt = (a["lat"], a["lng"]); break
    if nxt is None:
        for w in sorted(wpts, key=lambda x: x["trkpt_idx"]):
            if w["trkpt_idx"] > acpt_trkpt_idx:
                nxt = (w["lat"], w["lng"]); break
    if nxt is None:
        nxt = tuple(trkpts[-1])

    return prev, nxt
```

---

## セッション状態

| キー | 内容 | 新規/既存 |
|------|------|------|
| `app_mode` | `"select"` / `"new_route"` / `"gpx"` | 新規 |
| `acpts` | `[{lat, lng, trkpt_idx}, ...]` | 新規 |
| `active_points` | trkpt座標リスト `[[lat, lng], ...]` | **既存流用** |
| `edit_turns` | wptリスト `[{lat, lng, trkpt_idx, name, delta, ...}]` | **既存流用** |
| `_undo_state` | `{acpts, active_points, edit_turns}` の1世代前 | 新規 |
| `_map_center` | `{lat, lng}` | **既存流用** |
| `_map_zoom` | ズームレベル（int） | **既存流用** |
| `_map_event_ts` | 最後に処理したイベントのtimestamp（重複防止） | 新規 |
| `_focus_wpt_idx` | フォーカスするwptのリストインデックス | **既存流用** |
| `_skip_map_center_save` | マップ中心の上書き抑制フラグ | **既存流用** |

---

## ファイル構成

```
gpx-navi/
├── gpxconverter.py      # メインアプリ（改修）
├── leaflet_map.py       # Leaflet HTMLビルダー（新規）
├── routing.py           # Valhalla /route ラッパー（新規）
├── index.html           # ナビ再生（変更なし）
└── check_api_status.py  # API確認ツール（変更なし）
```

---

## 実装順序

1. `routing.py` — `calc_route_segment` と `find_segment_boundaries` を実装・単体テスト
2. `leaflet_map.py` — 表示のみ（イベントなし）で `st_folium` を置き換え、trkpt/acpt/wptを描画
3. JS→Pythonのイベント基盤（ts付きイベント、重複防止）
4. `click_empty` → acpt追加 → セグメント計算 → wpt自動検出
5. `acpt_drag_end` → セグメント再計算 → wpt再検出
6. `acpt_delete` → 再接続 → wpt再検出
7. `dialog_result` → acpt/wpt選択ダイアログ（Leaflet popup）
8. `wpt_click` → テキストボックスフォーカス（現行ロジック流用）
9. undo機能
10. 開始画面（モード選択） + GPXモードとの統合テスト
11. 標高補正・マップマッチングをGPXモードで維持確認

---

## 既存コードの流用方針

| 既存関数/変数 | 流用方法 |
|------|------|
| `fetch_intersection_names(turns, radius)` | wpt自動検出時にそのまま呼ぶ |
| `fetch_spot_name(lat, lon, radius)` | 同上 |
| `with_name(trkpt, iname)` | 同上 |
| `wpt_style(t)` | wptの矢印・色の決定 |
| `nearest_trkpt_index(lat, lng, points)` | click_threshold判定に流用 |
| `calculate_bearing` / `angle_diff` | wpt自動検出のターン角計算 |
| `active_points` | trkpt列として流用（型は `[[lat, lng], ...]` のまま） |
| `edit_turns` | wptリストとして流用 |
| 右パネル（wpt一覧・名前入力・削除ボタン） | ほぼそのまま |
| GPX出力ロジック | 変更なし |
| 標高補正UI・マップマッチングUI | GPXモードで維持 |
| `_focus_wpt_idx` / `_skip_map_center_save` | セッション状態キーそのまま流用 |

---

## 注意事項

- Valhalla `/route` のレスポンスの `shape` フィールドは**encoded polyline（精度6）**。`polyline` ライブラリで `polyline.decode(s, geojson=False)` でデコードすると `[(lat, lng), ...]` が得られる。`pip install polyline` が必要。
- Leaflet の座標系は `[lat, lng]` 順。GeoJSON は `[lng, lat]` 順。混在に注意。
- `components.html` は呼び出しのたびにiframeが再生成される。マップの描画データはすべて HTML 文字列に埋め込んで渡す（WebSocket等は使わない）。
- `Streamlit.setComponentValue` を呼ぶと必ずrerunが発生する。map_moveはイベントとして送らず、他イベントに乗せることでrerunを最小化する。
