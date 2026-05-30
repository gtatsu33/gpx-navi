# gpxconverter マップ刷新 & 新規ルート作成機能 — 要件仕様 / 実装プロンプト

## 概要

`gpxconverter.py` のマップコンポーネントを `st_folium` から `declare_component` 製のフル Leaflet.js コンポーネントに置き換える。

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
| **trkpt** | ルートを構成する座標点列。OSRM /route の計算結果 |
| **acpt（アンカーポイント）** | ルートの形を決める中継点。ユーザーが配置・ドラッグする。ナビ案内なし |
| **wpt（ウェイポイント）** | ナビ案内点。名前付き。`edit_turns` リストで管理。`wpt[0]` = スタート、`wpt[-1]` = ゴール |
| **スタート** | `wpt[0]` かつ `acpts[0]`。緑ピン（S）で表示 |
| **ゴール** | `wpt[-1]` かつ `acpts[-1]`。赤ピン（G）で表示 |
| **セグメント** | 隣接する2つのacpt間のtrkpt列 |
| **境界点** | セグメントの端点。acpt > wpt > trkpt先頭/末尾 の優先順で決まる |

---

## アプリ全体フロー

```
┌─────────────────────────────────────────────┐
│                  開始画面                    │
│  [ファイルをドロップ or 参照...]              │
│              ── または ──                    │
│         [🗺️ 新規ルートを作成する]            │
└──────────┬──────────────────────────────────┘
           │ GPXあり                  │ 新規ボタン
   ┌───────▼───────────────────┐     │
   │ GPX読み込み処理             │     │
   │ マップマッチング（任意）     │     │
   │ 標高補正（任意）            │     │
   │ trkpt確定・S/G をacpt+wpt設定│     │
   └───────┬───────────────────┘     │
           └─────────────────────────┘
                       │
         ┌─────────────▼──────────────────────┐
         │      ルート編集（常時）               │
         │  ・空マップからクリックでルート作成可  │
         │  ・acpt追加/ドラッグ/削除            │
         │  ・wpt追加/名前編集/削除             │
         │  ・acptを全削除すれば空状態に戻れる   │
         └─────────────┬──────────────────────┘
                       │
               ┌───────▼────────┐
               │    GPX 出力     │
               └────────────────┘
```

2つの開始方法は「最初にGPXを読み込むか否か」の違いだけで、その後は同一の編集UIに入る。モード切り替え機能は持たない。

---

## 開始方法

### GPXを読み込む場合

1. ファイルアップローダーにGPXをドロップ
2. マップマッチング（Valhalla `/trace_attributes`）・標高補正（任意、現行UIを流用）
3. trkptを確定。**acptはS/Gの2点のみ初期配置**
4. wpt初期化：
   - **GPXにwptあり**：既存wptをそのまま使用
   - **GPXにwptなし**：detect_turns 自動実行 → 先頭に `{name:"スタート"}`、末尾に `{name:"ゴール"}` を挿入
5. ルート編集へ

### 新規ルートを作成する場合

- 「🗺️ 新規ルートを作成する」ボタンを押すと空マップで編集開始
- マップマッチング・標高補正はスキップ
- **1点目クリック**：`trkpts = [click]`、スタートwpt かつ acpt を作成
- **2点目クリック**：OSRM でセグメント計算 → ゴールwpt かつ acpt を追加
- **3点目以降**（click_empty）：旧ゴールを wpt から外し acpt として残す → 新ゴールを末尾に追加

### 編集画面からの戻り方

編集画面のページ上部（タイトル行付近）に「↩ 編集を破棄して戻る」ボタンを設置する。
押下時に確認ダイアログを表示する：
- 「🏠 スタート画面に戻る」→ `_STATE_KEYS` + `_file_key` + `_new_route_mode` を全リセットしてスタート画面へ
- 「✏️ 編集を続ける」→ ダイアログを閉じて何もしない

これにより「別のGPXファイルに切り替えたい」「新規作成に切り替えたい」もこのボタン経由で対応できる。

**`_new_route_mode` フラグの用途：**
スタート画面の分岐制御と初期化処理（map matchingスキップ等）にのみ使用する。
編集画面のUIは両パスで完全に同一とし、フラグによる分岐は設けない。

### GPX出力（GPXなしの場合）

`build_enhanced_gpx(gpx_content_str=None, ...)` に対応が必要。`None` の場合は空の GPX トラックをゼロ生成する。

```python
if gpx_content_str:
    enhanced = gpxpy.parse(gpx_content_str)
else:
    enhanced = gpxpy.GPX()
    track = gpxpy.gpx.GPXTrack()
    seg   = gpxpy.gpx.GPXTrackSegment()
    for pt in (matched_points or active_points):
        seg.points.append(gpxpy.gpx.GPXTrackPoint(pt[0], pt[1]))
    track.segments.append(seg)
    enhanced.tracks.append(track)
```

---

## ルート編集モード（両モード共通・常時）

### マップ表示

| 要素 | 表示 |
|------|------|
| ルートポリライン | 青いライン（trkpt列） |
| acpt | 小さな白抜き●（ドラッグ可、右クリックで削除） |
| wpt（中間） | 大きな●（現行と同じ色分け・矢印アイコン）（ドラッグ不可） |
| スタート（`wpt[0]`） | 緑ピン（S）。矢印アイコンなし |
| ゴール（`wpt[-1]`） | 赤ピン（G）。矢印アイコンなし |
| ドラッグ中の予告ライン | acptから前後境界点への点線 |

### クリック挙動

「trkptに近いか」の判定がacpt挿入か末尾追加かの分岐点になる。

| クリック位置 | 判断 | 動作 |
|------|------|------|
| trkptから遠い場所（>閾値） | ゴールの後ろへ延長 | acptを**末尾に追加** → 前のacpt/境界とのセグメントを計算 → route_modified=True |
| trkptに近い場所（≤閾値） | 既存セグメントを分割 | ダイアログ表示（下記） |
| 既存のacpt（右クリック） | acpt削除 | acptを削除 → 前後を再接続 → route_modified=True |
| 既存のwpt | wpt削除 | wptを削除（現行と同じ） |

**ダイアログ（trkptに近い場合）：**
- 「ゴールを延長する」→ 現在の末尾acptからクリック地点へルートを追加（末尾追加）
- 「アンカーポイントを挿入する」→ 最寄りtrkptの位置にacptを**挿入**（セグメント分割）
- 「ターンポイントを追加する」→ 最寄りtrkptにwptを挿入（現行ロジック流用）
- 「キャンセル」→ 何もしない

往復ルート・ループルートなど既存trkptに近い場所でゴールを延長したい場合も「ゴールを延長する」で対応できる。

**「trkptに近い」の閾値：** 画面上20px相当（メートル換算はズームレベルで動的計算）

**ダイアログUI：** Leaflet の `L.popup` を使ってマップ上にカスタムポップアップを表示。ブラウザのalert/confirmは使わない。

### acptドラッグ

- ドラッグ開始：前後境界点への点線プレビューを表示
- ドラッグ完了：`acpt_drag_end` イベントを Python へ送信
  1. `_undo_state` に現在状態を保存
  2. 前後セグメントを OSRM /route で再計算
  3. `route_modified = True`
  4. rerun

### リルート時のセグメント境界

```
前の境界 = 直前のacpt → なければ直前のwpt → なければスタート地点
後の境界 = 直後のacpt → なければ直後のwpt → なければゴール地点
```

OSRM /route の呼び出し：`境界A（前） → acpt → 境界B（後）`

境界がwptやスタート/ゴールの場合もその座標をそのままOSRMに渡す。

**`_adj_idx()` ヘルパー仕様（gpxconverter.py内）：**

```python
def _adj_idx(trkpt_idx, acpts, wpts, n_pts, direction):
    """
    acpts と wpts の両方を境界候補として、
    direction < 0 → trkpt_idx より前の最大インデックスを返す
    direction > 0 → trkpt_idx より後の最小インデックスを返す
    """
    boundary_idxs = [a["trkpt_idx"] for a in acpts] + [t["index"] for t in wpts]
    if direction < 0:
        cands = [i for i in boundary_idxs if i < trkpt_idx]
        return max(cands) if cands else 0
    else:
        cands = [i for i in boundary_idxs if i > trkpt_idx]
        return min(cands) if cands else n_pts - 1
```

呼び出し元（`acpt_drag_end`・`acpt_delete`・`dialog_result: action="acpt"`）でも `wpts` を第3引数として渡す。

### wpt検出タイミング

**自動実行（GPX読み込み時のみ）：**
- GPXファイル読み込み後、`detect_turns` + `fetch_intersection_names` を自動実行してwptを設定する

**手動実行（「wpt検出」ボタン押下時）：**
- `detect_turns` + `fetch_intersection_names` を実行し、全ルート上のwptを再設定する
- `route_modified` フラグを `False` にリセットする
- ※ユーザーが手動で付けた名前は失われる（仕様）

**ルート変更操作時（acpt追加・ドラッグ・削除、ゴール延伸）：**
- `detect_turns` は実行しない
- `route_modified = True` をセットする

**スタート・ゴール wpt の保護ルール：**
- `detect_turns` によって `wpt[0]`・`wpt[-1]` が上書き・削除されることはない

### undo

- 操作前の `{acpts, active_points, edit_turns}` を `session_state["_undo_state"]` に1世代保存
- 「↩ 元に戻す」ボタンで復元・rerun
- undo後はundo不可（1回のみ）

---

## マップコンポーネント仕様（Leaflet カスタム）

### 実装方式

`declare_component(path="frontend/")` を使う。`frontend/index.html` が Leaflet コンポーネント本体。

```python
# leaflet_map.py
_component_func = None

def _get_component():
    global _component_func
    if _component_func is None:
        # モジュールレベルではなく初回呼び出し時に登録する。
        # モジュールレベルで呼ぶと Streamlit のモジュールウォッチャーが
        # ScriptRunContext なしでインポートするためレジストリ登録が失敗する。
        _component_func = components.declare_component(
            "gpx_map",
            path=str(Path(__file__).parent / "frontend"),
        )
    return _component_func

def render_map(data: dict, height: int = 520, key: str = "gpx_map"):
    payload = { ... }
    return _get_component()(data=payload, height=height, key=key, default=None)

# gpxconverter.py での使い方
event = render_map(data=map_data, height=520, key="gpx_map")
if isinstance(event, dict) and event.get("ts", 0) != st.session_state.get("_map_event_ts", 0):
    st.session_state["_map_event_ts"] = event["ts"]
    # イベント処理 ...
```

**JS→Python 通信（postMessage プロトコル）：**

Streamlit 1.51.0 以降は `isStreamlitMessage: true` が必須。ない場合メッセージが無視される。

```javascript
// frontend/index.html
function _post(obj) {
    obj.isStreamlitMessage = true;
    window.parent.postMessage(obj, '*');
}
function _setH(h) { _post({type:'streamlit:setFrameHeight', height:h}); }
function _setV(v) { _post({type:'streamlit:setComponentValue', value:v, dataType:'json'}); }

// 起動時
_post({type:'streamlit:componentReady', apiVersion:1});

// Python からの描画データ受信
window.addEventListener('message', function(e) {
    if (e.data && e.data.type === 'streamlit:render') {
        renderMap(e.data.args.data, e.data.args.height || 520);
    }
});
```

**Leaflet はローカルファイルを使う（CDN 不可）：**

`declare_component` の iframe は CDN 読み込みがタイムアウトすると `componentReady` が届かずコンポーネントが表示されない。`frontend/leaflet.js` / `frontend/leaflet.css` としてローカルに配置し、`<script src="leaflet.js">` で読み込む。

### Python → JS（描画データ）

```python
map_data = {
    "trkpts": [[lat, lng], ...],
    "acpts": [
        {"lat": ..., "lng": ..., "trkpt_idx": 0},
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
{ type: "dialog_result", ts, center, zoom, action: "extend"|"acpt"|"wpt"|"cancel", lat, lng, nearest_trkpt_idx }

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
click_empty  ※ dialog_result: action="extend" も同じ処理
  → 前の境界点 = 最後のacpt座標。acptがなければ trkpts[-1]（ルート末尾）
  → 前の境界点 ～ クリック地点を calc_route_segment() で計算
  → trkptを末尾に追加
  → 旧ゴール（wpt[-1]）を wpt リストから外し、acpt として残す
  → 新ゴール {name:"ゴール", trkpt_idx:新しいlast} を wpt末尾・acpt末尾に追加
  → route_modified = True
  → rerun

dialog_result: action="acpt"
  → 最寄りtrkptの位置にacptを挿入（セグメント分割）
  → 前後セグメントを再計算
  → route_modified = True
  → rerun

dialog_result: action="wpt"
  → 現行の「最寄りtrkptにwptを挿入」ロジックを流用
  → rerun

dialog_result: action="cancel"
  → 何もしない

acpt_delete
  → _undo_state に保存
  → 削除対象が中間acptの場合：
      前後境界を再接続（calc_route_segment()）
  → 削除対象が S（先頭acpt）の場合：
      _adj_idx(old_S, acpts, wpts, n, +1) で「次の境界」を探す
      ┌ 境界がacptの場合（従来通り）：
      │   そのacptを新Sに昇格
      │   offset = そのacptのtrkpt_idx
      │   trkpts = trkpts[offset:]
      │   全acpt["trkpt_idx"] -= offset
      │   全wpt["index"] -= offset、index <= 0 のものは削除
      │   新Sに "スタート" wptを先頭に挿入
      └ 境界がwptの場合（新仕様）：
          そのwptをacptに昇格（acpts先頭に追加）かつ新Sとして扱う
          offset = そのwptのindex
          trkpts = trkpts[offset:]
          全acpt["trkpt_idx"] -= offset
          全wpt["index"] -= offset、index <= 0 のものは削除
          昇格したwptの name を "スタート" に変更し wpt[0] とする
  → 削除対象が G（末尾acpt）の場合：
      _adj_idx(old_G, acpts, wpts, n, -1) で「前の境界」を探す
      ┌ 境界がacptの場合（従来通り）：
      │   そのacptを新Gに昇格
      │   trkpts = trkpts[:新Gのtrkpt_idx + 1]
      │   新Gに "ゴール" wptを末尾に追加
      └ 境界がwptの場合（新仕様）：
          そのwptをacptに昇格（acpts末尾に追加）かつ新Gとして扱う
          trkpts = trkpts[:そのwptのindex + 1]
          昇格したwptの name を "ゴール" に変更し wpt[-1] とする
          それより後ろのwptは削除
  → acpts が 1 つになった場合（境界が見つからない場合）：スタートのみ状態
      trkpts = [S座標のみ]、wpts = [S_wpt のみ]
  → route_modified = True
  → rerun

wpt_click
  → session_state["_focus_wpt_idx"] を設定
  → rerun（現行と同じ、テキストボックスにフォーカス）

acpt_drag_end
  → _undo_state に保存
  → acptの座標を更新
  → 前後セグメントを calc_route_segment() で再計算
    ※ 境界点は「直前/直後のacpt → なければ直前/直後のwpt → なければスタート/ゴール」の優先順
    ※ S（先頭acpt）ドラッグ：後のセグメントのみ再計算
       後の境界 = 直後のacptまたはwpt（最寄り優先）
       境界より後ろの区間（trkpt・wpt）は保持される
    ※ G（末尾acpt）ドラッグ：前のセグメントのみ再計算
       前の境界 = 直前のacptまたはwpt（最寄り優先）
       境界より前の区間（trkpt・wpt）は保持される
    ※ 中間acptドラッグ：前後セグメントをともに再計算
  → _adj_idx() は acpts と wpts の両方を境界候補として検索する
  → route_modified = True
  → rerun

全イベント共通
  → event["center"] / event["zoom"] を _map_center / _map_zoom に保存
```

---

## ルーティング API（routing.py）

**使用API: OSRM public instance** (`http://router.project-osrm.org`)

Valhalla (`valhalla1.openstreetmap.de`) は road データが空で全リクエストが 400 エラーになるため使用不可。

```python
import requests

def calc_route_segment(
    points: list[tuple[float, float]],  # [(lat, lng), ...] 2点以上
    costing: str = "bicycle",           # 現在未使用（OSRMは bike 固定）
) -> list[tuple[float, float]]:
    """
    複数点を経由する道路沿いtrkpt列を返す。
    失敗時は points をそのまま返す（直線フォールバック）。
    """
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
            prev = (a["lat"], a["lng"]); break
    if prev is None:
        for w in sorted(wpts, key=lambda x: x["trkpt_idx"], reverse=True):
            if w["trkpt_idx"] < acpt_trkpt_idx:
                prev = (w["lat"], w["lng"]); break
    if prev is None:
        prev = tuple(trkpts[0])

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

> **注意:** Valhalla は `trace_attributes`（マップマッチング）でも使用しているが、
> そちらも同様に動作不可の可能性がある。マップマッチング処理は GPX 読み込みモードの
> オプション機能なので、動作しない場合はスキップされる。

---

## セッション状態

| キー | 内容 | 新規/既存 |
|------|------|------|
| `_new_route_mode` | `True` のとき新規ルート作成モード（GPXなし） | 新規 |
| `acpts` | `[{lat, lng, trkpt_idx}, ...]` | 新規 |
| `active_points` | trkpt座標リスト `[[lat, lng], ...]` | **既存流用** |
| `edit_turns` | wptリスト `[{lat, lng, trkpt_idx, name, delta, ...}]` | **既存流用** |
| `_undo_state` | `{acpts, active_points, edit_turns}` の1世代前 | 新規 |
| `_map_center` | `{lat, lng}` | **既存流用** |
| `_map_zoom` | ズームレベル（int） | **既存流用** |
| `_map_event_ts` | 最後に処理したイベントのtimestamp（重複防止） | 新規 |
| `_focus_wpt_idx` | フォーカスするwptのリストインデックス | **既存流用** |
| `_skip_map_center_save` | マップ中心の上書き抑制フラグ | **既存流用** |
| `route_modified` | ルート変更後・wpt検出未実施フラグ（bool） | 新規 |
| `_confirm_back` | 「戻る」確認ダイアログ表示フラグ（bool） | 新規 |

**状態リセットのトリガー：**

| 条件 | リセット対象 |
|------|------|
| GPXファイル名変更（`_file_name` キーで検知） | `_STATE_KEYS` 全体 |
| 新規ルートモードボタン押下 | `_STATE_KEYS` 全体 + `_file_key` クリア |
| 「スタート画面に戻る」押下 | `_STATE_KEYS` 全体 + `_file_key` + `_new_route_mode` クリア |

`_new_route_mode` は `_STATE_KEYS` に含めない（リセット後も新規モードを維持するため）。
新規ルートモードでは `uploaded` は `None` のまま扱い、GPX依存の処理（マップマッチング・標高補正・`_has_wpts` 判定）をすべてスキップする。

---

## ファイル構成

```
gpx-navi/
├── gpxconverter.py        # メインアプリ（改修）
├── leaflet_map.py         # declare_component ラッパー（新規）
├── routing.py             # OSRM /route ラッパー（新規）
├── frontend/
│   ├── index.html         # Leaflet.js コンポーネント本体
│   ├── leaflet.js         # Leaflet 1.9.4（ローカル）
│   └── leaflet.css        # Leaflet 1.9.4（ローカル）
├── index.html             # ナビ再生（変更なし）
└── check_api_status.py    # API確認ツール（変更なし）
```

---

## 実装フェーズ

### フェーズ1：土台（表示まで）

**実装内容**
1. `routing.py` — `calc_route_segment` と `find_segment_boundaries` を実装
2. `leaflet_map.py` — 表示のみ（イベントなし）で `st_folium` を置き換え、trkpt/acpt/wptを描画

**テスト手順**
- 既存のGPXファイルを読み込み、ルートのポリラインが地図に青線で表示されることを確認
- wptマーカーが正しい色・位置に表示されることを確認
- 地図をパン・ズームしてからサイドパネルのwptボタンを押し、地図位置がリセットされないことを確認（localStorageの動作確認）
- wptボタンを押すと地図の中心がそのwpt位置に移動することを確認（`_map_center` の引き継ぎ）
- ターミナルで `routing.py` を直接実行して、2点間のtrkptが返ることを確認：
  ```python
  from routing import calc_route_segment
  pts = calc_route_segment([(35.681, 139.767), (35.690, 139.760)])
  print(len(pts), pts[0], pts[-1])  # 数十点以上返れば成功
  ```

---

### フェーズ2：クリックでルート作成

**実装内容**
3. JS→Pythonイベント基盤（ts付きイベント、重複防止）
4. `click_empty` → acpt追加 → セグメント計算 → route_modified=True

**テスト手順**（GPX読み込みモードで実施）

> フェーズ4（開始画面）実装前は GPX ファイルを読み込んだ状態でテストする。
> 新規ルートモードのテストはフェーズ4完了後に実施する。

- GPXファイルを読み込み、地図が表示された状態で**ルート末尾より遠い場所**をクリック → ルートポリラインが延伸されることを確認
- さらに2回以上クリック → クリックのたびにルートが伸び続けることを確認
- クリックのたびに旧ゴールがacpt（白抜き●）として残り、新ゴール（赤ピン）が移動することを確認
- wptリストにはスタート・ゴールのみ残り、中間wptが増えないことを確認（detect_turnsは走らない）
- 同じ場所を素早く2回クリックしても1回分しか処理されないことを確認（`ts` 重複防止）
- クリック後のrerunでマップの表示位置・ズームがリセットされないことを確認（`localStorage` が効いている）
- OSRMに繋がらない状況（機内モード等）でクリックしても、直線補間でルートが引かれることを確認（フォールバック）

---

### フェーズ3：フル編集

**実装内容**
5. `acpt_drag_end` — ドラッグ完了 → 前後セグメント再計算 → route_modified=True（S/G ドラッグは片側のみ再計算）
6. `acpt_delete` — 右クリック削除。中間acptは前後再接続、S/G削除は昇格ロジック → route_modified=True
7. `dialog_result` — 4択ポップアップ（ゴールを延長 / acpt挿入 / ターンポイント追加 / キャンセル）
8. `wpt_click` — テキストボックスフォーカス（現行ロジック流用）
9. undo機能
10. 「強化GPX生成」押下時の route_modified 警告ダイアログ（[続行] [キャンセル]）

※「wpt検出」ボタンは Phase 2 作業中に実装済み

**テスト手順**
- acptをドラッグして放すと、前後のルートが再計算されることを確認
- ドラッグ後、undo ボタンで元の形に戻ることを確認（1回のみ）
- undo後にさらにundoしても何も起きない（2回目は不可）ことを確認
- acptを右クリック → 削除 → 前後が直接つながるルートに再計算されることを確認
- ルート変更操作後に「強化GPX生成」を押すと警告ダイアログ（「ナビ案内点がルートと一致していない可能性があります」→ [続行] [キャンセル]）が表示されることを確認
- 「wpt検出」ボタンを押すと detect_turns が実行され、wptが更新されることを確認
- 「wpt検出」後に「強化GPX生成」を押すと警告が出ないことを確認
- 既存trkptの近くをクリック → 4択ダイアログが表示されることを確認
  - 「ゴールを延長する」→ ルートが末尾から伸びることを確認
  - 「アンカーポイントを挿入する」→ そのtrkpt位置にacptが増え、セグメントが分割されることを確認
  - 「ターンポイントを追加する」→ 右パネルにwptが追加されることを確認
  - 「キャンセル」→ 何も変化しないことを確認
- S（スタート）acptを右クリック削除：
  - 次の境界がacptの場合 → そのacptがスタートに昇格し、trkptの前方が切り捨てられることを確認
  - 次の境界がwptの場合 → そのwptがacptに昇格しスタートになり、名前が「スタート」に変わることを確認
- G（ゴール）acptを右クリック削除：
  - 前の境界がacptの場合 → そのacptがゴールに昇格し、trkptの後方が切り捨てられることを確認
  - 前の境界がwptの場合 → そのwptがacptに昇格しゴールになり、名前が「ゴール」に変わることを確認
- acptが1つだけ（かつwptもない）の状態で削除 → ルートが空になることを確認
- Sをドラッグ → スタート位置が移動し後方セグメントのみ再計算されることを確認
  - wptが複数ある場合：最初のwptが境界となり、そのwptより後ろのwptが保持されることを確認
- Gをドラッグ → ゴール位置が移動し前方セグメントのみ再計算されることを確認
  - wptが複数ある場合：最後のwptが境界となり、そのwptより前のwptが保持されることを確認
- **往復ルートシナリオ**：A→B→Aと折り返すルートを作り、復路でゴールを延長できることを確認
- **ループシナリオ**：スタート付近に戻ってきたとき、ダイアログで「ゴールを延長する」を選べばループが閉じられることを確認
- wptマーカーをクリック → 右パネルの対応テキストボックスにフォーカスが当たることを確認

---

### フェーズ4：アプリ統合

**実装内容**
11. 開始画面：ファイルアップローダー ＋「🗺️ 新規ルートを作成する」ボタンを並べて表示
    - ファイルなし かつ `_new_route_mode` が False → 開始画面を表示して `st.stop()`
    - ボタン押下 → `_new_route_mode = True` + `_STATE_KEYS` リセット + rerun
    - ファイルあり → 既存GPX読み込みフロー（`_new_route_mode = False`）
12. `build_enhanced_gpx(gpx_content_str=None, ...)` 対応：GPXなし時は空トラックをゼロ生成
13. `_new_route_mode` のとき、マップマッチング・標高補正・`_has_wpts` 判定をすべてスキップ
14. 編集画面上部に「↩ 編集を破棄して戻る」ボタンを追加
    - 押下 → `_confirm_back = True` + rerun
    - ダイアログ：「🏠 スタート画面に戻る」→ 全リセット / 「✏️ 編集を続ける」→ フラグクリア
    - `_new_route_mode` による編集画面のUI分岐を廃止し、両パスで同一コードを使用

**テスト手順（GPXを読み込む場合）**
- 起動時にファイルアップローダーと「新規ルートを作成する」ボタンが表示されることを確認
- GPXファイルをアップロード → マップマッチング・標高補正のUIが表示されることを確認
- trkpt確定後、acptを追加・ドラッグしてルートを変形できることを確認
- wptを編集してGPX出力 → ナビアプリで開けることを確認（エンドツーエンドテスト）

**テスト手順（新規ルートを作成する場合）**
- 「🗺️ 新規ルートを作成する」ボタンを押す → 空のマップが表示されることを確認
- 1点目クリック → スタートピン（緑S）のみ表示されることを確認
- 2点目クリック → ルートポリラインが引かれ、ゴールピン（赤G）が表示されることを確認
- 3点目以降クリック → ルートが延長され、旧ゴールがacpt（白●）に変わることを確認
- acptドラッグ・削除・wpt検出・GPX出力がGPX読み込みの場合と同様に動くことを確認
- 新規作成したルートをGPX出力 → ファイルが正常にダウンロードでき、trkptとwptが含まれることを確認
- 新規ルート作成中にGPXファイルをアップロード → 新規ルートの状態がリセットされGPX読み込みに切り替わることを確認

**テスト手順（共通）**
- フェーズ1〜3で確認した機能が両方の開始方法で壊れていないことを一通り再確認

---

## 既存コードの流用方針

| 既存関数/変数 | 流用方法 |
|------|------|
| `fetch_intersection_names(turns, radius)` | 「wpt検出」ボタン押下時に呼ぶ |
| `fetch_spot_name(lat, lon, radius)` | 同上 |
| `with_name(trkpt, iname)` | 同上 |
| `wpt_style(t)` | wptの矢印・色の決定 |
| `nearest_trkpt_index(lat, lng, points)` | click_threshold判定に流用 |
| `calculate_bearing` / `angle_diff` | wpt検出のターン角計算 |
| `active_points` | trkpt列として流用（型は `[[lat, lng], ...]` のまま） |
| `edit_turns` | wptリストとして流用 |
| 右パネル（wpt一覧・名前入力・削除ボタン） | ほぼそのまま |
| GPX出力ロジック | 変更なし |
| 標高補正UI・マップマッチングUI | GPXモードで維持 |
| `_focus_wpt_idx` / `_skip_map_center_save` | セッション状態キーそのまま流用 |

---

## 注意事項

- Leaflet の座標系は `[lat, lng]` 順。GeoJSON は `[lng, lat]` 順。混在に注意。
- `declare_component` の iframe は rerun のたびに再生成されない（`components.html` と異なる）。描画データは `streamlit:render` メッセージ経由で渡し、JS側でマップを再描画する。
- `setComponentValue` を呼ぶと必ず rerun が発生する。map_move はイベントとして送らず、他イベントに乗せることで rerun を最小化する。
- **マップ位置のリセット問題**：rerun 時に Python から渡す `center`/`zoom` だけに頼ると、純粋なパン/ズーム後の rerun で位置がリセットされる。対策として `localStorage` を使う：
  ```javascript
  map.on('moveend', function() {
      var c = map.getCenter();
      localStorage.setItem('gpxnavi_map_center', JSON.stringify({lat:c.lat, lng:c.lng}));
      localStorage.setItem('gpxnavi_map_zoom', String(map.getZoom()));
  });
  // 初期化時は localStorage を優先
  var sc = JSON.parse(localStorage.getItem('gpxnavi_map_center') || 'null');
  var sz = parseInt(localStorage.getItem('gpxnavi_map_zoom') || '0') || null;
  var iC = force ? pyC : (sc || pyC);
  var iZ = sz || pyZ;
  map = L.map('map').setView([iC.lat, iC.lng], iZ);
  ```
- **`declare_component` の登録タイミング**：`declare_component` をモジュールレベルで呼ぶと Streamlit のモジュールウォッチャーが ScriptRunContext なしでインポートしたときに登録が失敗し、コンポーネントが灰色のまま表示されない。必ず初回レンダリング時（関数内）で遅延登録する。
