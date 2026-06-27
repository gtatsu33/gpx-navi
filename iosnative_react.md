# gpx-navi React Native 移行ガイド

## 目的

現在の Web PWA（index.html）を React Native で iOS ネイティブアプリとして作り直す。
Web 版と Native 版を同一リポジトリで共存させ、共通ロジックを両方から再利用する。

---

## リポジトリ構成

```
gpx-navi/
├── index.html          # 既存 Web PWA（そのまま維持）
├── sw.js               # Service Worker
├── config.js           # Supabase 設定（gitignore 済み）
├── shared/             # ★ Web と Native で共有するロジック
│   ├── gpxParser.js    #   GPX パース
│   ├── geoCalc.js      #   距離・標高計算
│   └── supabaseApi.js  #   Supabase アクセス（route_files, Storage）
└── ios/                # ★ React Native プロジェクト
    ├── package.json
    ├── App.tsx
    ├── src/
    │   ├── screens/
    │   ├── components/
    │   └── storage/    #   IndexedDB → AsyncStorage 相当
    └── ios/            #   Xcode プロジェクト（自動生成）
```

---

## 共通化できる範囲

| 処理 | 共有可否 | 備考 |
|---|---|---|
| GPX パース | ◯ | `shared/gpxParser.js` をそのまま import |
| 距離・標高計算（geoDist 等） | ◯ | `shared/geoCalc.js` |
| Supabase アクセス | ◯ | supabase-js は RN でも動く |
| ルートのローカル保存 | △ | Web は IndexedDB → RN は AsyncStorage か SQLite |
| 地図表示 | ✗ | Leaflet は使えない → react-native-maps |
| UI コンポーネント | ✗ | HTML/CSS → View / StyleSheet |
| 音声案内 | △ | Web Speech API → expo-speech |

---

## 主要ライブラリ

| 機能 | ライブラリ |
|---|---|
| 地図 | `react-native-maps`（MapKit ベース） |
| GPS | `expo-location` |
| ルートストレージ | `@react-native-async-storage/async-storage` |
| 音声案内 | `expo-speech` |
| Supabase | `@supabase/supabase-js`（Web と同じ） |
| 画面常時点灯 | `expo-keep-awake` |

Expo managed workflow を使うと Xcode を直接触る機会が減り、Claude Code との相性が良い。

---

## 開発環境

1. **Node.js** — 既存環境で OK
2. **Xcode** — iOS ビルド・シミュレータに必要（macOS 必須）
3. **Expo CLI** — `npm install -g expo-cli`
4. **VS Code + Claude Code** — コードは VS Code で書く。Xcode はビルド専用

```bash
# プロジェクト作成
npx create-expo-app ios --template blank-typescript
cd ios
```

---

## 移行フェーズ

### Phase 1：共通ロジックの切り出し（Web 版を壊さない）

index.html の中から以下を `shared/` に切り出す：

- `parseGPX()` → `shared/gpxParser.js`
- `geoDist()` `calcElevationGain()` 等 → `shared/geoCalc.js`
- `openNetworkPicker()` `loadNetworkRoute()` の API 部分 → `shared/supabaseApi.js`

Web 版は切り出した関数を `<script src="shared/gpxParser.js">` で読む形に変更。
動作確認して Web 版が壊れていないことを確認してから次フェーズへ。

### Phase 2：React Native プロジェクト作成・画面構成

最低限の画面遷移を作る：

```
HomeScreen        ルート一覧（ローカル保存済みルート）
  ├─ RoutePickerModal   ネットワークからルートを選ぶ
  └─ NavigationScreen   ナビ走行画面（地図 + 指示）
```

### Phase 3：地図・GPS 実装

- `react-native-maps` でルートを polyline 描画
- `expo-location` でリアルタイム GPS 取得（バックグラウンドモードを有効化）
- 現在地とルートの照合（Web 版の計算ロジックを shared から再利用）

### Phase 4：音声案内・UX

- `expo-speech` で曲がり角案内
- `expo-keep-awake` で画面常時点灯
- バックグラウンド位置情報の許可設定（Info.plist）

### Phase 5：Supabase 連携

- `shared/supabaseApi.js` をそのまま import して動作確認
- Supabase キーは `ios/.env`（gitignore）に記述し、`react-native-dotenv` 等で読む

---

## Supabase キーの扱い（iOS）

Web 版は `config.js`（gitignore）に記述。
iOS 版は `ios/.env`（gitignore）に記述：

```
SUPABASE_URL=https://eefdcdbaqxncyvlgqxqz.supabase.co
SUPABASE_KEY=eyJ...（JWT 形式 service_role キー）
```

TestFlight / 配布時はビルド時に環境変数を注入する（EAS Build の secrets 機能）。

---

## 注意点

- **Expo Go アプリ**：開発中は実機にインストールして即確認できる（シミュレータ不要）
- **バックグラウンド位置情報**：`expo-location` の `startLocationUpdatesAsync` を使う。Info.plist に `NSLocationAlwaysAndWhenInUseUsageDescription` が必要
- **react-native-maps**：iOS では MapKit がデフォルト。Google Maps に切り替えることも可能
- **Web 版との分岐**：`shared/` のコードは `Platform.OS` を使わず純粋なロジックのみにする。プラットフォーム依存は各側（index.html / RN）で処理
