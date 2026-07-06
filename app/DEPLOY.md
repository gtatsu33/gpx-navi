# gpx-editor（HTML+JS版）Cloudflare Pages デプロイ手順

implement.txt 15章の決定に基づき、この `app/` ディレクトリを既存の
GitHub Pages配信（リポジトリ直下の `index.html`、gpx-navi本体アプリ）とは
独立した、別のCloudflare Pagesプロジェクトとしてデプロイする。

## 0. 前提: Supabase側の準備（必須）

デプロイ前に、Supabaseダッシュボードで以下を必ず済ませておくこと
（implement.txt 13章）。

1. **Publishable keyの取得**
   - Supabaseは2025年に新しいAPIキー体系（Publishable key / Secret key）を
     導入しており、旧来の `anon` / `service_role` キー（JWT形式）は
     2026年末までに廃止予定。新規実装では新形式を使う。
   - `Secret key`（全権限、`sb_secret_...`）ではなく `Publishable key`
     （公開用、`sb_publishable_...`、旧`anon`キーと同等の低権限）を使う。
   - Publishable keyは Supabaseダッシュボード → Project Settings →
     API Keys →「Publishable key」から取得する。
   - RLSポリシーの働き方は旧`anon`キーと同じ。
2. **招待ユーザーの登録（Supabase Auth、spec.txt 19章）**
   - Authentication → Sign in / Providers → Email で、一般ユーザーの
     自己サインアップ（Allow new users to sign up）を無効化する。
   - Authentication → Users →「Invite user」で、クラウド機能を使わせたい
     友人のメールアドレスを個別に登録する（マジックリンクが届く）。
   - Authentication → URL Configuration → Redirect URLs に、本番URL
     （例: `https://xxxx.pages.dev`）とローカル開発URL
     （例: `http://localhost:5173`）を登録する。
3. **RLSポリシーの設定**
   - `route_files` テーブル: `authenticated` ロールによる
     `INSERT` / `SELECT` を許可するポリシーを作成する
     （`anon`ロールへの許可は行わない）。
   - `gpx_routes` ストレージバケット: `authenticated` ロールによる
     `upload` / `download` を許可するポリシーを作成する。
   - クラウド保存・読み込みは招待ユーザー限定の機能とするため
     （spec.txt 19章）、ログイン済み（authenticated）ユーザーのみに
     絞る。ブロックしたままだと招待ユーザーもクラウド機能を使えない。
   - 【注意】この設定変更により、同一Supabaseプロジェクトをanonキーで
     読み書きしているiOSアプリ（`ios/`）は、Supabase Auth対応するまで
     一時的にクラウド機能が使えなくなる（合意済み。iOS側の対応は別途
     後日行う）。

## 1. Cloudflare Pagesプロジェクトの作成

1. Cloudflareダッシュボード → Workers & Pages → 「アプリケーションを作成」
   → 「Pages」→ 「Gitに接続」を選択し、この`gpx-navi`リポジトリを選ぶ。
2. ビルド設定を以下のように入力する:

   | 項目 | 値 |
   |---|---|
   | フレームワークプリセット | `Vite`（または「なし」でも可） |
   | ルートディレクトリ (Root directory) | `app` |
   | ビルドコマンド | `npm run build` |
   | ビルド出力ディレクトリ | `dist` |

   ※ リポジトリ直下ではなく `app` をルートディレクトリに指定することで、
   既存の`index.html`（GitHub Pages配信中のgpx-navi本体）とは無関係に
   `app/`配下だけがビルド対象になる。

## 2. 環境変数の設定

Cloudflare Pagesプロジェクト → Settings → Environment variables に、
Production・Preview両方の環境で以下を登録する。

| 変数名 | 値 |
|---|---|
| `VITE_SUPABASE_URL` | SupabaseプロジェクトのURL |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | 0.で取得した `Publishable key`（`sb_publishable_...`） |

`app/.env.local`はGit管理対象外（`.gitignore`の`*.local`に一致）のため、
ローカルの値をそのままコピーしてここに貼り付ければよい。

## 3. デプロイ実行

上記設定後、「保存してデプロイ」を実行する。以後は`main`ブランチへの
pushで自動的に再デプロイされる（Cloudflare Pagesの標準動作）。

## 4. デプロイ後の動作確認チェックリスト

- [ ] スタート画面モーダルが表示される（キャンセル不可）
- [ ] ローカルGPXファイルの読み込み → 編集画面へ遷移
- [ ] 新規ルート作成 → 地図クリックでルート延伸
- [ ] acptのドラッグ・右クリック削除
- [ ] ターンポイント検出・名称編集・削除
- [ ] 標高グラフの表示・ホバー連動
- [ ] 「💾 ルートを保存」→ 標高整合性チェック → ダウンロード
- [ ] 未ログイン状態で「☁️ クラウドからルートを選ぶ」「☁️ クラウドにも保存」が
      グレーアウトされ、ツールチップ（「招待ユーザー限定の機能です」等）が
      表示される
- [ ] スタート画面の⚙️アイコンから、招待済みメールアドレスでログインリンクを
      受け取り、クリックしてログインできる
- [ ] ログイン後、「☁️ クラウドに保存」チェック → Supabaseへのアップロード成功
- [ ] ログイン後、「🔍 ルートを選ぶ」→ ネットワークからの一覧取得・ダウンロード成功
- [ ] ログアウト操作でログイン状態が解除され、再度クラウド機能がグレーアウトされる
- [ ] 「↩ 編集を破棄して戻る」→ 破棄確認 → スタート画面に戻る

Supabase関連の項目が失敗する場合は、まず0.の招待ユーザー登録・
authenticatedロールのRLS設定を再確認する。
