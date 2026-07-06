import { createClient } from '@supabase/supabase-js'

const BUCKET = 'gpx_routes'

/**
 * spec.txt 17-5章。implement.txt 13章の通り、鍵はPublishable（公開）キーを
 * ビルド時の環境変数から読み込む。Secretキーは絶対に使わない。
 * （2025年導入の新API key体系。旧anon keyは2026年末までに廃止予定のため
 * publishable keyを使う。RLSポリシー上の扱いは旧anon keyと同じ。）
 */
export function isSupabaseConfigured() {
  return Boolean(import.meta.env.VITE_SUPABASE_URL && import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY)
}

let cachedClient = null
function defaultClient() {
  if (!cachedClient) {
    cachedClient = createClient(import.meta.env.VITE_SUPABASE_URL, import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY)
  }
  return cachedClient
}

/**
 * GPXをStorageにアップロードしてroute_filesにメタデータを登録する。
 * 戻り値: { ok, fileKey|message, errorType }
 * errorType: null=成功 / "file_key_dup" / "display_name_dup" / "other"
 *
 * 【要確認事項】(implement.txt 18-3章) supabase-js が実際に返すエラー文字列は
 * まだ実機で確認していない。Python版と同じ文字列判定を暫定で移植している。
 */
export async function uploadGpx(
  xmlStr,
  filename,
  { displayName, distanceM = null, elevationGainM = null, client = defaultClient() } = {}
) {
  const fileKey = `${filename}.gpx`

  const { error: uploadError } = await client.storage.from(BUCKET).upload(fileKey, new Blob([xmlStr], { type: 'application/gpx+xml' }))
  if (uploadError) {
    const msg = uploadError.message || String(uploadError)
    if (/already exists|duplicate|409/i.test(msg)) {
      return { ok: false, message: 'このファイルキーは既に使用されています。別のファイル名を指定してください。', errorType: 'file_key_dup' }
    }
    return { ok: false, message: `Storageへのアップロードに失敗しました: ${msg}`, errorType: 'other' }
  }

  const { error: insertError } = await client.from('route_files').insert({
    file_key: fileKey,
    display_name: displayName,
    distance_m: distanceM,
    elevation_gain_m: elevationGainM,
  })
  if (insertError) {
    await client.storage.from(BUCKET).remove([fileKey])
    const msg = insertError.message || String(insertError)
    if (/display_name/i.test(msg) && /(23505|duplicate|unique)/i.test(msg)) {
      return { ok: false, message: 'この表示名は既に使用されています。別のファイル名を入力してください。', errorType: 'display_name_dup' }
    }
    if (/(23505|duplicate)/i.test(msg)) {
      return { ok: false, message: 'このファイルキーは既に使用されています。別のファイル名を指定してください。', errorType: 'file_key_dup' }
    }
    return { ok: false, message: `DB登録に失敗しました（Storageはロールバック済み）: ${msg}`, errorType: 'other' }
  }

  return { ok: true, fileKey, errorType: null }
}

/** route_files を新しい順で全件取得する。 */
export async function listRoutes({ client = defaultClient() } = {}) {
  const { data, error } = await client.from('route_files').select('*').order('created_at', { ascending: false })
  if (error) return { ok: false, error: error.message || String(error) }
  return { ok: true, routes: data ?? [] }
}

/** Storage から GPX をダウンロードして文字列で返す。 */
export async function downloadGpx(fileKey, { client = defaultClient() } = {}) {
  const { data, error } = await client.storage.from(BUCKET).download(fileKey)
  if (error) return { ok: false, error: error.message || String(error) }
  const content = await data.text()
  return { ok: true, content }
}

/**
 * 招待制ログイン（Supabase Auth マジックリンク）。spec.txt 19章／implement.txt 13章。
 * サインアップは行わない（Supabaseダッシュボードで事前にInviteされたユーザーのみ）。
 */
export async function sendMagicLink(email, { client = defaultClient() } = {}) {
  const { error } = await client.auth.signInWithOtp({
    email,
    options: { emailRedirectTo: window.location.origin },
  })
  if (error) return { ok: false, error: error.message || String(error) }
  return { ok: true }
}

export async function getSession({ client = defaultClient() } = {}) {
  const { data } = await client.auth.getSession()
  return data.session
}

export function onAuthStateChange(callback, { client = defaultClient() } = {}) {
  const { data } = client.auth.onAuthStateChange((_event, session) => callback(session))
  return () => data.subscription.unsubscribe()
}

export async function signOut({ client = defaultClient() } = {}) {
  await client.auth.signOut()
}
