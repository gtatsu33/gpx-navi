import { useEffect, useState } from 'react'
import { downloadGpx, listRoutes } from '../lib/supabase.js'

function fmtDist(r) {
  return r.distance_m !== null && r.distance_m !== undefined ? `${(r.distance_m / 1000).toFixed(1)} km` : '---'
}
function fmtGain(r) {
  return r.elevation_gain_m !== null && r.elevation_gain_m !== undefined ? `${Math.round(r.elevation_gain_m)} m` : '---'
}

/**
 * ネットワークから読み込むダイアログ。spec.txt 3-3章。
 */
export default function NetworkPickerDialog({ onCancel, onLoaded, isLoggedIn }) {
  const [routes, setRoutes] = useState(null)
  const [error, setError] = useState(null)
  const [selectedIdx, setSelectedIdx] = useState(null)
  const [downloading, setDownloading] = useState(false)
  const [downloadError, setDownloadError] = useState(null)

  useEffect(() => {
    // spec.txt 19章: 招待ユーザー限定機能。isLoggedInがfalseでこのダイアログが
    // 開かれることは通常無い（呼び出し元のボタンが無効化されているため）が、
    // 念のための防御的分岐。
    if (!isLoggedIn) return undefined
    let cancelled = false
    listRoutes().then((result) => {
      if (cancelled) return
      if (!result.ok) {
        setError(result.error)
      } else {
        setRoutes(result.routes)
      }
    })
    return () => {
      cancelled = true
    }
  }, [isLoggedIn])

  async function handleLoad() {
    if (selectedIdx === null) return
    setDownloading(true)
    setDownloadError(null)
    const selected = routes[selectedIdx]
    const result = await downloadGpx(selected.file_key)
    setDownloading(false)
    if (!result.ok) {
      setDownloadError(result.error)
      return
    }
    onLoaded(result.content, selected.file_key)
  }

  const selected = selectedIdx !== null && routes ? routes[selectedIdx] : null

  return (
    <div className="modal-overlay">
      <div className="modal-box network-picker">
        <h3>☁️ ネットワークから読み込む</h3>

        {!isLoggedIn && <p className="error">招待ユーザー限定の機能です。ログインしてください。</p>}
        {isLoggedIn && error && <p className="error">取得に失敗しました: {error}</p>}
        {isLoggedIn && !error && routes === null && <p>ルート一覧を取得中…</p>}
        {isLoggedIn && !error && routes && routes.length === 0 && <p>ネットワーク上にルートがありません。</p>}

        {!error && routes && routes.length > 0 && (
          <ul className="network-route-list">
            {routes.map((r, i) => (
              <li key={r.file_key}>
                <label>
                  <input type="radio" name="net-route" checked={selectedIdx === i} onChange={() => setSelectedIdx(i)} />
                  {r.display_name}　{fmtDist(r)}　獲得標高 {fmtGain(r)}
                </label>
              </li>
            ))}
          </ul>
        )}

        {selected ? (
          <p>
            <strong>{selected.display_name}</strong>　距離 {fmtDist(selected)}　獲得標高 {fmtGain(selected)}
          </p>
        ) : (
          routes && routes.length > 0 && <p>ルートを選択してください</p>
        )}

        {downloadError && <p className="error">ダウンロード失敗: {downloadError}</p>}

        <div className="save-dialog-buttons">
          <button type="button" className="btn-secondary" onClick={onCancel}>
            キャンセル
          </button>
          <button type="button" className="btn-primary" disabled={selected === null || downloading} onClick={handleLoad}>
            {downloading ? 'ダウンロード中…' : '読み込む →'}
          </button>
        </div>
      </div>
    </div>
  )
}
