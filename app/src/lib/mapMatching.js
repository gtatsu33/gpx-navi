import { fetchWithTimeout } from './http.js'

const VALHALLA_URL = 'https://valhalla1.openstreetmap.de/trace_attributes'

/**
 * Valhalla trace_attributes で1チャンクをスナップする（bicycle固定）。
 * spec.txt 17-2章。失敗時（タイムアウト含む）は例外を投げる
 * （呼び出し側＝チャンク処理ロジックがフェーズ8で継続/中断を判断する）。
 */
export async function matchChunk(chunk, { searchRadius = 50, fetchImpl = fetch, timeoutMs = 30000 } = {}) {
  const res = await fetchWithTimeout(VALHALLA_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      shape: chunk.map(([lat, lon]) => ({ lat, lon })),
      costing: 'bicycle',
      shape_match: 'map_snap',
      search_radius: searchRadius,
      filters: { attributes: ['matched.point', 'matched.type'], action: 'include' },
    }),
    fetchImpl,
    timeoutMs,
  })
  if (!res.ok) {
    throw new Error(`Valhalla request failed: ${res.status}`)
  }
  return res.json()
}

/**
 * matchChunkのレスポンスから、matched/interpolatedの点だけを座標に反映する。
 * それ以外（unmatched等）は元座標を保持する。spec.txt 10-2章。
 */
export function applyMatchedPoints(originalChunk, matchResponse) {
  const matchedPoints = matchResponse?.matched_points ?? []
  return originalChunk.map((pt, i) => {
    const mp = matchedPoints[i]
    if (mp && (mp.type === 'matched' || mp.type === 'interpolated')) {
      return [mp.lat, mp.lon]
    }
    return pt
  })
}

const CHUNK_SIZE = 50

/**
 * マップマッチングのチャンク処理ロジック（React状態管理から独立した純粋な非同期処理）。
 * spec.txt 10-2章。50点チャンクずつ順次実行し、進捗をonProgressで通知する。
 * shouldCancel()がtrueを返した時点でキャンセル扱いにする。
 *
 * 戻り値: { matchedPoints, nSnapped, status: '完了'|'エラー'|'キャンセル', error }
 */
export async function matchRoute(points, { onProgress = () => {}, shouldCancel = () => false, matchChunkImpl = matchChunk } = {}) {
  const totalChunks = Math.max(1, Math.ceil(points.length / CHUNK_SIZE))
  const matched = [...points]
  let nSnapped = 0
  const errors = []

  for (let c = 0; c < totalChunks; c++) {
    if (shouldCancel()) {
      const error = 'キャンセルされました' + (errors.length ? '; ' + errors.join('; ') : '')
      onProgress({ chunkIdx: c, totalChunks, nSnapped, status: 'キャンセル', error })
      return { matchedPoints: matched, nSnapped, status: 'キャンセル', error }
    }

    const start = c * CHUNK_SIZE
    const end = Math.min(start + CHUNK_SIZE, points.length)
    const chunk = points.slice(start, end)

    try {
      const response = await matchChunkImpl(chunk)
      const applied = applyMatchedPoints(chunk, response)
      applied.forEach((pt, i) => {
        if (pt !== chunk[i]) {
          matched[start + i] = pt
          nSnapped += 1
        }
      })
    } catch (err) {
      if (c === 0) {
        const error = '1チャンク目タイムアウトにより自動キャンセル'
        onProgress({ chunkIdx: 0, totalChunks, nSnapped: 0, status: 'キャンセル', error })
        return { matchedPoints: points, nSnapped: 0, status: 'キャンセル', error }
      }
      errors.push(`chunk ${c}: ${err.message}`)
    }

    onProgress({ chunkIdx: c + 1, totalChunks, nSnapped, status: 'running', error: null })
  }

  const finalStatus = nSnapped > 0 ? '完了' : 'エラー'
  const error = errors.length ? errors.join('; ') : null
  onProgress({ chunkIdx: totalChunks, totalChunks, nSnapped, status: finalStatus, error })
  return { matchedPoints: matched, nSnapped, status: finalStatus, error }
}
