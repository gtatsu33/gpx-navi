import { haversine } from './geo.js'
import { defaultSleep, fetchWithTimeout } from './http.js'

const OVERPASS_URLS = [
  'https://lz4.overpass-api.de/api/interpreter',
  'https://z.overpass-api.de/api/interpreter',
  'https://overpass.kumi.systems/api/interpreter',
]

const INTERSECTION_HIGHWAY_TAGS = '"^(traffic_signals|crossing|give_way|stop|mini_roundabout|motorway_junction)$"'
const POI_TAGS = ['tourism', 'amenity', 'leisure', 'historic', 'natural', 'shop']

export function buildIntersectionQuery(turns, radius, timeoutSec) {
  const unionParts = turns
    .map((t) => `node(around:${radius},${t.lat},${t.lon})[name][highway~${INTERSECTION_HIGHWAY_TAGS}];`)
    .join('')
  return `[out:json][timeout:${timeoutSec}];(${unionParts});out body;`
}

export function buildPoiQuery(lat, lon, radius, timeoutSec) {
  const unionParts = POI_TAGS.map((tag) => `node(around:${radius},${lat},${lon})[name]["${tag}"];`).join('')
  return `[out:json][timeout:${timeoutSec}];(${unionParts});out body;`
}

async function postOverpassQuery(url, query, { fetchImpl = fetch, timeoutMs = 30000 } = {}) {
  const res = await fetchWithTimeout(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded', Accept: 'application/json' },
    body: `data=${encodeURIComponent(query)}`,
    fetchImpl,
    timeoutMs,
  })
  if (res.status === 429) {
    const err = new Error('Overpass rate limited')
    err.status = 429
    throw err
  }
  if (!res.ok) {
    throw new Error(`Overpass request failed: ${res.status}`)
  }
  const data = await res.json()
  return data.elements ?? []
}

/**
 * 複数ミラーへのフェイルオーバー付きクエリ実行。
 * implement.txt 18-5章: 429受信時は単純な次ミラー切替だけでなく、
 * 少し待ってから同一ミラーへ再試行してから次に進む。
 */
export async function queryOverpassWithFailover(
  query,
  { urls = OVERPASS_URLS, fetchImpl = fetch, timeoutMs = 30000, sleep = defaultSleep, retryDelayMs = 2000 } = {}
) {
  for (const url of urls) {
    try {
      return await postOverpassQuery(url, query, { fetchImpl, timeoutMs })
    } catch (e) {
      if (e.status === 429) {
        try {
          await sleep(retryDelayMs)
          return await postOverpassQuery(url, query, { fetchImpl, timeoutMs })
        } catch {
          // 待機後の再試行も失敗 → 次のミラーへ
        }
      }
      // 通常のエラーは次のミラーへ
    }
  }
  return null
}

/** Overpassの応答要素から、各ターン候補の最近傍name（半径内）を求める。純粋関数。 */
export function nearestNameMatch(turns, elements, radius) {
  const result = {}
  for (const t of turns) {
    let nearestName = null
    let nearestDist = Infinity
    for (const node of elements) {
      const d = haversine(t.lat, t.lon, node.lat, node.lon)
      if (d < nearestDist) {
        nearestDist = d
        nearestName = node.tags?.name ?? null
      }
    }
    if (nearestName && nearestDist <= radius) {
      result[t.index] = nearestName
    }
  }
  return result
}

/**
 * Overpass APIでターンポイント付近の交差点名を取得する。spec.txt 12-1章。
 * 戻り値: { trkpt_index: name } の辞書。
 */
export async function fetchIntersectionNames(
  turns,
  { radius = 20, httpTimeout = 30, maxAttempts = null, fetchImpl = fetch, sleep = defaultSleep } = {}
) {
  if (!turns.length) return {}
  const urls = maxAttempts === null ? OVERPASS_URLS : OVERPASS_URLS.slice(0, maxAttempts)
  const query = buildIntersectionQuery(turns, radius, Math.max(1, httpTimeout - 2))
  const elements = await queryOverpassWithFailover(query, {
    urls,
    fetchImpl,
    timeoutMs: httpTimeout * 1000,
    sleep,
  })
  if (elements === null) return {}
  return nearestNameMatch(turns, elements, radius)
}

/** クリック位置付近のPOI名を取得する（交差点名が見つからなかった場合のフォールバック）。spec.txt 12-2章。 */
export async function fetchSpotName(
  lat,
  lon,
  { radius = 20, httpTimeout = 15, maxAttempts = null, fetchImpl = fetch, sleep = defaultSleep } = {}
) {
  const urls = maxAttempts === null ? OVERPASS_URLS : OVERPASS_URLS.slice(0, maxAttempts)
  const query = buildPoiQuery(lat, lon, radius, Math.max(1, httpTimeout - 2))
  const elements = await queryOverpassWithFailover(query, {
    urls,
    fetchImpl,
    timeoutMs: httpTimeout * 1000,
    sleep,
  })
  if (!elements || !elements.length) return null
  let nearest = null
  let nearestDist = Infinity
  for (const node of elements) {
    const d = haversine(lat, lon, node.lat, node.lon)
    if (d < nearestDist) {
      nearestDist = d
      nearest = node
    }
  }
  return nearest?.tags?.name ?? null
}
