import { fetchWithTimeout } from './http.js'

const OSRM_BASE = 'https://router.project-osrm.org/route/v1/bike'

/**
 * 複数点を経由する道路沿いtrkpt列を返す。OSRM公開APIを使用。
 * 失敗時は points をそのまま返す（直線フォールバック）。spec.txt 17-1章。
 * points: [[lat, lon], ...]
 */
export async function calcRouteSegment(points, { fetchImpl = fetch, timeoutMs = 30000 } = {}) {
  const coordsStr = points.map(([lat, lon]) => `${lon},${lat}`).join(';')
  const url = `${OSRM_BASE}/${coordsStr}?overview=full&geometries=geojson`
  try {
    const res = await fetchWithTimeout(url, { fetchImpl, timeoutMs })
    if (!res.ok) return points
    const data = await res.json()
    if (data.code !== 'Ok') return points
    return data.routes[0].geometry.coordinates.map(([lon, lat]) => [lat, lon])
  } catch {
    return points
  }
}
