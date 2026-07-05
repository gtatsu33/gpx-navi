import { angleDiff, calculateBearing, haversine } from './geo.js'

/**
 * 角度法によるターン検出。points は [[lat, lon], ...] 形式。
 * spec.txt 11章のアルゴリズムに対応。
 */
export function detectTurns(points, { minTurnAngle = 45, minDist = 100, smooth = 1 } = {}) {
  const n = points.length
  const candidates = []
  for (let i = smooth; i < n - smooth; i++) {
    const A = points[i - smooth]
    const X = points[i]
    const B = points[i + smooth]
    const bearingIn = calculateBearing(A[0], A[1], X[0], X[1])
    const bearingOut = calculateBearing(X[0], X[1], B[0], B[1])
    const turn = angleDiff(bearingIn, bearingOut)
    if (Math.abs(turn) >= minTurnAngle) {
      candidates.push({ lat: X[0], lon: X[1], delta: turn, index: i })
    }
  }

  if (!candidates.length) return []

  const sorted = [...candidates].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
  const used = new Set()
  const turns = []
  for (const c of sorted) {
    if (used.has(c.index)) continue
    turns.push(c)
    for (const c2 of candidates) {
      if (haversine(c.lat, c.lon, c2.lat, c2.lon) < minDist) {
        used.add(c2.index)
      }
    }
  }

  turns.sort((a, b) => a.index - b.index)
  return turns
}

/** delta（度）から [ラベル, 矢印, 色] を返す。spec.txt 11章末尾。 */
export function turnLabel(delta) {
  if (delta >= 60) return ['右折', '⇒', '#e74c3c']
  if (delta >= 25) return ['やや右', '↗', '#e67e22']
  if (delta <= -60) return ['左折', '⇐', '#2980b9']
  if (delta <= -25) return ['やや左', '↖', '#8e44ad']
  return ['直進維持', '↑', '#7f8c8d']
}

/** delta＋交差点名から最終的な案内名を組み立てる。spec.txt 8-5章 with_name()相当。 */
export function combineTurnName(delta, intersectionName) {
  return intersectionName ? `${intersectionName}を${turnLabel(delta)[0]}` : turnLabel(delta)[0]
}

/** wpt情報から [矢印, 色] を返す。spec.txt 7-1章（中間wptマーカーの色）。 */
export function wptStyle(wpt) {
  if (wpt && wpt.delta !== null && wpt.delta !== undefined) {
    const [, arrow, color] = turnLabel(wpt.delta)
    return [arrow, color]
  }
  return ['📍', '#27ae60']
}
