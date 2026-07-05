import { haversine } from './geo.js'

/** 画面上のピクセル距離を、指定緯度・ズームでの実距離（m）に換算する。spec.txt 7-2章。 */
export function pxToMeters(px, zoom, lat) {
  return (px * 156543.03392 * Math.cos((lat * Math.PI) / 180)) / 2 ** zoom
}

/** points（[[lat,lon],...]）の中で最近傍の { idx, dist } を返す。 */
export function nearestPoint(points, lat, lon) {
  let bestIdx = -1
  let bestDist = Infinity
  points.forEach((p, i) => {
    const d = haversine(lat, lon, p[0], p[1])
    if (d < bestDist) {
      bestDist = d
      bestIdx = i
    }
  })
  return bestIdx === -1 ? null : { idx: bestIdx, dist: bestDist }
}

/** thresholdM以内にある全候補を { idx, dist }[] で返す（points順=idx昇順）。 */
export function nearAllWithinThreshold(points, lat, lon, thresholdM) {
  const result = []
  points.forEach((p, i) => {
    const d = haversine(lat, lon, p[0], p[1])
    if (d <= thresholdM) result.push({ idx: i, dist: d })
  })
  return result
}

/** cumDists（累積距離, m）の中でtargetMに最も近いインデックスを返す。spec.txt 7-3章。 */
export function nearestIndexAtDistance(cumDists, targetM) {
  let best = 0
  let bestDiff = Infinity
  for (let i = 0; i < cumDists.length; i++) {
    const diff = Math.abs(cumDists[i] - targetM)
    if (diff < bestDiff) {
      bestDiff = diff
      best = i
    }
  }
  return best
}

/**
 * 候補をルート距離gapM以内でクラスタリングし、各クラスタの代表（最近傍）を返す。
 * spec.txt 7-2章。candidatesはidx昇順であること（cumDistsもidxに対応する累積距離）。
 */
export function clusterCandidates(candidates, cumDists, gapM = 200) {
  if (!candidates.length) return []
  const clusters = []
  let cur = [candidates[0]]
  for (let i = 1; i < candidates.length; i++) {
    const gap = Math.abs(cumDists[candidates[i].idx] - cumDists[candidates[i - 1].idx])
    if (gap < gapM) {
      cur.push(candidates[i])
    } else {
      clusters.push(cur)
      cur = [candidates[i]]
    }
  }
  clusters.push(cur)
  return clusters.map((c) => c.reduce((a, b) => (a.dist < b.dist ? a : b)))
}
