export function calculateBearing(lat1, lon1, lat2, lon2) {
  const toRad = (d) => (d * Math.PI) / 180
  const [rlat1, rlon1, rlat2, rlon2] = [lat1, lon1, lat2, lon2].map(toRad)
  const dlon = rlon2 - rlon1
  const x = Math.sin(dlon) * Math.cos(rlat2)
  const y =
    Math.cos(rlat1) * Math.sin(rlat2) -
    Math.sin(rlat1) * Math.cos(rlat2) * Math.cos(dlon)
  return ((Math.atan2(x, y) * 180) / Math.PI + 360) % 360
}

export function angleDiff(a, b) {
  return ((b - a + 180) % 360 + 360) % 360 - 180
}

/** start以降のpoints（[[lat,lon],...]）の中で最近傍のインデックスを返す。 */
export function nearestPointIndexFrom(points, lat, lon, start = 0) {
  const s = start >= points.length ? 0 : start
  let bestIdx = s
  let bestDist = Infinity
  for (let i = s; i < points.length; i++) {
    const d = haversine(lat, lon, points[i][0], points[i][1])
    if (d < bestDist) {
      bestDist = d
      bestIdx = i
    }
  }
  return bestIdx
}

export function haversine(lat1, lon1, lat2, lon2) {
  const R = 6371000
  const toRad = (d) => (d * Math.PI) / 180
  const [rlat1, rlon1, rlat2, rlon2] = [lat1, lon1, lat2, lon2].map(toRad)
  const a =
    Math.sin((rlat2 - rlat1) / 2) ** 2 +
    Math.cos(rlat1) * Math.cos(rlat2) * Math.sin((rlon2 - rlon1) / 2) ** 2
  return R * 2 * Math.asin(Math.sqrt(Math.max(0, a)))
}
