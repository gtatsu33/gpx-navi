function perpendicularDistance(point, lineStart, lineEnd) {
  const [x, y] = point
  const [x1, y1] = lineStart
  const [x2, y2] = lineEnd
  if (x1 === x2 && y1 === y2) {
    return Math.hypot(x - x1, y - y1)
  }
  const num = Math.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
  const den = Math.hypot(y2 - y1, x2 - x1)
  return num / den
}

/**
 * Douglas-Peucker間引き。points は [[lat, lon], ...] 形式。
 * 戻り値は points と同じ長さの真偽値マスク（残す点=true）。
 * Python版 rdp パッケージの rdp(..., return_mask=True) と同じアルゴリズム。
 */
export function rdpMask(points, epsilon) {
  const n = points.length
  const mask = new Array(n).fill(false)
  if (n === 0) return mask
  mask[0] = true
  mask[n - 1] = true
  if (n < 3) return mask

  function recurse(start, end) {
    let maxDist = -1
    let index = -1
    for (let i = start + 1; i < end; i++) {
      const d = perpendicularDistance(points[i], points[start], points[end])
      if (d > maxDist) {
        maxDist = d
        index = i
      }
    }
    if (maxDist > epsilon) {
      mask[index] = true
      recurse(start, index)
      recurse(index, end)
    }
  }

  recurse(0, n - 1)
  return mask
}

/** マスクに基づいて間引き後の点列と、元配列における残存インデックスを返す。 */
export function rdpSimplify(points, epsilon) {
  const mask = rdpMask(points, epsilon)
  const keptIndices = []
  const simplified = []
  mask.forEach((keep, i) => {
    if (keep) {
      keptIndices.push(i)
      simplified.push(points[i])
    }
  })
  return { simplified, keptIndices, mask }
}
