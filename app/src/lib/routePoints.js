/** spec.txt 5-1章のRoutePoint構造を生成する。 */
export function makeRoutePoint(
  lat,
  lon,
  { eleOrg = null, eleFix = null, isAcpt = false, wpt = null, changed = true } = {}
) {
  return { lat, lon, eleOrg, eleFix, isAcpt, wpt, changed }
}

export function deepCopyRoutePoints(rp) {
  return rp.map((p) => ({ ...p, wpt: p.wpt ? { ...p.wpt } : null }))
}

/** idxより前にあるis_acptまたはwptのうち最大のindexを返す。spec.txt 5-2/8-2章。 */
export function prevBoundary(idx, rp) {
  let result = 0
  for (let i = 0; i < idx; i++) {
    if (rp[i].isAcpt || rp[i].wpt !== null) result = i
  }
  return result
}

/** idxより後にあるis_acptまたはwptのうち最小のindexを返す。 */
export function nextBoundary(idx, rp) {
  for (let i = idx + 1; i < rp.length; i++) {
    if (rp[i].isAcpt || rp[i].wpt !== null) return i
  }
  return rp.length - 1
}

/** 連続するインデックスをまとめて [start, end] タプルのリストにする。spec.txt 15章。 */
export function contiguousRanges(indices) {
  if (!indices.length) return []
  const ranges = []
  let start = indices[0]
  let end = indices[0]
  for (let k = 1; k < indices.length; k++) {
    const i = indices[k]
    if (i === end + 1) {
      end = i
    } else {
      ranges.push([start, end])
      start = i
      end = i
    }
  }
  ranges.push([start, end])
  return ranges
}
