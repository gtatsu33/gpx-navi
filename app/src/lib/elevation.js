import { fetchWithTimeout } from './http.js'
import { haversine } from './geo.js'

const GSI_URL = 'https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php'
const BACKOFF_DELAYS_MS = [0, 1000, 2000] // implement.txt 2-5章: 1回目即時・2回目1秒後・3回目2秒後

function median(values) {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

export function cumulativeDistances(points) {
  const cum = [0]
  for (let i = 1; i < points.length; i++) {
    cum.push(cum[i - 1] + haversine(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1]))
  }
  return cum
}

export function elevationGrades(points, elevations, cumDists = null) {
  const cum = cumDists ?? cumulativeDistances(points)
  const grades = []
  for (let i = 0; i < points.length - 1; i++) {
    if (elevations[i] === null || elevations[i + 1] === null) {
      grades.push(null)
      continue
    }
    const dist = cum[i + 1] - cum[i]
    if (dist <= 0) {
      grades.push(null)
      continue
    }
    grades.push(((elevations[i + 1] - elevations[i]) / dist) * 100)
  }
  return grades
}

function localMedianElevation(i, cumDists, elevations, windowM) {
  const lo = cumDists[i] - windowM
  const hi = cumDists[i] + windowM
  const vals = []
  for (let j = 0; j < elevations.length; j++) {
    if (elevations[j] !== null && cumDists[j] >= lo && cumDists[j] <= hi) {
      vals.push(elevations[j])
    }
  }
  return vals.length ? median(vals) : null
}

function clusterSegments(segIndexes, cumDists, clusterGapM) {
  if (!segIndexes.length) return []
  const clusters = []
  let cur = { startSeg: segIndexes[0], endSeg: segIndexes[0] }
  for (let k = 1; k < segIndexes.length; k++) {
    const segIdx = segIndexes[k]
    const gapM = cumDists[segIdx] - cumDists[cur.endSeg + 1]
    if (gapM <= clusterGapM) {
      cur.endSeg = segIdx
    } else {
      clusters.push(cur)
      cur = { startSeg: segIdx, endSeg: segIdx }
    }
  }
  clusters.push(cur)
  return clusters
}

/**
 * 標高スパイク除去。spec.txt 13-3章のアルゴリズム。
 * points: [[lat, lon], ...]、elevations: (number|null)[]
 * 戻り値: { cleaned, stats }
 */
export function cleanElevationSpikes(points, elevations, badGradeThreshold = 15.0, clusterGapM = 250.0) {
  const n = points.length
  if (n < 4 || !elevations || elevations.length !== n) {
    return { cleaned: elevations, stats: { clusters: 0, points: 0, maxGradeBefore: 0.0, maxGradeAfter: 0.0 } }
  }

  const HARD_SPIKE_THRESHOLD = 35.0
  const NEAR_BAD_THRESHOLD = 15.0
  const MIN_ELEVATION_JUMP_M = 2.0
  const NEAR_ELEVATION_JUMP_M = 3.0
  const SHORT_SEG_M = 10.0
  const MERGE_GAP_M = 50.0
  const MAX_ANCHOR_SEARCH_M = 600.0
  const ANCHOR_GRADE_LIMIT = 12.0
  const BOUNDARY_GRADE_LIMIT = 13.0
  const MEDIAN_WINDOW_M = 150.0
  const ANCHOR_MEDIAN_DEV_M = 5.0
  const MAX_ANCHOR_GRADE = 15.0
  const MIN_CORRECTION_GRADE_PCT = 1.0

  const cleaned = [...elevations]
  const cumDists = cumulativeDistances(points)
  let grades = elevationGrades(points, cleaned, cumDists)
  const definedAbs = grades.filter((g) => g !== null).map(Math.abs)
  const maxGradeBefore = definedAbs.length ? Math.max(...definedAbs) : 0.0

  const badSegments = []
  grades.forEach((grade, i) => {
    if (grade === null || cleaned[i] === null || cleaned[i + 1] === null) return
    const dz = cleaned[i + 1] - cleaned[i]
    const shortSeg = cumDists[i + 1] - cumDists[i] < SHORT_SEG_M
    if (
      (Math.abs(grade) >= badGradeThreshold && (shortSeg || Math.abs(dz) >= MIN_ELEVATION_JUMP_M)) ||
      (Math.abs(grade) >= HARD_SPIKE_THRESHOLD && (shortSeg || Math.abs(dz) >= NEAR_ELEVATION_JUMP_M))
    ) {
      badSegments.push(i)
    }
  })

  if (!badSegments.length) {
    return { cleaned, stats: { clusters: 0, points: 0, maxGradeBefore, maxGradeAfter: maxGradeBefore } }
  }

  const nearSegments = new Set(badSegments)
  for (const badIdx of badSegments) {
    const center = (cumDists[badIdx] + cumDists[badIdx + 1]) / 2
    grades.forEach((grade, i) => {
      if (grade === null || cleaned[i] === null || cleaned[i + 1] === null) return
      const segCenter = (cumDists[i] + cumDists[i + 1]) / 2
      const dz = cleaned[i + 1] - cleaned[i]
      const shortSeg = cumDists[i + 1] - cumDists[i] < SHORT_SEG_M
      if (
        Math.abs(segCenter - center) <= clusterGapM &&
        Math.abs(grade) >= NEAR_BAD_THRESHOLD &&
        (shortSeg || Math.abs(dz) >= NEAR_ELEVATION_JUMP_M)
      ) {
        nearSegments.add(i)
      }
    })
  }

  const clusters = clusterSegments([...nearSegments].sort((a, b) => a - b), cumDists, clusterGapM)

  const isAnchorCandidate = (i) => {
    if (i <= 0 || i >= n - 1 || cleaned[i] === null) return false
    const prevG = grades[i - 1]
    const nextG = grades[i]
    if (prevG === null || nextG === null) return false
    if (Math.abs(prevG) > ANCHOR_GRADE_LIMIT || Math.abs(nextG) > ANCHOR_GRADE_LIMIT) return false
    const localMed = localMedianElevation(i, cumDists, cleaned, MEDIAN_WINDOW_M)
    return localMed !== null && Math.abs(cleaned[i] - localMed) <= ANCHOR_MEDIAN_DEV_M
  }

  const findAnchor = (startI, direction) => {
    const startDist = cumDists[startI]
    let i = startI
    let stableRun = []
    while (i > 0 && i < n - 1 && Math.abs(cumDists[i] - startDist) <= MAX_ANCHOR_SEARCH_M) {
      if (isAnchorCandidate(i)) {
        stableRun.push(i)
        if (stableRun.length >= 2) return stableRun[0]
      } else {
        stableRun = []
      }
      i += direction
    }
    return null
  }

  const isLeftBoundaryAnchor = (i) =>
    i > 0 && i < n - 1 && cleaned[i] !== null && grades[i - 1] !== null && Math.abs(grades[i - 1]) <= BOUNDARY_GRADE_LIMIT

  const isRightBoundaryAnchor = (i) =>
    i > 0 && i < n - 1 && cleaned[i] !== null && grades[i] !== null && Math.abs(grades[i]) <= BOUNDARY_GRADE_LIMIT

  const repairRanges = []
  for (const cluster of clusters) {
    const startPt = cluster.startSeg
    const endPt = cluster.endSeg + 1
    const leftAnchor =
      isLeftBoundaryAnchor(startPt) && isAnchorCandidate(startPt) ? startPt : findAnchor(startPt - 1, -1)
    const rightAnchor =
      isRightBoundaryAnchor(endPt) && isAnchorCandidate(endPt) ? endPt : findAnchor(endPt + 1, 1)
    if (leftAnchor === null || rightAnchor === null || leftAnchor >= rightAnchor) continue
    const distM = cumDists[rightAnchor] - cumDists[leftAnchor]
    if (distM <= 0 || cleaned[leftAnchor] === null || cleaned[rightAnchor] === null) continue
    const netGrade = ((cleaned[rightAnchor] - cleaned[leftAnchor]) / distM) * 100
    if (Math.abs(netGrade) > MAX_ANCHOR_GRADE) continue
    repairRanges.push({ left: leftAnchor, right: rightAnchor, badStart: startPt, badEnd: endPt })
  }

  if (!repairRanges.length) {
    return { cleaned, stats: { clusters: 0, points: 0, maxGradeBefore, maxGradeAfter: maxGradeBefore } }
  }

  repairRanges.sort((a, b) => a.left - b.left)
  const merged = [repairRanges[0]]
  for (let k = 1; k < repairRanges.length; k++) {
    const r = repairRanges[k]
    const prev = merged[merged.length - 1]
    const gapM = cumDists[r.left] - cumDists[prev.right]
    if (r.left <= prev.right || gapM <= MERGE_GAP_M) {
      prev.right = Math.max(prev.right, r.right)
      prev.badStart = Math.min(prev.badStart, r.badStart)
      prev.badEnd = Math.max(prev.badEnd, r.badEnd)
    } else {
      merged.push(r)
    }
  }

  const correctedPoints = new Set()
  for (const r of merged) {
    const { left, right } = r
    if (right - left < 2) continue
    const distM = cumDists[right] - cumDists[left]
    if (distM <= 0 || cleaned[left] === null || cleaned[right] === null) continue
    const netGrade = ((cleaned[right] - cleaned[left]) / distM) * 100
    if (Math.abs(netGrade) > MAX_ANCHOR_GRADE) continue
    for (let i = left + 1; i < right; i++) {
      if (cleaned[i] === null) continue
      const ratio = (cumDists[i] - cumDists[left]) / distM
      const newEle = cleaned[left] + (cleaned[right] - cleaned[left]) * ratio
      const minAdj = Math.min(cumDists[i] - cumDists[i - 1], cumDists[i + 1] - cumDists[i])
      if (Math.abs(cleaned[i] - newEle) >= (MIN_CORRECTION_GRADE_PCT / 100) * minAdj) {
        cleaned[i] = Math.round(newEle * 10) / 10
        correctedPoints.add(i)
      }
    }
  }

  const gradesAfter = elevationGrades(points, cleaned, cumDists)
  const definedAbsAfter = gradesAfter.filter((g) => g !== null).map(Math.abs)
  const maxGradeAfter = definedAbsAfter.length ? Math.max(...definedAbsAfter) : 0.0

  return {
    cleaned,
    stats: {
      clusters: merged.length,
      points: correctedPoints.size,
      maxGradeBefore,
      maxGradeAfter,
    },
  }
}

function defaultSleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * 国土地理院標高APIを1点だけ呼び出す。spec.txt 17-3章。
 * 取得不可（-9999、"-----"、レスポンス不正）はnullを返す。
 */
export async function fetchElevationRaw(lat, lon, { fetchImpl = fetch, timeoutMs = 10000 } = {}) {
  const url = `${GSI_URL}?lat=${lat}&lon=${lon}&outtype=JSON`
  const res = await fetchWithTimeout(url, { fetchImpl, timeoutMs })
  if (!res.ok) {
    throw new Error(`GSI request failed: ${res.status}`)
  }
  const data = await res.json()
  const ev = data.elevation
  if (ev === null || ev === undefined || ev === -9999 || ev === '-----') return null
  return Number(ev)
}

/**
 * 指数バックオフ付きで1点を再試行する。implement.txt 2-5章。
 * 最大3回失敗したら「取得不可」（null）として確定する（それ以上は自動再試行しない）。
 */
export async function fetchElevationWithRetry(lat, lon, { fetchImpl = fetch, timeoutMs = 10000, sleep = defaultSleep } = {}) {
  for (let attempt = 0; attempt < BACKOFF_DELAYS_MS.length; attempt++) {
    if (BACKOFF_DELAYS_MS[attempt] > 0) {
      await sleep(BACKOFF_DELAYS_MS[attempt])
    }
    try {
      return await fetchElevationRaw(lat, lon, { fetchImpl, timeoutMs })
    } catch {
      // 次の試行へ（最終試行の失敗はループを抜けてnullを返す）
    }
  }
  return null
}

/**
 * 短時間の連続失敗を検知して自動取得を一時停止するサーキットブレーカー。
 * implement.txt 2-5章:「直近10リクエスト中5回以上」失敗で30秒停止。
 */
export class ElevationCircuitBreaker {
  constructor({ windowSize = 10, failureThreshold = 5, cooldownMs = 30000, now = () => Date.now() } = {}) {
    this.windowSize = windowSize
    this.failureThreshold = failureThreshold
    this.cooldownMs = cooldownMs
    this.now = now
    this.results = []
    this.openedAt = null
  }

  isOpen() {
    if (this.openedAt === null) return false
    if (this.now() - this.openedAt >= this.cooldownMs) {
      this.openedAt = null
      this.results = []
      return false
    }
    return true
  }

  recordResult(success) {
    this.results.push(success)
    if (this.results.length > this.windowSize) this.results.shift()
    const failures = this.results.filter((r) => !r).length
    if (this.openedAt === null && failures >= this.failureThreshold) {
      this.openedAt = this.now()
    }
  }
}

/** 勾配リストから上り最大・下り最大（符号付き）を返す。データなしはnull。 */
export function computeGradeStats(points, elevations) {
  const grades = elevationGrades(points, elevations)
  const valid = grades.filter((g) => g !== null)
  if (!valid.length) return null
  const positives = valid.filter((g) => g > 0)
  const negatives = valid.filter((g) => g < 0)
  return {
    max: positives.length ? Math.max(...positives) : 0.0,
    min: negatives.length ? Math.min(...negatives) : 0.0,
  }
}

/**
 * 指定したインデックス群についてele_fixをまとめて取得する。spec.txt 16-2章。
 * 保存画面起動時の標高整合性チェック（org完全ならケース1で全点、
 * 編集済みならケース2でele_fix未確定点のみ）から呼び出す想定。
 * 戻り値: [{ trkptIndex, value }]（SET_ELE_FIX_BATCHにそのまま渡せる形）
 */
export async function fetchElevationsForIndices(
  points,
  indices,
  { concurrency = 5, onProgress = () => {}, fetchImpl = fetch, sleep = undefined } = {}
) {
  const assignments = []
  let done = 0
  const total = indices.length
  const queue = [...indices]

  async function worker() {
    while (queue.length) {
      const idx = queue.shift()
      const [lat, lon] = points[idx]
      const value = await fetchElevationWithRetry(lat, lon, sleep ? { fetchImpl, sleep } : { fetchImpl })
      assignments.push({ trkptIndex: idx, value })
      done += 1
      onProgress({ done, total })
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, total || 1) }, worker))
  return assignments
}
