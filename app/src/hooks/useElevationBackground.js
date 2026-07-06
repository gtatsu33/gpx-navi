import { useEffect, useRef, useState } from 'react'
import { ElevationCircuitBreaker, fetchElevationWithRetry } from '../lib/elevation.js'

const DEBOUNCE_MS = 800
const CONCURRENCY = 5

function isInJapan(lat, lon) {
  return lat >= 24.0 && lat <= 46.0 && lon >= 122.0 && lon <= 154.0
}

function coordKey(lat, lon) {
  return `${lat.toFixed(5)},${lon.toFixed(5)}`
}

/** spec.txt 13-2章: ele_orgが既にある点は取得不要（無駄なAPI呼び出しを避ける）。 */
function hasOwnElevation(p) {
  return p.eleOrg !== null && p.eleOrg !== undefined
}

/**
 * 国土地理院補正標高のリアルタイム背景取得。spec.txt 13-2章／implement.txt 2章。
 * ルート編集（新規点追加・座標変更）に追従し、明示的な開始操作なしで自動的に
 * 背景取得する。取得結果はSET_ELE_FIX_BATCHで随時反映し、全点の取得試行が
 * 完了した時点でFINALIZE_ELE_FIXを1回だけdispatchしてスパイク除去・勾配統計を行う。
 */
export function useElevationBackground(routePoints, dispatch) {
  const [status, setStatus] = useState({ state: 'idle', done: 0, total: 0, unavailable: 0 })

  const routePointsRef = useRef(routePoints)
  routePointsRef.current = routePoints
  const cacheRef = useRef(new Map()) // coordKey -> value|null（null=このセッションで恒久的に取得不可）
  const breakerRef = useRef(new ElevationCircuitBreaker())
  const debounceTimerRef = useRef(null)
  const runningRef = useRef(false)
  const rerunRequestedRef = useRef(false)

  function maybeFinalize() {
    const rp = routePointsRef.current
    if (!rp.length) return
    const allAttempted = rp.every(
      (p) => hasOwnElevation(p) || p.eleFix !== null || cacheRef.current.has(coordKey(p.lat, p.lon))
    )
    if (allAttempted) {
      dispatch({ type: 'FINALIZE_ELE_FIX' })
      setStatus((s) => ({ ...s, state: 'done' }))
    }
  }

  async function runBatch() {
    if (runningRef.current) {
      rerunRequestedRef.current = true
      return
    }
    const rp = routePointsRef.current
    if (!rp.length) return
    if (!isInJapan(rp[0].lat, rp[0].lon)) {
      setStatus({ state: 'out_of_japan', done: 0, total: 0, unavailable: 0 })
      return
    }
    if (breakerRef.current.isOpen()) {
      setStatus((s) => ({ ...s, state: 'paused' }))
      return
    }

    const targets = []
    const cachedAssignments = []
    rp.forEach((p, i) => {
      if (hasOwnElevation(p)) return
      if (p.eleFix !== null) return
      const key = coordKey(p.lat, p.lon)
      if (cacheRef.current.has(key)) {
        // 別の点として既に取得済みの座標と一致する場合、その値をこの点にも反映する
        // （implement.txt 2-2章の「同一座標の結果を再利用する」を実際に適用する）
        cachedAssignments.push({ trkptIndex: i, value: cacheRef.current.get(key) })
        return
      }
      targets.push(i)
    })

    if (cachedAssignments.length) {
      dispatch({ type: 'SET_ELE_FIX_BATCH', payload: { assignments: cachedAssignments } })
    }

    if (!targets.length) {
      maybeFinalize()
      return
    }

    runningRef.current = true
    let doneCount = 0
    let unavailableCount = 0
    setStatus({ state: 'running', done: 0, total: targets.length, unavailable: 0 })

    const queue = [...targets]
    const results = []

    async function worker() {
      while (queue.length) {
        const idx = queue.shift()
        const p = routePointsRef.current[idx]
        if (!p || hasOwnElevation(p) || p.eleFix !== null) continue
        const value = await fetchElevationWithRetry(p.lat, p.lon)
        breakerRef.current.recordResult(value !== null)
        if (value === null) unavailableCount += 1
        results.push({ index: idx, lat: p.lat, lon: p.lon, value })
        doneCount += 1
        setStatus({ state: 'running', done: doneCount, total: targets.length, unavailable: unavailableCount })
        if (breakerRef.current.isOpen()) {
          queue.length = 0
          break
        }
      }
    }

    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, targets.length) }, worker))

    const rpNow = routePointsRef.current
    const assignments = []
    results.forEach((r) => {
      // ルート編集で対象点がずれた/消えた場合はキャッシュにも書き込まない
      // （書き込んでしまうと、同じ座標の点が二度と再取得されなくなる）
      if (rpNow[r.index] && rpNow[r.index].lat === r.lat && rpNow[r.index].lon === r.lon) {
        cacheRef.current.set(coordKey(r.lat, r.lon), r.value)
        assignments.push({ trkptIndex: r.index, value: r.value })
      }
    })
    if (assignments.length) {
      dispatch({ type: 'SET_ELE_FIX_BATCH', payload: { assignments } })
    }

    runningRef.current = false

    if (breakerRef.current.isOpen()) {
      setStatus({ state: 'circuit_open', done: doneCount, total: targets.length, unavailable: unavailableCount })
      setTimeout(() => {
        if (!breakerRef.current.isOpen()) runBatch()
      }, breakerRef.current.cooldownMs + 100)
      return
    }

    if (rerunRequestedRef.current) {
      rerunRequestedRef.current = false
      runBatch()
    } else {
      maybeFinalize()
    }
  }

  useEffect(() => {
    if (!routePoints.length) return undefined
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    debounceTimerRef.current = setTimeout(() => {
      runBatch()
    }, DEBOUNCE_MS)
    return () => clearTimeout(debounceTimerRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routePoints])

  function retryFailed() {
    for (const [key, value] of cacheRef.current.entries()) {
      if (value === null) cacheRef.current.delete(key)
    }
    breakerRef.current = new ElevationCircuitBreaker()
    runBatch()
  }

  return { status, retryFailed }
}
