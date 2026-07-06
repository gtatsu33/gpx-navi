import { useMemo, useReducer, useRef, useState } from 'react'
import MapView from './components/MapView.jsx'
import ElevationChart from './components/ElevationChart.jsx'
import TurnPointList from './components/TurnPointList.jsx'
import StatusBadge from './components/StatusBadge.jsx'
import MapMatchingDialog from './components/MapMatchingDialog.jsx'
import SaveDialog from './components/SaveDialog.jsx'
import NetworkPickerDialog from './components/NetworkPickerDialog.jsx'
import StartModal from './components/StartModal.jsx'
import DiscardConfirmModal from './components/DiscardConfirmModal.jsx'
import { useElevationBackground } from './hooks/useElevationBackground.js'
import { useMapMatching } from './hooks/useMapMatching.js'
import { useAuth } from './hooks/useAuth.js'
import { parseGpx } from './lib/gpx.js'
import { haversine } from './lib/geo.js'
import { combineTurnName, detectTurns, wptStyle } from './lib/turns.js'
import { contiguousRanges, nextBoundary, prevBoundary } from './lib/routePoints.js'
import { rdpSimplify } from './lib/rdp.js'
import { calcRouteSegment } from './lib/routing.js'
import { fetchIntersectionNames, fetchSpotName } from './lib/overpass.js'
import { cumulativeDistances } from './lib/elevation.js'
import { nearestIndexAtDistance } from './lib/mapInteractions.js'
import { routeReducer, initialRouteState } from './hooks/useRouteReducer.js'
import { APP_VERSION } from './constants.js'
import './App.css'

const DEFAULT_CENTER = { lat: 35.681, lng: 139.767 } // 東京

function App() {
  const [state, dispatch] = useReducer(routeReducer, undefined, initialRouteState)
  const stateRef = useRef(state)
  stateRef.current = state
  const [error, setError] = useState(null)
  const [focusCenter, setFocusCenter] = useState(null)
  const [focusWpt, setFocusWpt] = useState(null)
  const [hoveredKm, setHoveredKm] = useState(null)
  const [rawGpxString, setRawGpxString] = useState(null)
  const [gpxFilename, setGpxFilename] = useState('')
  const [trackName, setTrackName] = useState(null)
  const [eleSourceGsi, setEleSourceGsi] = useState(false)
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [showNetworkDialog, setShowNetworkDialog] = useState(false)
  const [started, setStarted] = useState(false)
  const [showDiscardConfirm, setShowDiscardConfirm] = useState(false)
  const mapViewRef = useRef(null)
  const { status: eleStatus, retryFailed: retryEleFailed } = useElevationBackground(state.routePoints, dispatch)
  const { state: mapMatchState, run: runMapMatching, cancel: cancelMapMatching } = useMapMatching()
  const { user, sendMagicLink, signOut } = useAuth()
  const isLoggedIn = Boolean(user)

  // spec.txt 6章・11章・12章: wptを含まないGPXの読込時に、全区間のターン自動検出
  // ＋交差点名取得を行う（非同期のOverpass呼び出しを伴うためreducerの外で行う）
  async function detectAndNameTurns(points2d) {
    const raw = detectTurns(points2d, { minTurnAngle: 45, minDist: 100, smooth: 1 })
    const cands = raw.map((t) => ({ lat: points2d[t.index][0], lon: points2d[t.index][1], index: t.index, delta: t.delta }))
    const inames = await fetchIntersectionNames(cands)
    return cands.map((c) => ({
      trkptIndex: c.index,
      delta: c.delta,
      name: combineTurnName(c.delta, inames[c.index] ?? null),
    }))
  }

  async function loadGpxText(text, filename, isActualRide) {
    const { trkpts, waypoints, acptIndices, eleSourceGsi, trackName: parsedTrackName } = parseGpx(text)
    if (trkpts.length < 6) {
      setError('トラックポイントが少なすぎます。')
      return
    }
    setError(null)
    setRawGpxString(text)
    setGpxFilename(filename.replace(/\.(gpx|xml)$/i, '').replace(/_gne$/i, ''))
    setTrackName(parsedTrackName || null)
    setEleSourceGsi(eleSourceGsi)

    const hasWpts = waypoints.length > 0
    if (isActualRide && !hasWpts) {
      // spec.txt 6章・10章: 実走行データはRDP間引き→マップマッチングを行う
      // （wpt確定済みGPXの場合はスキップしてルートデータと同じ経路を通る）
      const points = trkpts.map((t) => [t.lat, t.lon])
      const { simplified, keptIndices } = rdpSimplify(points, 0.00005)
      const { matchedPoints } = await runMapMatching(simplified)
      const origElevations = keptIndices.map((i) => trkpts[i].ele)
      const turnAssignments = await detectAndNameTurns(matchedPoints)
      dispatch({
        type: 'LOAD_MATCHED_ROUTE',
        payload: { matchedPoints, origElevations, waypoints, acptIndices, turnAssignments, eleSourceGsi },
      })
    } else if (!hasWpts) {
      const points = trkpts.map((t) => [t.lat, t.lon])
      const turnAssignments = await detectAndNameTurns(points)
      dispatch({ type: 'LOAD_PARSED_GPX', payload: { trkpts, waypoints, acptIndices, turnAssignments, eleSourceGsi } })
    } else {
      dispatch({ type: 'LOAD_PARSED_GPX', payload: { trkpts, waypoints, acptIndices, eleSourceGsi } })
    }
    setStarted(true)
  }

  const handleFileChange = async (e, isActualRide) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (!/\.(gpx|xml)$/i.test(file.name)) {
      setError('GPXファイル（.gpx または .xml）を選択してください。')
      return
    }
    try {
      const text = await file.text()
      await loadGpxText(text, file.name, isActualRide)
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleNetworkLoaded(text, fileKey) {
    setShowNetworkDialog(false)
    try {
      await loadGpxText(text, fileKey, false)
    } catch (err) {
      setError(err.message)
    }
  }

  function handleNewRoute() {
    setTrackName(null)
    setEleSourceGsi(false)
    setStarted(true)
  }

  function handleDiscardConfirmed() {
    dispatch({ type: 'RESET' })
    setRawGpxString(null)
    setGpxFilename('')
    setTrackName(null)
    setEleSourceGsi(false)
    setError(null)
    setShowDiscardConfirm(false)
    setStarted(false)
  }

  const allAcptIndices = (rp) => rp.map((p, i) => (p.isAcpt ? i : null)).filter((i) => i !== null)

  async function handleMapEvent(evt) {
    const rp = stateRef.current.routePoints

    const isExtend = evt.type === 'click_empty' || (evt.type === 'dialog_result' && evt.action === 'extend')
    if (isExtend) {
      const acptExists = rp.some((p) => p.isAcpt)
      if (!acptExists) {
        dispatch({ type: 'ADD_FIRST_POINT', payload: { lat: evt.lat, lon: evt.lng } })
      } else {
        const lastAcpt = [...rp].reverse().find((p) => p.isAcpt)
        const seg = await calcRouteSegment([[lastAcpt.lat, lastAcpt.lon], [evt.lat, evt.lng]])
        dispatch({ type: 'EXTEND', payload: { segmentPoints: seg } })
      }
      return
    }

    if (evt.type === 'acpt_drag_end') {
      const allAcpts = allAcptIndices(rp)
      const acptIndex = evt.acptIdx
      const isFirst = acptIndex === 0
      const isLast = acptIndex === allAcpts.length - 1
      let backwardSegment = null
      let forwardSegment = null
      if (isFirst) {
        const nxtIdx = nextBoundary(allAcpts[0], rp)
        forwardSegment = await calcRouteSegment([[evt.lat, evt.lng], [rp[nxtIdx].lat, rp[nxtIdx].lon]])
      } else if (isLast) {
        const prevIdx = prevBoundary(allAcpts[allAcpts.length - 1], rp)
        backwardSegment = await calcRouteSegment([[rp[prevIdx].lat, rp[prevIdx].lon], [evt.lat, evt.lng]])
      } else {
        const trkptIdx = allAcpts[acptIndex]
        const prevIdx = prevBoundary(trkptIdx, rp)
        const nxtIdx = nextBoundary(trkptIdx, rp)
        backwardSegment = await calcRouteSegment([[rp[prevIdx].lat, rp[prevIdx].lon], [evt.lat, evt.lng]])
        forwardSegment = await calcRouteSegment([[evt.lat, evt.lng], [rp[nxtIdx].lat, rp[nxtIdx].lon]])
      }
      dispatch({ type: 'ACPT_DRAG_END', payload: { acptIndex, backwardSegment, forwardSegment } })
      return
    }

    if (evt.type === 'acpt_delete') {
      const allAcpts = allAcptIndices(rp)
      const acptIndex = evt.acptIdx
      const isFirst = acptIndex === 0
      const isLast = acptIndex === allAcpts.length - 1
      let middleSegment
      if (!isFirst && !isLast) {
        const trkptIdx = allAcpts[acptIndex]
        const prevIdx = prevBoundary(trkptIdx, rp)
        const nxtIdx = nextBoundary(trkptIdx, rp)
        middleSegment = await calcRouteSegment([[rp[prevIdx].lat, rp[prevIdx].lon], [rp[nxtIdx].lat, rp[nxtIdx].lon]])
      }
      dispatch({ type: 'ACPT_DELETE', payload: { acptIndex, middleSegment } })
      return
    }

    if (evt.type === 'dialog_result' && evt.action === 'acpt') {
      const nearIdx = evt.nearestTrkptIdx
      const prevIdx = prevBoundary(nearIdx, rp)
      const nxtIdx = nextBoundary(nearIdx, rp)
      const backwardSegment = await calcRouteSegment([[rp[prevIdx].lat, rp[prevIdx].lon], [rp[nearIdx].lat, rp[nearIdx].lon]])
      const forwardSegment = await calcRouteSegment([[rp[nearIdx].lat, rp[nearIdx].lon], [rp[nxtIdx].lat, rp[nxtIdx].lon]])
      dispatch({ type: 'INSERT_ACPT', payload: { trkptIndex: nearIdx, backwardSegment, forwardSegment } })
      return
    }

    if (evt.type === 'dialog_result' && evt.action === 'wpt') {
      const nearIdx = evt.nearestTrkptIdx
      if (rp[nearIdx].wpt !== null) return
      const point = { lat: rp[nearIdx].lat, lon: rp[nearIdx].lon, index: nearIdx }
      const inames = await fetchIntersectionNames([point], { httpTimeout: 5, maxAttempts: 1 })
      const intersectionName = inames[nearIdx] ?? null
      let poiName = null
      if (!intersectionName) {
        poiName = await fetchSpotName(rp[nearIdx].lat, rp[nearIdx].lon, { httpTimeout: 5, maxAttempts: 1 })
      }
      dispatch({ type: 'INSERT_WPT', payload: { trkptIndex: nearIdx, intersectionName, poiName } })
      return
    }

    if (evt.type === 'wpt_delete') {
      dispatch({ type: 'DELETE_WPT', payload: { trkptIndex: evt.trkptIdx } })
      return
    }

    if (evt.type === 'wpt_click') {
      const wpts = rp.map((p, i) => (p.wpt ? { i } : null)).filter(Boolean)
      const target = wpts[evt.wptIdx]
      if (target) {
        setFocusCenter({ lat: rp[target.i].lat, lng: rp[target.i].lon })
        setFocusWpt({ trkptIdx: target.i })
      }
      dispatch({ type: 'WPT_CLICK', payload: { wptIndex: evt.wptIdx } })
    }
  }

  function handleClickKm(km) {
    const rp = stateRef.current.routePoints
    if (rp.length < 2) return
    const cumDistsM = cumulativeDistances(rp.map((p) => [p.lat, p.lon]))
    const idx = nearestIndexAtDistance(cumDistsM, km * 1000)
    if (rp[idx].wpt !== null) {
      const wpts = rp.map((p, i) => (p.wpt ? { i } : null)).filter(Boolean)
      const wptIdx = wpts.findIndex((w) => w.i === idx)
      setFocusCenter({ lat: rp[idx].lat, lng: rp[idx].lon })
      setFocusWpt({ trkptIdx: idx })
      dispatch({ type: 'WPT_CLICK', payload: { wptIndex: wptIdx } })
    } else {
      mapViewRef.current?.openInsertMenuAtTrkpt(idx)
    }
  }

  async function handleDetectTurns() {
    const rp = stateRef.current.routePoints
    const changedIdx = rp.map((p, i) => (p.changed ? i : null)).filter((i) => i !== null)
    if (!changedIdx.length) return
    const ranges = contiguousRanges(changedIdx)
    const assignments = []
    for (const [rstart, rend] of ranges) {
      const sub = rp.slice(rstart, rend + 1).map((p) => [p.lat, p.lon])
      const raw = detectTurns(sub, { minTurnAngle: 45, minDist: 100, smooth: 1 })
      const cands = raw.map((t) => ({
        lat: rp[rstart + t.index].lat,
        lon: rp[rstart + t.index].lon,
        index: rstart + t.index,
        delta: t.delta,
      }))
      const inames = await fetchIntersectionNames(cands)
      cands.forEach((c) => {
        assignments.push({ trkptIndex: c.index, delta: c.delta, name: combineTurnName(c.delta, inames[c.index] ?? null) })
      })
    }
    dispatch({ type: 'APPLY_TURN_DETECTION', payload: { assignments } })
  }

  const { trkptsForMap, acptsForMap, wptsForMap, center, totalDistKm, gainM } = useMemo(() => {
    const rp = state.routePoints
    const trkptsForMap = rp.map((p) => [p.lat, p.lon])
    const acptsForMap = rp
      .map((p, i) => (p.isAcpt ? { lat: p.lat, lng: p.lon, trkptIdx: i } : null))
      .filter(Boolean)
    const wptsForMap = rp
      .map((p, i) => {
        if (!p.wpt) return null
        const [, color] = wptStyle(p.wpt)
        return { lat: p.lat, lng: p.lon, trkptIdx: i, name: p.wpt.name, color }
      })
      .filter(Boolean)

    let center = DEFAULT_CENTER
    if (rp.length) {
      const q = Math.floor(rp.length / 4)
      center = { lat: rp[q].lat, lng: rp[q].lon }
    }

    let totalDistKm = 0
    for (let i = 0; i < rp.length - 1; i++) {
      totalDistKm += haversine(rp[i].lat, rp[i].lon, rp[i + 1].lat, rp[i + 1].lon)
    }
    totalDistKm /= 1000

    const eleKey = state.eleChoice === 'fix' ? 'eleFix' : 'eleOrg'
    const elevs = rp.map((p) => p[eleKey])
    let gainM = null
    if (elevs.some((v) => v !== null && v !== undefined)) {
      gainM = 0
      for (let i = 0; i < elevs.length - 1; i++) {
        if (elevs[i] !== null && elevs[i + 1] !== null) {
          gainM += Math.max(0, elevs[i + 1] - elevs[i])
        }
      }
    }

    return { trkptsForMap, acptsForMap, wptsForMap, center, totalDistKm, gainM }
  }, [state.routePoints, state.eleChoice])

  // spec.txt 4章: GPXのトラック名 → ファイル名 → 新規ルート の優先順で決定
  const routeDisplayName = trackName || gpxFilename || '新規ルート'

  return (
    <div className="editor">
      <h1>
        🚴 gpx-editor <span className="app-version">v{APP_VERSION}</span>
      </h1>
      {!started && (
        <StartModal
          error={error}
          onFileChange={handleFileChange}
          onOpenNetworkPicker={() => setShowNetworkDialog(true)}
          onNewRoute={handleNewRoute}
          isLoggedIn={isLoggedIn}
          userEmail={user?.email ?? null}
          onSendMagicLink={sendMagicLink}
          onSignOut={signOut}
        />
      )}
      <div className="toolbar">
        {started && (
          <button type="button" className="btn-secondary" onClick={() => setShowDiscardConfirm(true)}>
            ↩ 編集を破棄して戻る
          </button>
        )}
        {error && <span className="error">{error}</span>}
        {state.routePoints.length > 0 && (
          <span className="metric">
            {routeDisplayName}
            {eleSourceGsi && '（GSI標高）'}
            　総距離 {totalDistKm.toFixed(1)} km　獲得標高 {gainM !== null ? `${Math.round(gainM)} m` : '--- m'}
          </span>
        )}
        {state.routePoints.length > 0 && <StatusBadge status={eleStatus} onRetry={retryEleFailed} />}
        {state.routePoints.length > 0 && (
          <button type="button" className="btn-primary btn-save-route" disabled={wptsForMap.length === 0} onClick={() => setShowSaveDialog(true)}>
            💾 ルートを保存
          </button>
        )}
      </div>
      {showDiscardConfirm && (
        <DiscardConfirmModal onDiscard={handleDiscardConfirmed} onCancel={() => setShowDiscardConfirm(false)} />
      )}
      <MapMatchingDialog state={mapMatchState} onCancel={cancelMapMatching} />
      {showNetworkDialog && (
        <NetworkPickerDialog
          onCancel={() => setShowNetworkDialog(false)}
          onLoaded={handleNetworkLoaded}
          isLoggedIn={isLoggedIn}
        />
      )}
      {showSaveDialog && (
        <SaveDialog
          routePoints={state.routePoints}
          gradeOrg={state.gradeOrg}
          eleChoice={state.eleChoice}
          routeModified={state.routeModified}
          dispatch={dispatch}
          rawGpxString={rawGpxString}
          defaultFilename={gpxFilename || 'new_route'}
          totalDistKm={totalDistKm}
          gainM={gainM}
          onClose={() => setShowSaveDialog(false)}
          isLoggedIn={isLoggedIn}
        />
      )}
      <div className="main-area">
        <div className="map-col">
          <MapView
            ref={mapViewRef}
            trkpts={trkptsForMap}
            acpts={acptsForMap}
            wpts={wptsForMap}
            center={center}
            zoom={13}
            onEvent={handleMapEvent}
            focusCenter={focusCenter}
            hoveredKm={hoveredKm}
          />
          <ElevationChart
            routePoints={state.routePoints}
            hoveredKm={hoveredKm}
            onHoverKm={setHoveredKm}
            onClickKm={handleClickKm}
          />
        </div>
        <div className="list-col">
          <TurnPointList
            routePoints={state.routePoints}
            routeModified={state.routeModified}
            canUndo={state.undoSnapshot !== null}
            dispatch={dispatch}
            onDetectTurns={handleDetectTurns}
            onFocus={setFocusCenter}
            focusWpt={focusWpt}
          />
        </div>
      </div>
    </div>
  )
}

export default App
