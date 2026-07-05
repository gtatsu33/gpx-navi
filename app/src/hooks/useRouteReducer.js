import { useReducer } from 'react'
import { angleDiff, calculateBearing, nearestPointIndexFrom } from '../lib/geo.js'
import { cleanElevationSpikes, computeGradeStats } from '../lib/elevation.js'
import { deepCopyRoutePoints, makeRoutePoint, nextBoundary, prevBoundary } from '../lib/routePoints.js'
import { combineTurnName } from '../lib/turns.js'

/**
 * spec.txt 8章の各イベントに対応するreducer。
 * ルーティングAPI呼び出し等の非同期処理は行わず、呼び出し側（hooks/lib層）で
 * 事前に計算済みの座標列を action.payload として受け取る（implement.txt 5章）。
 */

/**
 * points（[[lat,lon],...]）＋対応するele値からRoutePoint配列を組み立て、
 * 元データ標高のスパイク除去・勾配統計・acpt復元・waypoint割り当てまで行う。
 * spec.txt 6章。LOAD_PARSED_GPX（ルートデータ）・LOAD_MATCHED_ROUTE
 * （実走行データ、マップマッチング後）の両方から使う共通ロジック。
 */
function buildInitialRoutePoints(points, elevations, waypoints, acptIndices, turnAssignments, eleSourceGsi) {
  const useExtensionAcpts = acptIndices && acptIndices.size >= 2
  const rp = points.map(([lat, lon], i) =>
    makeRoutePoint(lat, lon, {
      eleOrg: elevations[i] ?? null,
      isAcpt: useExtensionAcpts ? acptIndices.has(i) : i === 0 || i === points.length - 1,
      changed: false,
    })
  )

  let gradeOrg = null
  if (rp.length && rp.some((p) => p.eleOrg !== null && p.eleOrg !== undefined)) {
    const { cleaned } = cleanElevationSpikes(points, elevations.map((v) => v ?? null))
    cleaned.forEach((v, i) => {
      rp[i] = { ...rp[i], eleOrg: v }
    })
    gradeOrg = computeGradeStats(points, cleaned)
  }

  // spec.txt 5-4章・16-2章: 読み込んだGPXが既に国土地理院データ（eleSource=gsi）
  // であれば、ele_fixにも同じ値を入れておき、保存画面での無駄な再取得を避ける。
  let gradeFix = null
  if (eleSourceGsi) {
    rp.forEach((p, i) => {
      if (p.eleOrg !== null && p.eleOrg !== undefined) {
        rp[i] = { ...rp[i], eleFix: p.eleOrg }
      }
    })
    gradeFix = gradeOrg
  }

  if (waypoints.length && rp.length) {
    // spec.txt 6章「既にwptを含むGPXの場合」: 最近傍trkptへの割り当て。
    // 先頭・末尾は無ければ補うのみ（既存のwpt名は尊重する）
    let searchFrom = 0
    waypoints.forEach((w) => {
      let delta = null
      if (w.desc && w.desc.startsWith('bearing_change:')) {
        const parsed = parseFloat(w.desc.split(':')[1])
        if (!Number.isNaN(parsed)) delta = parsed
      }
      const idx = nearestPointIndexFrom(points, w.lat, w.lon, searchFrom)
      rp[idx] = { ...rp[idx], wpt: { name: w.name || 'ターンポイント', delta } }
      searchFrom = idx
    })
    if (rp[0].wpt === null) rp[0] = { ...rp[0], wpt: { name: 'スタート', delta: null } }
    if (rp[rp.length - 1].wpt === null) {
      rp[rp.length - 1] = { ...rp[rp.length - 1], wpt: { name: '目的地', delta: null } }
    }
  } else if (turnAssignments && turnAssignments.length && rp.length) {
    // spec.txt 6章「wptを含まないGPXの場合」: 自動検出済みのターンを適用し、
    // 先頭・末尾は無条件で「スタート」「目的地」に上書きする
    turnAssignments.forEach(({ trkptIndex, delta, name }) => {
      rp[trkptIndex] = { ...rp[trkptIndex], wpt: { name, delta } }
    })
    rp[0] = { ...rp[0], wpt: { name: 'スタート', delta: null } }
    rp[rp.length - 1] = { ...rp[rp.length - 1], wpt: { name: '目的地', delta: null } }
  } else if (rp.length) {
    // ターン自動検出の結果がまだ無い場合でも、先頭・末尾のwptだけは設定しておく
    rp[0] = { ...rp[0], wpt: { name: 'スタート', delta: null } }
    rp[rp.length - 1] = { ...rp[rp.length - 1], wpt: { name: '目的地', delta: null } }
  }

  return { rp, gradeOrg, gradeFix }
}

export function initialRouteState() {
  return {
    routePoints: [],
    undoSnapshot: null,
    routeModified: false,
    gradeOrg: null,
    gradeFix: null,
    eleChoice: 'org',
  }
}

function withUndoSnapshot(state) {
  return deepCopyRoutePoints(state.routePoints)
}

function findAcptIndices(rp) {
  return rp.reduce((acc, p, i) => (p.isAcpt ? [...acc, i] : acc), [])
}

function computeDelta(rp, idx) {
  if (idx <= 0 || idx >= rp.length - 1) return null
  const bearingIn = calculateBearing(rp[idx - 1].lat, rp[idx - 1].lon, rp[idx].lat, rp[idx].lon)
  const bearingOut = calculateBearing(rp[idx].lat, rp[idx].lon, rp[idx + 1].lat, rp[idx + 1].lon)
  return angleDiff(bearingIn, bearingOut)
}

export function routeReducer(state, action) {
  switch (action.type) {
    // 8-1（前半）: ルートが空の状態からの最初の1点追加。Undo対象外（14章）
    case 'ADD_FIRST_POINT': {
      const { lat, lon } = action.payload
      const newPoint = makeRoutePoint(lat, lon, {
        isAcpt: true,
        wpt: { name: 'スタート', delta: null },
        changed: false,
      })
      return { ...state, routePoints: [newPoint], routeModified: true }
    }

    // 8-1（後半）: acptが既に存在する場合のゴール延伸。Undo対象
    case 'EXTEND': {
      const { segmentPoints } = action.payload
      const rp = [...state.routePoints]
      const undoSnapshot = withUndoSnapshot(state)
      if (rp.length && rp[rp.length - 1].wpt && rp[rp.length - 1].wpt.name === '目的地') {
        rp[rp.length - 1] = { ...rp[rp.length - 1], wpt: null }
      }
      const tail = segmentPoints.slice(1)
      const newPts = tail.map((pt, j) => makeRoutePoint(pt[0], pt[1], { isAcpt: j === tail.length - 1, changed: true }))
      if (newPts.length) {
        newPts[newPts.length - 1] = {
          ...newPts[newPts.length - 1],
          wpt: { name: '目的地', delta: null },
          changed: false,
        }
      }
      return { ...state, routePoints: [...rp, ...newPts], undoSnapshot, routeModified: true }
    }

    // 8-2: acptの移動。Undo対象
    case 'ACPT_DRAG_END': {
      const { acptIndex, backwardSegment, forwardSegment } = action.payload
      const rp = [...state.routePoints]
      const allAcpts = findAcptIndices(rp)
      if (acptIndex < 0 || acptIndex >= allAcpts.length) return state
      const undoSnapshot = withUndoSnapshot(state)
      const trkptIdx = allAcpts[acptIndex]
      const isFirst = acptIndex === 0
      const isLast = acptIndex === allAcpts.length - 1
      let newRp

      if (isFirst) {
        const nxtIdx = nextBoundary(trkptIdx, rp)
        const head = forwardSegment.slice(0, -1)
        const newPts = head.map((pt, j) => makeRoutePoint(pt[0], pt[1], { isAcpt: j === 0, changed: true }))
        if (newPts.length) {
          newPts[0] = { ...newPts[0], wpt: { name: 'スタート', delta: null }, changed: false }
        }
        newRp = [...newPts, ...rp.slice(nxtIdx)]
      } else if (isLast) {
        const prevIdx = prevBoundary(trkptIdx, rp)
        const tail = backwardSegment.slice(1)
        const newPts = tail.map((pt, j) => makeRoutePoint(pt[0], pt[1], { isAcpt: j === tail.length - 1, changed: true }))
        if (newPts.length) {
          newPts[newPts.length - 1] = {
            ...newPts[newPts.length - 1],
            wpt: { name: '目的地', delta: null },
            changed: false,
          }
        }
        newRp = [...rp.slice(0, prevIdx + 1), ...newPts]
      } else {
        const prevIdx = prevBoundary(trkptIdx, rp)
        const nxtIdx = nextBoundary(trkptIdx, rp)
        const bwdTail = backwardSegment.slice(1).map((pt) => makeRoutePoint(pt[0], pt[1], { changed: true }))
        const fwdMid = forwardSegment.slice(1, -1).map((pt) => makeRoutePoint(pt[0], pt[1], { changed: true }))
        const newPts = [...bwdTail, ...fwdMid]
        const acptPos = bwdTail.length - 1
        if (acptPos >= 0 && acptPos < newPts.length) {
          newPts[acptPos] = { ...newPts[acptPos], isAcpt: true }
        }
        newRp = [...rp.slice(0, prevIdx + 1), ...newPts, ...rp.slice(nxtIdx)]
      }

      return { ...state, routePoints: newRp, undoSnapshot, routeModified: true }
    }

    // 8-3: acptの削除。Undo対象
    case 'ACPT_DELETE': {
      const { acptIndex, middleSegment } = action.payload
      const rp = [...state.routePoints]
      const allAcpts = findAcptIndices(rp)
      if (acptIndex < 0 || acptIndex >= allAcpts.length) return state
      const undoSnapshot = withUndoSnapshot(state)
      const isFirst = acptIndex === 0
      const isLast = acptIndex === allAcpts.length - 1
      let newRp

      if (isFirst) {
        if (allAcpts.length <= 1) {
          newRp = []
        } else {
          const nxtIdx = allAcpts[1]
          newRp = rp.slice(nxtIdx)
          if (newRp.length) {
            newRp[0] = { ...newRp[0], isAcpt: true }
            if (!newRp[0].wpt || newRp[0].wpt.name !== 'スタート') {
              newRp[0] = { ...newRp[0], wpt: { name: 'スタート', delta: null } }
            }
          }
        }
      } else if (isLast) {
        if (allAcpts.length <= 1) {
          newRp = []
        } else {
          const prevIdx = allAcpts[allAcpts.length - 2]
          newRp = rp.slice(0, prevIdx + 1)
          if (newRp.length) {
            const lastI = newRp.length - 1
            newRp[lastI] = { ...newRp[lastI], isAcpt: true }
            if (!newRp[lastI].wpt || newRp[lastI].wpt.name !== '目的地') {
              newRp[lastI] = { ...newRp[lastI], wpt: { name: '目的地', delta: null } }
            }
          }
        }
      } else {
        const trkptIdx = allAcpts[acptIndex]
        const prevIdx = prevBoundary(trkptIdx, rp)
        const nxtIdx = nextBoundary(trkptIdx, rp)
        const newMid = middleSegment.slice(1, -1).map((pt) => makeRoutePoint(pt[0], pt[1], { changed: true }))
        newRp = [...rp.slice(0, prevIdx + 1), ...newMid, ...rp.slice(nxtIdx)]
      }

      return { ...state, routePoints: newRp, undoSnapshot, routeModified: true }
    }

    // 8-4: アンカーポイントの挿入。Undo対象
    case 'INSERT_ACPT': {
      const { trkptIndex, backwardSegment, forwardSegment } = action.payload
      const rp = [...state.routePoints]
      const undoSnapshot = withUndoSnapshot(state)
      const prevIdx = prevBoundary(trkptIndex, rp)
      const nxtIdx = nextBoundary(trkptIndex, rp)
      const seg1Tail = backwardSegment.slice(1).map((pt) => makeRoutePoint(pt[0], pt[1], { changed: true }))
      const seg2Mid = forwardSegment.slice(1, -1).map((pt) => makeRoutePoint(pt[0], pt[1], { changed: true }))
      const newPts = [...seg1Tail, ...seg2Mid]
      const newAcptPos = seg1Tail.length - 1
      if (newAcptPos >= 0 && newAcptPos < newPts.length) {
        newPts[newAcptPos] = { ...newPts[newAcptPos], isAcpt: true }
      }
      const newRp = [...rp.slice(0, prevIdx + 1), ...newPts, ...rp.slice(nxtIdx)]
      return { ...state, routePoints: newRp, undoSnapshot, routeModified: true }
    }

    // 8-5: ターンポイントの追加。Undo対象（今回のUndo拡張で追加）
    case 'INSERT_WPT': {
      const { trkptIndex, intersectionName, poiName } = action.payload
      const rp = [...state.routePoints]
      if (rp[trkptIndex].wpt !== null) return state
      const undoSnapshot = withUndoSnapshot(state)
      const delta = computeDelta(rp, trkptIndex)
      let name
      if (delta !== null) {
        name = combineTurnName(delta, intersectionName)
      } else if (intersectionName) {
        name = intersectionName
      } else {
        name = poiName ? `「${poiName}」` : '追加したターンポイント'
      }
      rp[trkptIndex] = { ...rp[trkptIndex], wpt: { name, delta } }
      return { ...state, routePoints: rp, undoSnapshot }
    }

    // 8-6: 一覧パネルへのフォーカス。route_pointsを変更しないためUndo対象外
    case 'WPT_CLICK':
      return state

    // 15章: 名前入力欄の直接編集。連続入力のためUndo対象外（14章）
    case 'RENAME_WPT': {
      const { trkptIndex, name } = action.payload
      if (!state.routePoints[trkptIndex].wpt) return state
      const rp = [...state.routePoints]
      rp[trkptIndex] = { ...rp[trkptIndex], wpt: { ...rp[trkptIndex].wpt, name } }
      return { ...state, routePoints: rp }
    }

    // 14章: Undo（1回分のみ）
    case 'UNDO': {
      if (!state.undoSnapshot) return state
      const restored = state.undoSnapshot
      const routeModified = restored.some((p) => p.wpt !== null && p.changed)
      return { ...state, routePoints: restored, undoSnapshot: null, routeModified }
    }

    // 15章「🔍 ターンポイント検出」一括実行。Undo対象
    case 'APPLY_TURN_DETECTION': {
      const { assignments } = action.payload // [{ trkptIndex, delta, name }]
      const undoSnapshot = withUndoSnapshot(state)
      const rp = state.routePoints.map((p) => ({ ...p }))
      assignments.forEach(({ trkptIndex, delta, name }) => {
        if (rp[trkptIndex].wpt === null) {
          rp[trkptIndex] = { ...rp[trkptIndex], wpt: { name, delta } }
        }
      })
      const finalRp = rp.map((p) => ({ ...p, changed: false }))
      return { ...state, routePoints: finalRp, undoSnapshot, routeModified: false }
    }

    // 15章「🗑」削除ボタン。Undo対象（今回のUndo拡張で追加）
    case 'DELETE_WPT': {
      const { trkptIndex } = action.payload
      const undoSnapshot = withUndoSnapshot(state)
      const rp = [...state.routePoints]
      rp[trkptIndex] = { ...rp[trkptIndex], wpt: null, isAcpt: true }
      return { ...state, routePoints: rp, undoSnapshot }
    }

    // 16-2章: 保存画面での標高整合性チェック後、org/fixどちらを使うか選択する。
    // 表示切替のみでroute_pointsを変更しないためUndo対象外（14章）
    case 'SET_ELE_CHOICE': {
      return { ...state, eleChoice: action.payload.choice }
    }

    // 16-2章: 保存画面の標高整合性チェックの確定処理。呼び出し側（SaveDialog.jsx）
    // で追加取得・スパイク除去・勾配統計・推奨判定まで済ませた結果を一括反映する。
    case 'FINALIZE_SAVE_ELEVATION': {
      const { fixValues, gradeFix, choice } = action.payload
      const rp = state.routePoints.map((p, i) => ({ ...p, eleFix: fixValues[i] }))
      return { ...state, routePoints: rp, gradeFix, eleChoice: choice }
    }

    // パース済みGPXを読み込む（ルートデータ、またはwpt付きGPX＝
    // マップマッチング済みのためスキップする経路）。turnAssignmentsは
    // wptを含まないGPXの場合にApp.jsx側で事前計算されたターン自動検出
    // 結果（spec.txt 6章・11章・12章、非同期のOverpass呼び出しを伴うため
    // reducerの外で計算する）。
    case 'LOAD_PARSED_GPX': {
      const { trkpts, waypoints, acptIndices, turnAssignments, eleSourceGsi } = action.payload
      const points = trkpts.map((t) => [t.lat, t.lon])
      const elevations = trkpts.map((t) => t.ele)
      const { rp, gradeOrg, gradeFix } = buildInitialRoutePoints(points, elevations, waypoints, acptIndices, turnAssignments, eleSourceGsi)
      return { ...state, routePoints: rp, undoSnapshot: null, routeModified: false, gradeOrg, gradeFix }
    }

    // Phase8: 実走行データのマップマッチング後に読み込む。spec.txt 6章・10章。
    // 呼び出し側（App.jsx）でRDP間引き→マップマッチングまで完了させ、
    // マッチング後の座標列と、間引き前インデックスで対応付けた元標高を渡す。
    case 'LOAD_MATCHED_ROUTE': {
      const { matchedPoints, origElevations, waypoints, acptIndices, turnAssignments, eleSourceGsi } = action.payload
      const { rp, gradeOrg, gradeFix } = buildInitialRoutePoints(matchedPoints, origElevations, waypoints, acptIndices, turnAssignments, eleSourceGsi)
      return { ...state, routePoints: rp, undoSnapshot: null, routeModified: false, gradeOrg, gradeFix }
    }

    // 13-2章: リアルタイム背景取得で1点分のele_fixが確定した際に反映する。
    // Undo対象外（標高補正データはユーザーの編集操作ではないため）。
    case 'SET_ELE_FIX_BATCH': {
      const { assignments } = action.payload // [{ trkptIndex, value }]
      const rp = [...state.routePoints]
      assignments.forEach(({ trkptIndex, value }) => {
        if (rp[trkptIndex]) {
          rp[trkptIndex] = { ...rp[trkptIndex], eleFix: value }
        }
      })
      return { ...state, routePoints: rp }
    }

    // 13-2章: 全点の取得試行が完了した時点で1回だけスパイク除去＋勾配統計を実行する。
    // 計算結果が既存のeleFixと全点一致する場合はstateをそのまま返す（新しい配列参照を
    // 作らない）。そうしないとroutePointsの参照が変わるたびにuseElevationBackgroundの
    // debounce effectが再発火し、対象0件でも本actionを再dispatchし続ける無限ループになる。
    case 'FINALIZE_ELE_FIX': {
      const rp = state.routePoints
      const fixVals = rp.map((p) => p.eleFix)
      if (!rp.length || fixVals.every((v) => v === null)) return state
      const points2d = rp.map((p) => [p.lat, p.lon])
      const { cleaned } = cleanElevationSpikes(points2d, fixVals)
      if (state.gradeFix !== null && cleaned.every((v, i) => v === fixVals[i])) return state
      const newRp = rp.map((p, i) => ({ ...p, eleFix: cleaned[i] }))
      const gradeFix = computeGradeStats(points2d, cleaned)
      return { ...state, routePoints: newRp, gradeFix }
    }

    // implement.txt 1章: 破棄確認モーダルで「スタート画面に戻る」を選択した際、
    // 編集用の内部状態を全てクリアする
    case 'RESET':
      return initialRouteState()

    default:
      return state
  }
}

export function useRouteReducer() {
  return useReducer(routeReducer, undefined, initialRouteState)
}
