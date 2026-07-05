import { describe, expect, it } from 'vitest'
import { makeRoutePoint } from '../lib/routePoints.js'
import { initialRouteState, routeReducer } from './useRouteReducer.js'

function baseFivePointRoute() {
  return {
    ...initialRouteState(),
    routePoints: [
      makeRoutePoint(35.0, 139.0, { isAcpt: true, wpt: { name: 'スタート', delta: null }, changed: false }),
      makeRoutePoint(35.001, 139.0, { changed: false }),
      makeRoutePoint(35.002, 139.0, { isAcpt: true, changed: false }),
      makeRoutePoint(35.003, 139.0, { changed: false }),
      makeRoutePoint(35.004, 139.0, { isAcpt: true, wpt: { name: '目的地', delta: null }, changed: false }),
    ],
  }
}

describe('routeReducer', () => {
  describe('ADD_FIRST_POINT（8-1前半・Undo対象外）', () => {
    it('ルートが空から最初の1点を追加する', () => {
      const state = initialRouteState()
      const next = routeReducer(state, { type: 'ADD_FIRST_POINT', payload: { lat: 35, lon: 139 } })
      expect(next.routePoints).toHaveLength(1)
      expect(next.routePoints[0]).toMatchObject({ isAcpt: true, wpt: { name: 'スタート', delta: null } })
      expect(next.routeModified).toBe(true)
    })

    it('Undoスナップショットを保存しない', () => {
      const state = { ...initialRouteState(), undoSnapshot: null }
      const next = routeReducer(state, { type: 'ADD_FIRST_POINT', payload: { lat: 35, lon: 139 } })
      expect(next.undoSnapshot).toBeNull()
    })
  })

  describe('EXTEND（8-1後半・Undo対象）', () => {
    it('区間を末尾に追加し、末尾に目的地wptを設定する', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, {
        type: 'EXTEND',
        payload: { segmentPoints: [[35.004, 139.0], [35.005, 139.0], [35.006, 139.0]] },
      })
      expect(next.routePoints).toHaveLength(7)
      const last = next.routePoints[next.routePoints.length - 1]
      expect(last).toMatchObject({ isAcpt: true, wpt: { name: '目的地', delta: null } })
      expect(next.routeModified).toBe(true)
      expect(next.undoSnapshot).toEqual(state.routePoints)
    })

    it('旧・目的地wptを解除する', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, {
        type: 'EXTEND',
        payload: { segmentPoints: [[35.004, 139.0], [35.005, 139.0]] },
      })
      const oldGoalPoint = next.routePoints[4]
      expect(oldGoalPoint.wpt).toBeNull()
    })
  })

  describe('ACPT_DRAG_END（8-2・Undo対象）', () => {
    it('先頭acptを移動する', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, {
        type: 'ACPT_DRAG_END',
        payload: { acptIndex: 0, backwardSegment: null, forwardSegment: [[36.0, 139.0], [35.002, 139.0]] },
      })
      expect(next.routePoints).toHaveLength(4)
      expect(next.routePoints[0]).toMatchObject({ lat: 36.0, isAcpt: true, wpt: { name: 'スタート', delta: null } })
      expect(next.undoSnapshot).toEqual(state.routePoints)
    })

    it('末尾acptを移動する', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, {
        type: 'ACPT_DRAG_END',
        payload: { acptIndex: 2, backwardSegment: [[35.002, 139.0], [35.005, 139.001]], forwardSegment: null },
      })
      expect(next.routePoints).toHaveLength(4)
      const last = next.routePoints[next.routePoints.length - 1]
      expect(last).toMatchObject({ lat: 35.005, isAcpt: true, wpt: { name: '目的地', delta: null } })
    })

    it('中間acptを移動する', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, {
        type: 'ACPT_DRAG_END',
        payload: {
          acptIndex: 1,
          backwardSegment: [[35.0, 139.0], [35.0015, 139.0005]],
          forwardSegment: [[35.0015, 139.0005], [35.004, 139.0]],
        },
      })
      expect(next.routePoints).toHaveLength(3)
      expect(next.routePoints[1]).toMatchObject({ lat: 35.0015, lon: 139.0005, isAcpt: true })
    })
  })

  describe('ACPT_DELETE（8-3・Undo対象）', () => {
    it('先頭acptを削除する（複数acptあり）', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, { type: 'ACPT_DELETE', payload: { acptIndex: 0 } })
      expect(next.routePoints).toHaveLength(3)
      expect(next.routePoints[0]).toMatchObject({ isAcpt: true, wpt: { name: 'スタート', delta: null } })
    })

    it('末尾acptを削除する（複数acptあり）', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, { type: 'ACPT_DELETE', payload: { acptIndex: 2 } })
      expect(next.routePoints).toHaveLength(3)
      const last = next.routePoints[next.routePoints.length - 1]
      expect(last).toMatchObject({ isAcpt: true, wpt: { name: '目的地', delta: null } })
    })

    it('中間acptを削除して区間を再計算する', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, {
        type: 'ACPT_DELETE',
        payload: { acptIndex: 1, middleSegment: [[35.0, 139.0], [35.002, 139.0005], [35.004, 139.0]] },
      })
      expect(next.routePoints).toHaveLength(3)
      expect(next.routePoints[1]).toMatchObject({ lat: 35.002, lon: 139.0005 })
    })

    it('acptが1つだけの場合はルート全体を空にする', () => {
      const state = {
        ...initialRouteState(),
        routePoints: [makeRoutePoint(35, 139, { isAcpt: true, wpt: { name: 'スタート', delta: null } })],
      }
      const next = routeReducer(state, { type: 'ACPT_DELETE', payload: { acptIndex: 0 } })
      expect(next.routePoints).toEqual([])
    })
  })

  describe('INSERT_ACPT（8-4・Undo対象）', () => {
    it('指定trkptにアンカーポイントを挿入する', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, {
        type: 'INSERT_ACPT',
        payload: {
          trkptIndex: 1,
          backwardSegment: [[35.0, 139.0], [35.001, 139.0]],
          forwardSegment: [[35.001, 139.0], [35.002, 139.0]],
        },
      })
      expect(next.routePoints).toHaveLength(5)
      expect(next.routePoints[1]).toMatchObject({ isAcpt: true })
    })
  })

  describe('INSERT_WPT（8-5・Undo対象）', () => {
    function bendRoute() {
      return {
        ...initialRouteState(),
        routePoints: [
          makeRoutePoint(35.0, 139.0, { isAcpt: true, wpt: { name: 'スタート', delta: null } }),
          makeRoutePoint(35.001, 139.0),
          makeRoutePoint(35.001, 139.001, { isAcpt: true, wpt: { name: '目的地', delta: null } }),
        ],
      }
    }

    it('交差点名がある場合「{name}を{方向}」にする', () => {
      const state = bendRoute()
      const next = routeReducer(state, {
        type: 'INSERT_WPT',
        payload: { trkptIndex: 1, intersectionName: 'サンプル交差点', poiName: null },
      })
      expect(next.routePoints[1].wpt.name).toBe('サンプル交差点を右折')
      expect(next.undoSnapshot).toEqual(state.routePoints)
    })

    it('交差点名が無い場合は方向ラベルのみ', () => {
      const state = bendRoute()
      const next = routeReducer(state, {
        type: 'INSERT_WPT',
        payload: { trkptIndex: 1, intersectionName: null, poiName: null },
      })
      expect(next.routePoints[1].wpt.name).toBe('右折')
    })

    it('delta計算不可＋交差点名ありならそのまま名前にする', () => {
      const state = {
        ...initialRouteState(),
        routePoints: [
          makeRoutePoint(35.0, 139.0, { isAcpt: true }),
          makeRoutePoint(35.001, 139.0, { isAcpt: true, wpt: { name: '目的地', delta: null } }),
        ],
      }
      const next = routeReducer(state, {
        type: 'INSERT_WPT',
        payload: { trkptIndex: 0, intersectionName: 'どこかの地点', poiName: null },
      })
      expect(next.routePoints[0].wpt.name).toBe('どこかの地点')
    })

    it('delta計算不可＋交差点名なし＋POI名ありなら「」で囲む', () => {
      const state = {
        ...initialRouteState(),
        routePoints: [
          makeRoutePoint(35.0, 139.0, { isAcpt: true }),
          makeRoutePoint(35.001, 139.0, { isAcpt: true, wpt: { name: '目的地', delta: null } }),
        ],
      }
      const next = routeReducer(state, {
        type: 'INSERT_WPT',
        payload: { trkptIndex: 0, intersectionName: null, poiName: 'どこかの公園' },
      })
      expect(next.routePoints[0].wpt.name).toBe('「どこかの公園」')
    })

    it('何も見つからなければ「追加したターンポイント」', () => {
      const state = {
        ...initialRouteState(),
        routePoints: [
          makeRoutePoint(35.0, 139.0, { isAcpt: true }),
          makeRoutePoint(35.001, 139.0, { isAcpt: true, wpt: { name: '目的地', delta: null } }),
        ],
      }
      const next = routeReducer(state, {
        type: 'INSERT_WPT',
        payload: { trkptIndex: 0, intersectionName: null, poiName: null },
      })
      expect(next.routePoints[0].wpt.name).toBe('追加したターンポイント')
    })

    it('既にwptがある点には何もしない（stateをそのまま返す）', () => {
      const state = bendRoute()
      const next = routeReducer(state, {
        type: 'INSERT_WPT',
        payload: { trkptIndex: 0, intersectionName: 'x', poiName: null },
      })
      expect(next).toBe(state)
    })
  })

  describe('WPT_CLICK（8-6・Undo対象外・route_points変更なし）', () => {
    it('stateをそのまま返す', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, { type: 'WPT_CLICK', payload: { wptIndex: 0 } })
      expect(next).toBe(state)
    })
  })

  describe('RENAME_WPT（15章・Undo対象外）', () => {
    it('名前を書き換える', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, { type: 'RENAME_WPT', payload: { trkptIndex: 0, name: '新しい名前' } })
      expect(next.routePoints[0].wpt.name).toBe('新しい名前')
      expect(next.undoSnapshot).toBeNull()
    })
  })

  describe('UNDO（14章）', () => {
    it('undoSnapshotで復元し、スナップショットを消費する', () => {
      const original = baseFivePointRoute().routePoints
      const state = {
        routePoints: [makeRoutePoint(0, 0)],
        undoSnapshot: original,
        routeModified: true,
      }
      const next = routeReducer(state, { type: 'UNDO' })
      expect(next.routePoints).toBe(original)
      expect(next.undoSnapshot).toBeNull()
    })

    it('スナップショットが無ければ何もしない', () => {
      const state = { ...baseFivePointRoute(), undoSnapshot: null }
      const next = routeReducer(state, { type: 'UNDO' })
      expect(next).toBe(state)
    })
  })

  describe('APPLY_TURN_DETECTION（15章・Undo対象）', () => {
    it('wptが無い点にのみ設定し、changedを全てfalseに戻す', () => {
      const state = {
        ...initialRouteState(),
        routePoints: [
          makeRoutePoint(35.0, 139.0, { isAcpt: true, wpt: { name: 'スタート', delta: null }, changed: false }),
          makeRoutePoint(35.001, 139.0, { changed: true }),
          makeRoutePoint(35.002, 139.0, { wpt: { name: '既存', delta: 10 }, changed: true }),
        ],
      }
      const next = routeReducer(state, {
        type: 'APPLY_TURN_DETECTION',
        payload: {
          assignments: [
            { trkptIndex: 1, delta: 70, name: '右折' },
            { trkptIndex: 2, delta: 5, name: '上書きされないはず' },
          ],
        },
      })
      expect(next.routePoints[1].wpt).toEqual({ name: '右折', delta: 70 })
      expect(next.routePoints[2].wpt).toEqual({ name: '既存', delta: 10 })
      expect(next.routePoints.every((p) => p.changed === false)).toBe(true)
      expect(next.routeModified).toBe(false)
      expect(next.undoSnapshot).toEqual(state.routePoints)
    })
  })

  describe('LOAD_PARSED_GPX（Phase4暫定・表示確認用）', () => {
    it('wpt付きGPXを最近傍trkptに割り当てる', () => {
      const state = initialRouteState()
      const next = routeReducer(state, {
        type: 'LOAD_PARSED_GPX',
        payload: {
          trkpts: [
            { lat: 35.0, lon: 139.0, ele: 10 },
            { lat: 35.001, lon: 139.0, ele: 12 },
            { lat: 35.002, lon: 139.0, ele: 14 },
          ],
          waypoints: [
            { lat: 35.001, lon: 139.0, name: '右折ポイント', desc: 'bearing_change:70.0' },
          ],
        },
      })
      expect(next.routePoints).toHaveLength(3)
      expect(next.routePoints[0]).toMatchObject({ isAcpt: true, wpt: { name: 'スタート', delta: null } })
      expect(next.routePoints[1]).toMatchObject({ wpt: { name: '右折ポイント', delta: 70.0 } })
      expect(next.routePoints[2]).toMatchObject({ isAcpt: true, wpt: { name: '目的地', delta: null } })
    })

    it('eleSourceGsi=trueならele_orgと同じ値をele_fixにも複製する（5-4章・16-2章）', () => {
      const state = initialRouteState()
      const next = routeReducer(state, {
        type: 'LOAD_PARSED_GPX',
        payload: {
          trkpts: [
            { lat: 35.0, lon: 139.0, ele: 10 },
            { lat: 35.001, lon: 139.0, ele: 12 },
            { lat: 35.002, lon: 139.0, ele: 14 },
          ],
          waypoints: [],
          eleSourceGsi: true,
        },
      })
      expect(next.routePoints.map((p) => p.eleFix)).toEqual(next.routePoints.map((p) => p.eleOrg))
      expect(next.gradeFix).toEqual(next.gradeOrg)
    })

    it('eleSourceGsiが無ければele_fixはnullのまま（従来通り）', () => {
      const state = initialRouteState()
      const next = routeReducer(state, {
        type: 'LOAD_PARSED_GPX',
        payload: {
          trkpts: [
            { lat: 35.0, lon: 139.0, ele: 10 },
            { lat: 35.001, lon: 139.0, ele: 12 },
            { lat: 35.002, lon: 139.0, ele: 14 },
          ],
          waypoints: [],
        },
      })
      expect(next.routePoints.every((p) => p.eleFix === null)).toBe(true)
      expect(next.gradeFix).toBeNull()
    })

    it('gpxnavi:acpt拡張タグが2点以上あれば、それを中間点も含めて採用する', () => {
      const state = initialRouteState()
      const next = routeReducer(state, {
        type: 'LOAD_PARSED_GPX',
        payload: {
          trkpts: [
            { lat: 35.0, lon: 139.0, ele: 10 },
            { lat: 35.001, lon: 139.0, ele: 12 },
            { lat: 35.002, lon: 139.0, ele: 14 },
            { lat: 35.003, lon: 139.0, ele: 16 },
          ],
          waypoints: [],
          acptIndices: new Set([0, 2, 3]),
        },
      })
      expect(next.routePoints.map((p) => p.isAcpt)).toEqual([true, false, true, true])
    })

    it('拡張タグが1点以下しか無ければ先頭・末尾のみacptにフォールバックする', () => {
      const state = initialRouteState()
      const next = routeReducer(state, {
        type: 'LOAD_PARSED_GPX',
        payload: {
          trkpts: [
            { lat: 35.0, lon: 139.0, ele: 10 },
            { lat: 35.001, lon: 139.0, ele: 12 },
            { lat: 35.002, lon: 139.0, ele: 14 },
          ],
          waypoints: [],
          acptIndices: new Set([1]),
        },
      })
      expect(next.routePoints.map((p) => p.isAcpt)).toEqual([true, false, true])
    })

    it('wptを含まないGPXは先頭・末尾のみacptにし、スタート/目的地wptは無条件で設定する', () => {
      const state = initialRouteState()
      const next = routeReducer(state, {
        type: 'LOAD_PARSED_GPX',
        payload: {
          trkpts: [
            { lat: 35.0, lon: 139.0, ele: 10 },
            { lat: 35.001, lon: 139.0, ele: 12 },
          ],
          waypoints: [],
        },
      })
      expect(next.routePoints[0].isAcpt).toBe(true)
      expect(next.routePoints[1].isAcpt).toBe(true)
      expect(next.routePoints[0].wpt).toEqual({ name: 'スタート', delta: null })
      expect(next.routePoints[1].wpt).toEqual({ name: '目的地', delta: null })
    })

    it('turnAssignmentsが与えられれば自動検出済みターンを適用し、先頭・末尾を上書きする', () => {
      const state = initialRouteState()
      const next = routeReducer(state, {
        type: 'LOAD_PARSED_GPX',
        payload: {
          trkpts: [
            { lat: 35.0, lon: 139.0, ele: 10 },
            { lat: 35.001, lon: 139.0, ele: 12 },
            { lat: 35.002, lon: 139.0, ele: 14 },
          ],
          waypoints: [],
          turnAssignments: [{ trkptIndex: 1, delta: 70, name: 'テスト交差点を右折' }],
        },
      })
      expect(next.routePoints[1].wpt).toEqual({ name: 'テスト交差点を右折', delta: 70 })
      expect(next.routePoints[0].wpt).toEqual({ name: 'スタート', delta: null })
      expect(next.routePoints[2].wpt).toEqual({ name: '目的地', delta: null })
    })
  })

  describe('LOAD_MATCHED_ROUTE（Phase8・マップマッチング後の読込）', () => {
    it('マッチング後の座標と、間引き前対応の元標高でRoutePointを構築する', () => {
      const state = initialRouteState()
      const next = routeReducer(state, {
        type: 'LOAD_MATCHED_ROUTE',
        payload: {
          matchedPoints: [
            [35.0001, 139.0],
            [35.0011, 139.0],
            [35.0021, 139.0],
          ],
          origElevations: [10, 12, 14],
          waypoints: [],
          acptIndices: new Set(),
        },
      })
      expect(next.routePoints).toHaveLength(3)
      expect(next.routePoints[0]).toMatchObject({ lat: 35.0001, isAcpt: true, eleOrg: 10 })
      expect(next.routePoints[2]).toMatchObject({ lat: 35.0021, isAcpt: true, eleOrg: 14 })
    })
  })

  describe('SET_ELE_FIX_BATCH / FINALIZE_ELE_FIX（13-2章・リアルタイム背景取得）', () => {
    it('SET_ELE_FIX_BATCHは指定indexのele_fixのみ更新する', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, {
        type: 'SET_ELE_FIX_BATCH',
        payload: { assignments: [{ trkptIndex: 1, value: 55.5 }] },
      })
      expect(next.routePoints[1].eleFix).toBe(55.5)
      expect(next.routePoints[0].eleFix).toBeNull()
    })

    it('FINALIZE_ELE_FIXは全点nullなら何もしない', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, { type: 'FINALIZE_ELE_FIX' })
      expect(next).toBe(state)
    })

    it('FINALIZE_ELE_FIXはスパイク除去（cleanElevationSpikes）と勾配統計を反映する', () => {
      let state = baseFivePointRoute()
      const values = [100, 100.5, 101, 101.5, 102]
      state = routeReducer(state, {
        type: 'SET_ELE_FIX_BATCH',
        payload: { assignments: values.map((value, trkptIndex) => ({ trkptIndex, value })) },
      })
      const next = routeReducer(state, { type: 'FINALIZE_ELE_FIX' })
      expect(next.routePoints.map((p) => p.eleFix)).toEqual(values)
      expect(next.gradeFix).toEqual({ max: expect.any(Number), min: 0 })
    })
  })

  describe('SET_ELE_CHOICE / FINALIZE_SAVE_ELEVATION（16-2章・保存画面）', () => {
    it('SET_ELE_CHOICEはeleChoiceのみ変更する', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, { type: 'SET_ELE_CHOICE', payload: { choice: 'fix' } })
      expect(next.eleChoice).toBe('fix')
      expect(next.undoSnapshot).toBeNull()
    })

    it('FINALIZE_SAVE_ELEVATIONはele_fix全点・gradeFix・eleChoiceを一括反映する', () => {
      const state = baseFivePointRoute()
      const fixValues = [100, 101, 102, 103, 104]
      const gradeFix = { max: 1, min: 0 }
      const next = routeReducer(state, {
        type: 'FINALIZE_SAVE_ELEVATION',
        payload: { fixValues, gradeFix, choice: 'fix' },
      })
      expect(next.routePoints.map((p) => p.eleFix)).toEqual(fixValues)
      expect(next.gradeFix).toEqual(gradeFix)
      expect(next.eleChoice).toBe('fix')
    })
  })

  describe('DELETE_WPT（15章・Undo対象）', () => {
    it('wptを解除しつつacptとしては残す', () => {
      const state = baseFivePointRoute()
      const next = routeReducer(state, { type: 'DELETE_WPT', payload: { trkptIndex: 0 } })
      expect(next.routePoints[0].wpt).toBeNull()
      expect(next.routePoints[0].isAcpt).toBe(true)
      expect(next.undoSnapshot).toEqual(state.routePoints)
    })
  })
})
