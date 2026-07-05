import { describe, expect, it } from 'vitest'
import { contiguousRanges, deepCopyRoutePoints, makeRoutePoint, nextBoundary, prevBoundary } from './routePoints.js'

describe('routePoints.js', () => {
  it('makeRoutePointがデフォルト値を持つ', () => {
    expect(makeRoutePoint(35, 139)).toEqual({
      lat: 35,
      lon: 139,
      eleOrg: null,
      eleFix: null,
      isAcpt: false,
      wpt: null,
      changed: true,
    })
  })

  it('deepCopyRoutePointsがwptも含めて独立コピーする', () => {
    const rp = [makeRoutePoint(35, 139, { wpt: { name: 'a', delta: 1 } })]
    const copy = deepCopyRoutePoints(rp)
    copy[0].wpt.name = 'changed'
    expect(rp[0].wpt.name).toBe('a')
  })

  it('prevBoundary/nextBoundaryがacptまたはwptの境界を返す', () => {
    const rp = [
      makeRoutePoint(0, 0, { isAcpt: true, wpt: { name: 'スタート', delta: null } }),
      makeRoutePoint(0, 1),
      makeRoutePoint(0, 2, { isAcpt: true }),
      makeRoutePoint(0, 3),
      makeRoutePoint(0, 4, { isAcpt: true, wpt: { name: '目的地', delta: null } }),
    ]
    expect(prevBoundary(3, rp)).toBe(2)
    expect(nextBoundary(1, rp)).toBe(2)
    expect(prevBoundary(1, rp)).toBe(0)
    expect(nextBoundary(3, rp)).toBe(4)
  })

  it('contiguousRangesが連続インデックスをまとめる', () => {
    expect(contiguousRanges([1, 2, 3, 5, 6, 9])).toEqual([
      [1, 3],
      [5, 6],
      [9, 9],
    ])
    expect(contiguousRanges([])).toEqual([])
  })
})
