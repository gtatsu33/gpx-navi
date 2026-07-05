import { describe, expect, it, vi } from 'vitest'
import {
  buildIntersectionQuery,
  buildPoiQuery,
  fetchIntersectionNames,
  fetchSpotName,
  nearestNameMatch,
  queryOverpassWithFailover,
} from './overpass.js'

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => body }
}

describe('buildIntersectionQuery / buildPoiQuery', () => {
  it('複数ターン候補を1クエリにまとめる（バッチ）', () => {
    const q = buildIntersectionQuery([{ lat: 35, lon: 139, index: 0 }, { lat: 36, lon: 140, index: 1 }], 20, 28)
    expect(q).toContain('[out:json][timeout:28];')
    expect(q).toContain('around:20,35,139')
    expect(q).toContain('around:20,36,140')
  })

  it('POIクエリはタグを複数条件で並べる', () => {
    const q = buildPoiQuery(35, 139, 20, 13)
    expect(q).toContain('"tourism"')
    expect(q).toContain('"shop"')
  })
})

describe('nearestNameMatch', () => {
  it('半径内の最近傍ノードのnameを採用する', () => {
    const turns = [{ lat: 35.0, lon: 139.0, index: 5 }]
    const elements = [
      { lat: 35.0001, lon: 139.0001, tags: { name: '近い交差点' } },
      { lat: 35.01, lon: 139.01, tags: { name: '遠い交差点' } },
    ]
    expect(nearestNameMatch(turns, elements, 20)).toEqual({ 5: '近い交差点' })
  })

  it('半径外なら結果に含めない', () => {
    const turns = [{ lat: 35.0, lon: 139.0, index: 5 }]
    const elements = [{ lat: 36.0, lon: 140.0, tags: { name: '遠い' } }]
    expect(nearestNameMatch(turns, elements, 20)).toEqual({})
  })
})

describe('queryOverpassWithFailover', () => {
  it('最初のミラーが成功すればそれを使う', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ elements: [{ id: 1 }] }))
    const result = await queryOverpassWithFailover('Q', { urls: ['a', 'b', 'c'], fetchImpl })
    expect(result).toEqual([{ id: 1 }])
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })

  it('通常エラーは次のミラーへフェイルオーバーする', async () => {
    const fetchImpl = vi
      .fn()
      .mockRejectedValueOnce(new Error('down'))
      .mockResolvedValueOnce(jsonResponse({ elements: [{ id: 2 }] }))
    const result = await queryOverpassWithFailover('Q', { urls: ['a', 'b'], fetchImpl })
    expect(result).toEqual([{ id: 2 }])
    expect(fetchImpl).toHaveBeenCalledTimes(2)
  })

  it('429は待機してから同一ミラーへ再試行する', async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({}, { ok: false, status: 429 }))
      .mockResolvedValueOnce(jsonResponse({ elements: [{ id: 3 }] }))
    const sleep = vi.fn().mockResolvedValue(undefined)
    const result = await queryOverpassWithFailover('Q', { urls: ['a', 'b'], fetchImpl, sleep })
    expect(result).toEqual([{ id: 3 }])
    expect(fetchImpl).toHaveBeenCalledTimes(2)
    expect(fetchImpl.mock.calls[0][0]).toBe('a')
    expect(fetchImpl.mock.calls[1][0]).toBe('a')
    expect(sleep).toHaveBeenCalledWith(2000)
  })

  it('全ミラー失敗ならnullを返す', async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new Error('down'))
    const result = await queryOverpassWithFailover('Q', { urls: ['a', 'b', 'c'], fetchImpl })
    expect(result).toBeNull()
    expect(fetchImpl).toHaveBeenCalledTimes(3)
  })
})

describe('fetchIntersectionNames / fetchSpotName', () => {
  it('交差点名を取得して{index: name}を返す', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse({ elements: [{ lat: 35.0001, lon: 139.0001, tags: { name: 'テスト交差点' } }] })
    )
    const result = await fetchIntersectionNames([{ lat: 35.0, lon: 139.0, index: 3 }], { fetchImpl })
    expect(result).toEqual({ 3: 'テスト交差点' })
  })

  it('ターンが空なら通信せず{}を返す', async () => {
    const fetchImpl = vi.fn()
    const result = await fetchIntersectionNames([], { fetchImpl })
    expect(result).toEqual({})
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it('POI名フォールバックを取得する', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse({ elements: [{ lat: 35.0001, lon: 139.0001, tags: { name: '公園' } }] })
    )
    const result = await fetchSpotName(35.0, 139.0, { fetchImpl })
    expect(result).toBe('公園')
  })

  it('要素が無ければnullを返す', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ elements: [] }))
    const result = await fetchSpotName(35.0, 139.0, { fetchImpl })
    expect(result).toBeNull()
  })
})
