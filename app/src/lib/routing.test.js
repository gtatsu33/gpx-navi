import { describe, expect, it, vi } from 'vitest'
import { calcRouteSegment } from './routing.js'

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => body }
}

describe('routing.js calcRouteSegment', () => {
  it('正常系: GeoJSON座標を[lat,lon]に変換して返す', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse({
        code: 'Ok',
        routes: [{ geometry: { coordinates: [[139.0, 35.0], [139.001, 35.001]] } }],
      })
    )
    const result = await calcRouteSegment([[35.0, 139.0], [35.001, 139.001]], { fetchImpl })
    expect(result).toEqual([[35.0, 139.0], [35.001, 139.001]])
  })

  it('APIがcode!=="Ok"を返したら入力をそのまま返す（フォールバック）', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ code: 'NoRoute' }))
    const input = [[35.0, 139.0], [35.001, 139.001]]
    const result = await calcRouteSegment(input, { fetchImpl })
    expect(result).toBe(input)
  })

  it('HTTPエラー時は入力をそのまま返す（フォールバック）', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({}, { ok: false, status: 500 }))
    const input = [[35.0, 139.0], [35.001, 139.001]]
    const result = await calcRouteSegment(input, { fetchImpl })
    expect(result).toBe(input)
  })

  it('ネットワークエラー・タイムアウト時は入力をそのまま返す（フォールバック）', async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new Error('network error'))
    const input = [[35.0, 139.0], [35.001, 139.001]]
    const result = await calcRouteSegment(input, { fetchImpl })
    expect(result).toBe(input)
  })
})
