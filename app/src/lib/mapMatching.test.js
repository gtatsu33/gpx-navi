import { describe, expect, it, vi } from 'vitest'
import { applyMatchedPoints, matchChunk, matchRoute } from './mapMatching.js'

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => body }
}

function makePoints(n) {
  return Array.from({ length: n }, (_, i) => [35 + i * 0.0001, 139])
}

describe('mapMatching.js matchChunk', () => {
  it('正常系: レスポンスJSONを返す', async () => {
    const responseBody = { matched_points: [{ lat: 35.0, lon: 139.0, type: 'matched' }] }
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(responseBody))
    const result = await matchChunk([[35.0, 139.0]], { fetchImpl })
    expect(result).toEqual(responseBody)
  })

  it('HTTPエラー時は例外を投げる（チャンク処理側が継続/中断を判断する）', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({}, { ok: false, status: 504 }))
    await expect(matchChunk([[35.0, 139.0]], { fetchImpl })).rejects.toThrow()
  })

  it('タイムアウト（fetch自体のreject）時は例外を投げる', async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new Error('timeout'))
    await expect(matchChunk([[35.0, 139.0]], { fetchImpl })).rejects.toThrow()
  })
})

describe('mapMatching.js applyMatchedPoints', () => {
  it('matched/interpolatedのみ座標を置換し、それ以外は元座標を保持する', () => {
    const original = [[35.0, 139.0], [35.001, 139.0], [35.002, 139.0]]
    const response = {
      matched_points: [
        { lat: 35.0001, lon: 139.0001, type: 'matched' },
        { lat: 35.0011, lon: 139.0011, type: 'unmatched' },
        { lat: 35.0021, lon: 139.0021, type: 'interpolated' },
      ],
    }
    expect(applyMatchedPoints(original, response)).toEqual([
      [35.0001, 139.0001],
      [35.001, 139.0],
      [35.0021, 139.0021],
    ])
  })
})

describe('matchRoute（チャンク処理ロジック）', () => {
  it('正常系: 全チャンクをmatchedに置き換え、完了ステータスを返す', async () => {
    const points = makePoints(120) // 3チャンク
    const matchChunkImpl = vi.fn().mockImplementation(async (chunk) => ({
      matched_points: chunk.map(([lat, lon]) => ({ lat: lat + 0.001, lon, type: 'matched' })),
    }))
    const onProgress = vi.fn()
    const result = await matchRoute(points, { matchChunkImpl, onProgress })

    expect(result.status).toBe('完了')
    expect(result.nSnapped).toBe(120)
    expect(result.matchedPoints[0][0]).toBeCloseTo(35.001, 6)
    expect(matchChunkImpl).toHaveBeenCalledTimes(3)
    expect(onProgress).toHaveBeenLastCalledWith(
      expect.objectContaining({ chunkIdx: 3, totalChunks: 3, status: '完了' })
    )
  })

  it('1チャンク目のタイムアウトは自動的に全体をキャンセル扱いにする', async () => {
    const points = makePoints(60)
    const matchChunkImpl = vi.fn().mockRejectedValue(new Error('timeout'))
    const result = await matchRoute(points, { matchChunkImpl })

    expect(result.status).toBe('キャンセル')
    expect(result.error).toContain('1チャンク目タイムアウト')
    expect(result.matchedPoints).toBe(points)
    expect(matchChunkImpl).toHaveBeenCalledTimes(1)
  })

  it('2チャンク目以降のエラーはそのチャンクの元座標を保持しつつ継続する', async () => {
    const points = makePoints(100) // 2チャンク
    const matchChunkImpl = vi
      .fn()
      .mockResolvedValueOnce({
        matched_points: makePoints(50).map(([lat, lon]) => ({ lat: lat + 0.001, lon, type: 'matched' })),
      })
      .mockRejectedValueOnce(new Error('chunk2 down'))
    const result = await matchRoute(points, { matchChunkImpl })

    expect(result.status).toBe('完了')
    expect(result.nSnapped).toBe(50)
    expect(result.error).toContain('chunk 1')
    expect(result.matchedPoints[60]).toEqual(points[60]) // 2チャンク目は元座標のまま
  })

  it('ユーザーキャンセルはそれまでの部分結果を返す', async () => {
    const points = makePoints(150) // 3チャンク
    let calls = 0
    const matchChunkImpl = vi.fn().mockImplementation(async (chunk) => {
      calls += 1
      return { matched_points: chunk.map(([lat, lon]) => ({ lat: lat + 0.001, lon, type: 'matched' })) }
    })
    const result = await matchRoute(points, {
      matchChunkImpl,
      shouldCancel: () => calls >= 1,
    })

    expect(result.status).toBe('キャンセル')
    expect(result.error).toContain('キャンセルされました')
    expect(matchChunkImpl).toHaveBeenCalledTimes(1)
  })

  it('全点マッチしなければステータスはエラーになる', async () => {
    const points = makePoints(10)
    const matchChunkImpl = vi.fn().mockResolvedValue({
      matched_points: points.map(([lat, lon]) => ({ lat, lon, type: 'unmatched' })),
    })
    const result = await matchRoute(points, { matchChunkImpl })
    expect(result.status).toBe('エラー')
    expect(result.nSnapped).toBe(0)
  })
})
