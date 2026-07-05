import { describe, expect, it, vi } from 'vitest'
import { fetchElevationsForIndices } from './elevation.js'

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => body }
}

describe('fetchElevationsForIndices（spec.txt 16-2章の保存時再取得）', () => {
  it('指定indexすべてについて{trkptIndex, value}を返す', async () => {
    const points = [
      [35.0, 139.0],
      [35.001, 139.0],
      [35.002, 139.0],
    ]
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ elevation: 100 }))
    const sleep = vi.fn().mockResolvedValue(undefined)
    const assignments = await fetchElevationsForIndices(points, [0, 1, 2], { fetchImpl, sleep, concurrency: 2 })
    expect(assignments).toHaveLength(3)
    expect(assignments.map((a) => a.trkptIndex).sort()).toEqual([0, 1, 2])
    expect(assignments.every((a) => a.value === 100)).toBe(true)
  })

  it('進捗をonProgressで通知する', async () => {
    const points = [[35.0, 139.0], [35.001, 139.0]]
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ elevation: 50 }))
    const sleep = vi.fn().mockResolvedValue(undefined)
    const onProgress = vi.fn()
    await fetchElevationsForIndices(points, [0, 1], { fetchImpl, sleep, onProgress, concurrency: 1 })
    expect(onProgress).toHaveBeenLastCalledWith({ done: 2, total: 2 })
  })
})
