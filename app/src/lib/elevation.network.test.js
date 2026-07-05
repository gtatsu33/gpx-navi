import { describe, expect, it, vi } from 'vitest'
import { ElevationCircuitBreaker, fetchElevationRaw, fetchElevationWithRetry } from './elevation.js'

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => body }
}

describe('fetchElevationRaw', () => {
  it('正常系: 標高値を返す', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ elevation: 123.4 }))
    expect(await fetchElevationRaw(35, 139, { fetchImpl })).toBe(123.4)
  })

  it.each([[-9999], ['-----'], [null]])('取得不可の値(%s)はnullを返す', async (value) => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ elevation: value }))
    expect(await fetchElevationRaw(35, 139, { fetchImpl })).toBeNull()
  })

  it('HTTPエラー時は例外を投げる', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({}, { ok: false, status: 500 }))
    await expect(fetchElevationRaw(35, 139, { fetchImpl })).rejects.toThrow()
  })
})

describe('fetchElevationWithRetry（指数バックオフ）', () => {
  it('1回目で成功すればsleepを呼ばない', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse({ elevation: 100 }))
    const sleep = vi.fn().mockResolvedValue(undefined)
    const result = await fetchElevationWithRetry(35, 139, { fetchImpl, sleep })
    expect(result).toBe(100)
    expect(fetchImpl).toHaveBeenCalledTimes(1)
    expect(sleep).not.toHaveBeenCalled()
  })

  it('2回失敗し3回目で成功すれば、1秒→2秒の順でsleepする', async () => {
    const fetchImpl = vi
      .fn()
      .mockRejectedValueOnce(new Error('fail1'))
      .mockRejectedValueOnce(new Error('fail2'))
      .mockResolvedValueOnce(jsonResponse({ elevation: 50 }))
    const sleep = vi.fn().mockResolvedValue(undefined)
    const result = await fetchElevationWithRetry(35, 139, { fetchImpl, sleep })
    expect(result).toBe(50)
    expect(fetchImpl).toHaveBeenCalledTimes(3)
    expect(sleep.mock.calls).toEqual([[1000], [2000]])
  })

  it('3回とも失敗したらnull（取得不可）を返し、それ以上再試行しない', async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new Error('always fails'))
    const sleep = vi.fn().mockResolvedValue(undefined)
    const result = await fetchElevationWithRetry(35, 139, { fetchImpl, sleep })
    expect(result).toBeNull()
    expect(fetchImpl).toHaveBeenCalledTimes(3)
  })
})

describe('ElevationCircuitBreaker', () => {
  it('直近10件中5回未満の失敗ではopenにならない', () => {
    const breaker = new ElevationCircuitBreaker()
    for (let i = 0; i < 4; i++) breaker.recordResult(false)
    expect(breaker.isOpen()).toBe(false)
  })

  it('直近10件中5回以上失敗するとopenになる', () => {
    const breaker = new ElevationCircuitBreaker()
    for (let i = 0; i < 5; i++) breaker.recordResult(false)
    expect(breaker.isOpen()).toBe(true)
  })

  it('クールダウン経過後は自動的にcloseし、履歴もリセットされる', () => {
    let now = 0
    const breaker = new ElevationCircuitBreaker({ now: () => now, cooldownMs: 30000 })
    for (let i = 0; i < 5; i++) breaker.recordResult(false)
    expect(breaker.isOpen()).toBe(true)
    now = 30001
    expect(breaker.isOpen()).toBe(false)
    // リセット後は成功扱いの履歴もクリアされているはず
    for (let i = 0; i < 4; i++) breaker.recordResult(false)
    expect(breaker.isOpen()).toBe(false)
  })
})
