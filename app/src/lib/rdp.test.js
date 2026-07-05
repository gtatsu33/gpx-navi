import { describe, expect, it } from 'vitest'
import { rdpMask } from './rdp.js'
import fixture from '../__fixtures__/rdp.json'

describe('rdp.js (Python版 rdp パッケージとの突き合わせ)', () => {
  it('マスクがPython版と一致する', () => {
    const mask = rdpMask(fixture.input.points, fixture.input.epsilon)
    expect(mask).toEqual(fixture.mask)
  })
})
