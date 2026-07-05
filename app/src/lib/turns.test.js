import { describe, expect, it } from 'vitest'
import { detectTurns, turnLabel } from './turns.js'
import fixture from '../__fixtures__/turns.json'

describe('turns.js (Python版 gpxconverter.py との突き合わせ)', () => {
  it('detectTurnsがPython版と一致する', () => {
    const { points, min_turn_angle: minTurnAngle, min_dist: minDist, smooth } = fixture.detectTurns.input
    const result = detectTurns(points, { minTurnAngle, minDist, smooth })
    expect(result.length).toBe(fixture.detectTurns.output.length)
    result.forEach((r, i) => {
      const expected = fixture.detectTurns.output[i]
      expect(r.index).toBe(expected.index)
      expect(r.lat).toBeCloseTo(expected.lat, 9)
      expect(r.lon).toBeCloseTo(expected.lon, 9)
      expect(r.delta).toBeCloseTo(expected.delta, 6)
    })
  })

  it.each(fixture.turnLabel)('turnLabel %#', (c) => {
    expect(turnLabel(c.input)).toEqual(c.output)
  })
})
