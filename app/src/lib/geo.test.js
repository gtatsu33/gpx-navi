import { describe, expect, it } from 'vitest'
import { angleDiff, calculateBearing, haversine } from './geo.js'
import fixture from '../__fixtures__/geo.json'

describe('geo.js (Python版 gpxconverter.py との突き合わせ)', () => {
  it.each(fixture.bearingHaversine)('calculateBearing/haversine %#', (c) => {
    const { lat1, lon1, lat2, lon2 } = c.input
    expect(calculateBearing(lat1, lon1, lat2, lon2)).toBeCloseTo(c.bearing, 6)
    expect(haversine(lat1, lon1, lat2, lon2)).toBeCloseTo(c.haversine, 6)
  })

  it.each(fixture.angleDiff)('angleDiff %#', (c) => {
    expect(angleDiff(c.input.a, c.input.b)).toBeCloseTo(c.output, 9)
  })
})
