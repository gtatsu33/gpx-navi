import { describe, expect, it } from 'vitest'
import { cleanElevationSpikes, computeGradeStats } from './elevation.js'
import fixture from '../__fixtures__/elevation.json'

describe('elevation.js (Python版 gpxconverter.py との突き合わせ)', () => {
  it('cleanElevationSpikesがPython版と一致する', () => {
    const { points, elevations } = fixture.input
    const { cleaned, stats } = cleanElevationSpikes(points, elevations)

    cleaned.forEach((v, i) => {
      expect(v).toBeCloseTo(fixture.cleaned[i], 6)
    })
    expect(stats.clusters).toBe(fixture.stats.clusters)
    expect(stats.points).toBe(fixture.stats.points)
    expect(stats.maxGradeBefore).toBeCloseTo(fixture.stats.max_grade_before, 6)
    expect(stats.maxGradeAfter).toBeCloseTo(fixture.stats.max_grade_after, 6)
  })

  it('computeGradeStatsがPython版と一致する', () => {
    const { points } = fixture.input
    const gradeStats = computeGradeStats(points, fixture.cleaned)
    expect(gradeStats.max).toBeCloseTo(fixture.gradeStats.max, 6)
    expect(gradeStats.min).toBeCloseTo(fixture.gradeStats.min, 6)
  })
})
