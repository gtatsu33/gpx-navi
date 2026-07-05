import { describe, expect, it } from 'vitest'
import { buildGpx, GPXNAVI_NS, GPX_NS, parseGpx } from './gpx.js'

const SAMPLE_GPX = `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="${GPX_NS}" xmlns:gpxnavi="${GPXNAVI_NS}">
  <trk>
    <name>テストルート</name>
    <trkseg>
      <trkpt lat="35.000000" lon="139.000000"><ele>10</ele>
        <extensions><gpxnavi:acpt>1</gpxnavi:acpt></extensions>
      </trkpt>
      <trkpt lat="35.001000" lon="139.000000"><ele>12</ele></trkpt>
      <trkpt lat="35.002000" lon="139.000000"><ele>14</ele>
        <extensions><gpxnavi:acpt>1</gpxnavi:acpt></extensions>
      </trkpt>
    </trkseg>
  </trk>
  <wpt lat="35.000000" lon="139.000000">
    <name>スタート</name>
    <desc>manually added</desc>
  </wpt>
</gpx>`

describe('gpx.js', () => {
  it('parseGpxがtrkpt/wpt/acpt拡張タグを正しく読み取る', () => {
    const result = parseGpx(SAMPLE_GPX)
    expect(result.trkpts).toEqual([
      { lat: 35.0, lon: 139.0, ele: 10 },
      { lat: 35.001, lon: 139.0, ele: 12 },
      { lat: 35.002, lon: 139.0, ele: 14 },
    ])
    expect(result.acptIndices).toEqual(new Set([0, 2]))
    expect(result.hasWpts).toBe(true)
    expect(result.waypoints[0].name).toBe('スタート')
    expect(result.trackName).toBe('テストルート')
  })

  it('buildGpxがtrkpt完全置換・wpt再構築・acpt拡張タグ付与を行う', () => {
    const routePoints = [
      { lat: 35.0, lon: 139.0, eleOrg: 10, eleFix: 11, isAcpt: true, wpt: { name: 'スタート', delta: null } },
      { lat: 35.001, lon: 139.0, eleOrg: 12, eleFix: 13, isAcpt: false, wpt: null },
      {
        lat: 35.002,
        lon: 139.0,
        eleOrg: 14,
        eleFix: 15,
        isAcpt: true,
        wpt: { name: '', delta: 90 },
      },
    ]
    const xml = buildGpx({ baseXmlString: null, routePoints, eleChoice: 'org', routeName: 'new_route' })
    const reparsed = parseGpx(xml)

    expect(reparsed.trkpts.map((p) => p.ele)).toEqual([10, 12, 14])
    expect(reparsed.acptIndices).toEqual(new Set([0, 2]))
    expect(reparsed.waypoints).toHaveLength(2)
    expect(reparsed.waypoints[1].name).toBe('右折')
    expect(reparsed.waypoints[1].desc).toBe('bearing_change:90.0')
  })

  it('parseGpxがtrk/extensions/gpxnavi:eleSource="gsi"を読み取る', () => {
    const gsiGpx = `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="${GPX_NS}" xmlns:gpxnavi="${GPXNAVI_NS}">
  <trk>
    <extensions><gpxnavi:eleSource>gsi</gpxnavi:eleSource></extensions>
    <trkseg>
      <trkpt lat="35.000000" lon="139.000000"><ele>10</ele></trkpt>
      <trkpt lat="35.001000" lon="139.000000"><ele>12</ele></trkpt>
    </trkseg>
  </trk>
</gpx>`
    expect(parseGpx(gsiGpx).eleSourceGsi).toBe(true)
    expect(parseGpx(SAMPLE_GPX).eleSourceGsi).toBe(false)
  })

  it('buildGpxはeleChoice="fix"の場合のみeleSource=gsiを付与する', () => {
    const routePoints = [
      { lat: 35.0, lon: 139.0, eleOrg: 10, eleFix: 11, isAcpt: true, wpt: null },
      { lat: 35.001, lon: 139.0, eleOrg: 12, eleFix: 13, isAcpt: false, wpt: null },
    ]
    const fixXml = buildGpx({ baseXmlString: null, routePoints, eleChoice: 'fix', routeName: 'r' })
    expect(parseGpx(fixXml).eleSourceGsi).toBe(true)

    const orgXml = buildGpx({ baseXmlString: null, routePoints, eleChoice: 'org', routeName: 'r' })
    expect(parseGpx(orgXml).eleSourceGsi).toBe(false)
  })

  it('buildGpxはorg選択時に既存のeleSourceフラグを引き継がない', () => {
    const gsiGpx = `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="${GPX_NS}" xmlns:gpxnavi="${GPXNAVI_NS}">
  <trk>
    <extensions><gpxnavi:eleSource>gsi</gpxnavi:eleSource></extensions>
    <trkseg>
      <trkpt lat="35.000000" lon="139.000000"><ele>10</ele></trkpt>
      <trkpt lat="35.001000" lon="139.000000"><ele>12</ele></trkpt>
    </trkseg>
  </trk>
</gpx>`
    const routePoints = [
      { lat: 35.0, lon: 139.0, eleOrg: 10, eleFix: 10, isAcpt: true, wpt: null },
      { lat: 35.001, lon: 139.0, eleOrg: 12, eleFix: 12, isAcpt: false, wpt: null },
    ]
    const xml = buildGpx({ baseXmlString: gsiGpx, routePoints, eleChoice: 'org', routeName: 'r' })
    expect(parseGpx(xml).eleSourceGsi).toBe(false)
  })
})
