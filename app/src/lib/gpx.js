import { turnLabel } from './turns.js'

export const GPX_NS = 'http://www.topografix.com/GPX/1/1'
export const GPXNAVI_NS = 'https://gpxnavi'

function textOf(el) {
  return el && el.textContent ? el.textContent.trim() : ''
}

/**
 * GPX文字列をパースする。spec.txt 5-4章のマッピングに対応。
 * 戻り値: { trkpts, waypoints, hasWpts, acptIndices, trackName }
 */
export function parseGpx(xmlString) {
  const doc = new DOMParser().parseFromString(xmlString, 'application/xml')
  const parserError = doc.getElementsByTagName('parsererror')[0]
  if (parserError) {
    throw new Error(`GPXの解析に失敗しました: ${textOf(parserError)}`)
  }

  const trkptEls = Array.from(doc.getElementsByTagNameNS(GPX_NS, 'trkpt'))
  const trkpts = trkptEls.map((el) => {
    const eleEl = el.getElementsByTagNameNS(GPX_NS, 'ele')[0]
    return {
      lat: parseFloat(el.getAttribute('lat')),
      lon: parseFloat(el.getAttribute('lon')),
      ele: eleEl ? parseFloat(textOf(eleEl)) : null,
    }
  })

  const acptIndices = new Set()
  trkptEls.forEach((el, i) => {
    const ext = el.getElementsByTagNameNS(GPX_NS, 'extensions')[0]
    if (!ext) return
    const acptEl = ext.getElementsByTagNameNS(GPXNAVI_NS, 'acpt')[0]
    if (acptEl && textOf(acptEl) === '1') acptIndices.add(i)
  })

  const wptEls = Array.from(doc.getElementsByTagNameNS(GPX_NS, 'wpt'))
  const waypoints = wptEls.map((el) => ({
    lat: parseFloat(el.getAttribute('lat')),
    lon: parseFloat(el.getAttribute('lon')),
    name: textOf(el.getElementsByTagNameNS(GPX_NS, 'name')[0]),
    desc: textOf(el.getElementsByTagNameNS(GPX_NS, 'desc')[0]),
  }))

  const trkEl = doc.getElementsByTagNameNS(GPX_NS, 'trk')[0]
  const trkNameEl = trkEl?.getElementsByTagNameNS(GPX_NS, 'name')[0]

  // spec.txt 5-4章: trk/extensions/gpxnavi:eleSource="gsi" は、このファイルの
  // 標高値が既に国土地理院データ（スパイク除去済み）であることを示す。
  // 保存時（16-2章）の無駄な再取得を避けるために使う。
  let eleSourceGsi = false
  const trkExtEl = trkEl?.getElementsByTagNameNS(GPX_NS, 'extensions')[0]
  const eleSourceEl = trkExtEl?.getElementsByTagNameNS(GPXNAVI_NS, 'eleSource')[0]
  if (eleSourceEl && textOf(eleSourceEl) === 'gsi') eleSourceGsi = true

  return {
    trkpts,
    waypoints,
    hasWpts: waypoints.length > 0,
    acptIndices,
    trackName: trkNameEl ? textOf(trkNameEl) : null,
    eleSourceGsi,
  }
}

/**
 * GPX文字列をビルドする。spec.txt 16-2章のルールに対応。
 * routePoints: { lat, lon, eleOrg, eleFix, isAcpt, wpt: {name, delta}|null }[]
 */
export function buildGpx({ baseXmlString, routePoints, eleChoice = 'org', routeName } = {}) {
  let doc
  if (baseXmlString) {
    doc = new DOMParser().parseFromString(baseXmlString, 'application/xml')
  } else {
    doc = document.implementation.createDocument(GPX_NS, 'gpx', null)
    doc.documentElement.setAttribute('version', '1.1')
  }
  const gpxEl = doc.documentElement

  Array.from(doc.getElementsByTagNameNS(GPX_NS, 'wpt')).forEach((el) => el.parentNode.removeChild(el))

  let trkEl = doc.getElementsByTagNameNS(GPX_NS, 'trk')[0]
  if (!trkEl) {
    trkEl = doc.createElementNS(GPX_NS, 'trk')
    gpxEl.appendChild(trkEl)
  }
  Array.from(trkEl.getElementsByTagNameNS(GPX_NS, 'trkseg')).forEach((el) => trkEl.removeChild(el))

  // spec.txt 5-4章・16-2章: 出力するele値が国土地理院データ（fix）の場合のみ
  // eleSourceフラグを付与する。org選択時や既存フラグは常に一旦取り除く。
  Array.from(trkEl.getElementsByTagNameNS(GPX_NS, 'extensions')).forEach((el) => trkEl.removeChild(el))
  if (eleChoice === 'fix') {
    const trkExtEl = doc.createElementNS(GPX_NS, 'extensions')
    const eleSourceEl = doc.createElementNS(GPXNAVI_NS, 'gpxnavi:eleSource')
    eleSourceEl.textContent = 'gsi'
    trkExtEl.appendChild(eleSourceEl)
    trkEl.appendChild(trkExtEl)
  }

  const existingNameEl = trkEl.getElementsByTagNameNS(GPX_NS, 'name')[0]
  if (!existingNameEl && routeName) {
    const nameEl = doc.createElementNS(GPX_NS, 'name')
    nameEl.textContent = routeName
    trkEl.insertBefore(nameEl, trkEl.firstChild)
  }

  const trksegEl = doc.createElementNS(GPX_NS, 'trkseg')
  trkEl.appendChild(trksegEl)

  routePoints.forEach((p) => {
    const trkptEl = doc.createElementNS(GPX_NS, 'trkpt')
    trkptEl.setAttribute('lat', String(p.lat))
    trkptEl.setAttribute('lon', String(p.lon))
    const ele = eleChoice === 'fix' ? p.eleFix : p.eleOrg
    if (ele !== null && ele !== undefined) {
      const eleEl = doc.createElementNS(GPX_NS, 'ele')
      eleEl.textContent = String(ele)
      trkptEl.appendChild(eleEl)
    }
    if (p.isAcpt) {
      const extEl = doc.createElementNS(GPX_NS, 'extensions')
      const acptEl = doc.createElementNS(GPXNAVI_NS, 'gpxnavi:acpt')
      acptEl.textContent = '1'
      extEl.appendChild(acptEl)
      trkptEl.appendChild(extEl)
    }
    trksegEl.appendChild(trkptEl)
  })

  routePoints.forEach((p) => {
    if (!p.wpt) return
    const wptEl = doc.createElementNS(GPX_NS, 'wpt')
    wptEl.setAttribute('lat', String(p.lat))
    wptEl.setAttribute('lon', String(p.lon))

    let name = p.wpt.name
    if (!name && p.wpt.delta !== null && p.wpt.delta !== undefined) {
      name = turnLabel(p.wpt.delta)[0]
    }
    const nameEl = doc.createElementNS(GPX_NS, 'name')
    nameEl.textContent = name || ''
    wptEl.appendChild(nameEl)

    const descEl = doc.createElementNS(GPX_NS, 'desc')
    descEl.textContent =
      p.wpt.delta !== null && p.wpt.delta !== undefined
        ? `bearing_change:${p.wpt.delta.toFixed(1)}`
        : 'manually added'
    wptEl.appendChild(descEl)

    gpxEl.appendChild(wptEl)
  })

  return new XMLSerializer().serializeToString(doc)
}
