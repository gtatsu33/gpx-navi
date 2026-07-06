'use strict';

function parseGPX(text) {
  const doc = new DOMParser().parseFromString(text, 'text/xml');
  const NS  = 'http://www.topografix.com/GPX/1/1';
  // getElementsByTagNameNS と getElementsByTagName の両方を試し、
  // 多い方（実際に要素が入っている方）を使う。両方使うと重複する。
  const getEls = (p, t) => {
    const withNS = [...p.getElementsByTagNameNS(NS, t)];
    const withoutNS = [...p.getElementsByTagName(t)];
    return withNS.length >= withoutNS.length ? withNS : withoutNS;
  };
  const getText = (p, t) => { const el = getEls(p, t)[0]; return el ? el.textContent.trim() : ''; };

  const trkEl = getEls(doc, 'trk')[0];
  const name = (trkEl && (getText(trkEl, 'name') || getText(trkEl, 'n'))) || 'ルート';

  const track = [];
  let prevKey = null;
  for (const el of getEls(doc, 'trkpt')) {
    const lat = parseFloat(el.getAttribute('lat'));
    const lon = parseFloat(el.getAttribute('lon'));
    if (isNaN(lat) || isNaN(lon)) continue;
    const key = `${lat.toFixed(6)},${lon.toFixed(6)}`;
    if (key === prevKey) continue;
    prevKey = key;
    const ele = parseFloat(getText(el, 'ele'));
    track.push({ lat, lon, ele: isNaN(ele) ? null : ele });
  }

  const turns = getEls(doc, 'wpt').map(el => {
    const bc = parseFloat((getText(el, 'desc') || '').replace('bearing_change:', ''));
    return {
      lat:  parseFloat(el.getAttribute('lat')),
      lon:  parseFloat(el.getAttribute('lon')),
      name: getText(el, 'name') || getText(el, 'n') || '通過点',
      bearingChange: bc,
    };
  }).filter(p => !isNaN(p.lat) && !isNaN(p.lon));

  return { name, track, turns };
}

if (typeof module !== 'undefined') module.exports = { parseGPX };
