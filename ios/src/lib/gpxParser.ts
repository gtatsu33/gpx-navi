import { XMLParser } from 'fast-xml-parser';

export type TrackPoint = { lat: number; lon: number; ele: number | null };
export type RawTurn = { lat: number; lon: number; name: string; bearingChange: number };
export type ParsedGPX = { name: string; track: TrackPoint[]; turns: RawTurn[] };

function toArr<T>(v: T | T[] | undefined): T[] {
  if (v == null) return [];
  return Array.isArray(v) ? v : [v];
}

export function parseGPX(gpxText: string): ParsedGPX {
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: '',
    // strip namespace prefixes so <gpx:trk> and <trk> both become "trk"
    transformTagName: (tag) => tag.split(':').pop() ?? tag,
    textNodeName: '#text',
  });

  const doc = parser.parse(gpxText);
  const gpx: any = doc.gpx ?? Object.values(doc)[0] ?? {};

  const name: string =
    gpx.metadata?.name ??
    toArr(gpx.trk)[0]?.name ??
    gpx.name ??
    'ルート';

  const track: TrackPoint[] = [];
  let prevKey: string | null = null;

  for (const trk of toArr(gpx.trk)) {
    for (const seg of toArr(trk.trkseg)) {
      for (const pt of toArr<any>(seg.trkpt)) {
        const lat = parseFloat(pt.lat);
        const lon = parseFloat(pt.lon);
        if (isNaN(lat) || isNaN(lon)) continue;
        const key = `${lat.toFixed(6)},${lon.toFixed(6)}`;
        if (key === prevKey) continue;
        prevKey = key;
        const ele = parseFloat(pt.ele ?? pt['#text']);
        track.push({ lat, lon, ele: isNaN(ele) ? null : ele });
      }
    }
  }

  const turns: RawTurn[] = toArr<any>(gpx.wpt)
    .map((wpt: any) => {
      const lat = parseFloat(wpt.lat);
      const lon = parseFloat(wpt.lon);
      if (isNaN(lat) || isNaN(lon)) return null;
      const desc = String(wpt.desc ?? '');
      const bc = parseFloat(desc.replace('bearing_change:', ''));
      return {
        lat,
        lon,
        name: String(wpt.name ?? wpt.n ?? '通過点'),
        bearingChange: isNaN(bc) ? 0 : bc,
      };
    })
    .filter(Boolean) as RawTurn[];

  return { name, track, turns };
}
