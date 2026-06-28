import { TrackPoint, RawTurn } from './gpxParser';

export type Turn = RawTurn & { trkIdx: number };

export function geoDist(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371000;
  const dL = (lat2 - lat1) * Math.PI / 180;
  const dN = (lon2 - lon1) * Math.PI / 180;
  const a =
    Math.sin(dL / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dN / 2) ** 2;
  return R * 2 * Math.asin(Math.sqrt(Math.min(1, a)));
}

export function formatDist(m: number): string {
  return m >= 1000 ? `${(m / 1000).toFixed(1)}km` : `${Math.round(m)}m`;
}

export function buildWptTrkMapping(track: TrackPoint[], turns: RawTurn[]): Turn[] {
  return turns.map(wp => {
    let bestIdx = 0, bestD = Infinity;
    for (let i = 0; i < track.length; i++) {
      const d = geoDist(wp.lat, wp.lon, track[i].lat, track[i].lon);
      if (d < bestD) { bestD = d; bestIdx = i; }
    }
    return { ...wp, trkIdx: bestIdx };
  });
}

export function updateTrkIdx(
  track: TrackPoint[],
  currentIdx: number,
  lat: number,
  lon: number,
): number {
  if (!track.length) return 0;
  const distFromCurrent = geoDist(lat, lon, track[currentIdx].lat, track[currentIdx].lon);
  if (distFromCurrent > 200) {
    let bestD = Infinity, bestI = 0;
    for (let i = 0; i < track.length; i++) {
      const d = geoDist(lat, lon, track[i].lat, track[i].lon);
      if (d < bestD) { bestD = d; bestI = i; }
    }
    return Math.max(currentIdx, bestI);
  }
  const s = Math.max(0, currentIdx - 5);
  const e = Math.min(track.length - 1, currentIdx + 60);
  let bestD = Infinity, bestI = currentIdx;
  for (let i = s; i <= e; i++) {
    const d = geoDist(lat, lon, track[i].lat, track[i].lon);
    if (d < bestD) { bestD = d; bestI = i; }
  }
  return Math.max(currentIdx, bestI);
}

export function calcRemainingDist(track: TrackPoint[], fromIdx: number): number {
  let dist = 0;
  for (let i = fromIdx + 1; i < track.length; i++) {
    dist += geoDist(track[i - 1].lat, track[i - 1].lon, track[i].lat, track[i].lon);
  }
  return dist;
}

export function calcBearing(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const y = Math.sin(dLon) * Math.cos(lat2 * Math.PI / 180);
  const x =
    Math.cos(lat1 * Math.PI / 180) * Math.sin(lat2 * Math.PI / 180) -
    Math.sin(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.cos(dLon);
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

export type GradePoint = { cumDist: number; ele: number; grade: number };

export function buildTrackGrads(track: TrackPoint[]): GradePoint[] {
  if (track.length < 2 || !track.some(p => p.ele != null)) return [];
  const cumDists = [0];
  for (let i = 1; i < track.length; i++)
    cumDists.push(cumDists[i-1] + geoDist(track[i-1].lat, track[i-1].lon, track[i].lat, track[i].lon));

  const smooth = track.map((_, i) => {
    let sum = 0, w = 0;
    for (let j = Math.max(0, i-15); j < Math.min(track.length, i+15); j++) {
      const d = Math.abs(cumDists[i] - cumDists[j]);
      if (d <= 40) { const wt = 1 - d/40; sum += (track[j].ele ?? 0) * wt; w += wt; }
    }
    return w > 0 ? sum / w : (track[i].ele ?? 0);
  });

  return track.map((_, i) => {
    let pIdx = i, nIdx = i;
    while (pIdx > 0 && cumDists[i] - cumDists[pIdx] < 20) pIdx--;
    while (nIdx < track.length-1 && cumDists[nIdx] - cumDists[i] < 20) nIdx++;
    const dist = cumDists[nIdx] - cumDists[pIdx];
    const grade = dist > 15 ? ((smooth[nIdx] - smooth[pIdx]) / dist) * 100 : 0;
    return { cumDist: cumDists[i], ele: smooth[i], grade: Math.max(-25, Math.min(25, grade)) };
  });
}

export function gradeColor(g: number): string {
  if (g < 3)  return '#2ed573';
  if (g < 6)  return '#ffd32a';
  if (g < 9)  return '#ff6348';
  if (g < 12) return '#ff0000';
  return '#4C2E30';
}

export function reverseTurnName(name: string): string {
  if (name.startsWith('「') && name.endsWith('」')) return name;
  if (name.includes('やや右')) return name.replace('やや右', 'やや左');
  if (name.includes('やや左')) return name.replace('やや左', 'やや右');
  if (name.includes('右折')) return name.replace('右折', '左折');
  if (name.includes('左折')) return name.replace('左折', '右折');
  if (name.includes('右') || name.includes('左'))
    return name.replace('右', '§').replace('左', '右').replace('§', '左');
  return name;
}

export function getTurnArrow(bc: number): string {
  if (bc >  45) return '⇒';
  if (bc >  15) return '↗';
  if (bc < -45) return '⇐';
  if (bc < -15) return '↖';
  return '⬆';
}
