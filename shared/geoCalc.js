'use strict';

function geoDist(lat1, lon1, lat2, lon2) {
  const R = 6371000, dL = (lat2-lat1)*Math.PI/180, dN = (lon2-lon1)*Math.PI/180;
  const a = Math.sin(dL/2)**2 + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dN/2)**2;
  return R * 2 * Math.asin(Math.sqrt(Math.min(1, a)));
}

const formatDist = m => m >= 1000 ? `${(m/1000).toFixed(1)}km` : `${Math.round(m)}m`;

function latLonToTile(lat, lon, z) {
  const n = 2**z;
  return {
    x: Math.floor((lon+180)/360*n),
    y: Math.floor((1 - Math.log(Math.tan(lat*Math.PI/180) + 1/Math.cos(lat*Math.PI/180))/Math.PI)/2*n)
  };
}

function buildWptTrkMapping(track, turns) {
  return turns.map(wp => {
    let bestIdx = 0, bestD = Infinity;
    for (let i = 0; i < track.length; i++) {
      const d = geoDist(wp.lat, wp.lon, track[i].lat, track[i].lon);
      if (d < bestD) { bestD = d; bestIdx = i; }
    }
    return { ...wp, trkIdx: bestIdx };
  });
}

function buildTrackGrads(track) {
  if (!track || track.length < 2) return [];
  if (!track.some(p => p.ele != null)) return [];

  const cumDists = [0];
  for (let i = 1; i < track.length; i++)
    cumDists.push(cumDists[i-1] + geoDist(track[i-1].lat, track[i-1].lon, track[i].lat, track[i].lon));

  // ① スムージング（窓幅を広げてノイズを除去）
  const smooth = track.map((_, i) => {
    let sum = 0, weightSum = 0;
    const R = 40; // 前後40mを参照
    for (let j = Math.max(0, i - 15); j < Math.min(track.length, i + 15); j++) {
      const d = Math.abs(cumDists[i] - cumDists[j]);
      if (d <= R) {
        const w = 1 - d / R;
        sum += track[j].ele * w;
        weightSum += w;
      }
    }
    return weightSum > 0 ? sum / weightSum : track[i].ele;
  });

  // ② 中央差分による勾配計算（見た目の位置ズレを解消）
  const GRADE_WIN = 20; // 前後20m（計40m）で判定
  return track.map((_, i) => {
    let grade = 0;
    let pIdx = i, nIdx = i;
    while (pIdx > 0 && cumDists[i] - cumDists[pIdx] < GRADE_WIN) pIdx--;
    while (nIdx < track.length - 1 && cumDists[nIdx] - cumDists[i] < GRADE_WIN) nIdx++;

    const dist = cumDists[nIdx] - cumDists[pIdx];
    if (dist > 15) { // 最低15m以上の区間で計算
      grade = ((smooth[nIdx] - smooth[pIdx]) / dist) * 100;
    }
    return {
      cumDist: cumDists[i],
      ele: smooth[i],
      grade: Math.max(-25, Math.min(25, grade))
    };
  });
}

function gradeColor(g) {
  if (g < 3)   return '#2ed573';  // 下り〜ほぼ平坦: 緑
  if (g < 6)   return '#ffd32a';  // やや上り: 黄
  if (g < 9)   return '#ff6348';  // 上り: オレンジ
  if (g < 12)  return '#ff0000';  // 急坂: 赤
  return '#4C2E30';               // 激坂: エンジ
}

if (typeof module !== 'undefined') module.exports = { geoDist, formatDist, latLonToTile, buildWptTrkMapping, buildTrackGrads, gradeColor };
