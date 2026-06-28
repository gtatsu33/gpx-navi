import { useMemo } from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Line, Path, Text as SvgText } from 'react-native-svg';
import { GradePoint, gradeColor } from '../lib/geoCalc';

const H = 90;
const LABEL_H = 14;
const PH = H - LABEL_H;
const LOOKAHEAD = 1000;

type Props = { grads: GradePoint[]; trkIdx: number; width: number };

export default function GradeChart({ grads, trkIdx, width: W }: Props) {
  const paths = useMemo(() => {
    if (!grads.length) return [];
    const startDist = (grads[trkIdx] ?? grads[0]).cumDist;
    const endDist = startDist + LOOKAHEAD;
    const pts = grads.filter(p => p.cumDist >= startDist && p.cumDist <= endDist);
    if (pts.length < 2) return [];

    const eleMin = Math.min(...pts.map(p => p.ele));
    const eleRange = 50;
    const xOf = (d: number) => ((d - startDist) / LOOKAHEAD) * W;
    const yOf = (e: number) => PH - 2 - ((e - eleMin) / eleRange) * (PH - 20);

    const segments: { d: string; color: string }[] = [];
    let si = 1;
    while (si < pts.length) {
      const color = gradeColor(pts[si].grade);
      let ei = si;
      while (ei < pts.length && gradeColor(pts[ei].grade) === color) ei++;
      const x0 = xOf(pts[si-1].cumDist), y0 = yOf(pts[si-1].ele);
      let d = `M ${x0} ${PH} L ${x0} ${y0}`;
      for (let k = si; k < ei; k++) d += ` L ${xOf(pts[k].cumDist)} ${yOf(pts[k].ele)}`;
      d += ` L ${xOf(pts[ei-1].cumDist)} ${PH} Z`;
      segments.push({ d, color });
      si = ei;
    }

    // 稜線
    const ridgePts = pts.map(p => `${xOf(p.cumDist)},${yOf(p.ele)}`).join(' L ');
    const ridge = `M ${ridgePts}`;

    return { segments, ridge, pts, xOf, yOf, currentGrade: grads[trkIdx]?.grade ?? null };
  }, [grads, trkIdx]);

  if (!paths || !Array.isArray((paths as any).segments)) return null;
  const { segments, ridge, currentGrade } = paths as any;

  return (
    <View style={styles.container}>
      <Svg width={W} height={H}>
        {segments.map((seg: any, i: number) => (
          <Path key={i} d={seg.d} fill={seg.color} opacity={0.85} />
        ))}
        <Path d={ridge} stroke="rgba(255,255,255,0.35)" strokeWidth={1} fill="none" />
        <Line x1={2} y1={0} x2={2} y2={PH} stroke="#fff" strokeWidth={2} />
        <SvgText x={5} y={H - 2} fontSize={10} fill="#999">現在</SvgText>
        <SvgText x={W/2} y={H - 2} fontSize={10} fill="#999" textAnchor="middle">500m</SvgText>
        <SvgText x={W - 2} y={H - 2} fontSize={10} fill="#999" textAnchor="end">1km</SvgText>
        {currentGrade != null && (
          <SvgText x={W - 4} y={16} fontSize={14} fontWeight="bold" fill={gradeColor(currentGrade)} textAnchor="end">
            {`${currentGrade > 0 ? '+' : ''}${currentGrade.toFixed(1)}%`}
          </SvgText>
        )}
      </Svg>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'rgba(0,0,0,0.82)',
    borderRadius: 8,
    overflow: 'hidden',
  },
});
