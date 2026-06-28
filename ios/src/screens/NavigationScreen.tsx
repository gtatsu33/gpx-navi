import { useEffect, useMemo, useRef, useState } from 'react';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import MapView, { Circle, Marker, Polyline, Region } from 'react-native-maps';
import Svg, { Polygon } from 'react-native-svg';
import * as Location from 'expo-location';
import * as Speech from 'expo-speech';
import { useKeepAwake } from 'expo-keep-awake';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../App';
import { getAllRoutes, Route } from '../storage/routeStorage';
import {
  buildTrackGrads,
  calcBearing,
  formatDist,
  GradePoint,
  geoDist,
  getTurnArrow,
  reverseTurnName,
  updateTrkIdx,
} from '../lib/geoCalc';
import GradeChart from '../components/GradeChart';

type Props = NativeStackScreenProps<RootStackParamList, 'Navigation'>;
type Position = { latitude: number; longitude: number; heading: number };
type Phase = { trigger: '300' | '100' | '25'; text: string };
type Waypoint = {
  lat: number; lon: number; name: string;
  bearingChange: number; isGoal: boolean; trkIdx: number;
};

// オリジナルと同じ定数
const NEAR_DIST = 25;
const DEPART_M  = 5;

function formatTime(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function getTurnColor(bc: number): string {
  if (bc >  45) return '#ff4757';
  if (bc >  15) return '#ffa502';
  if (bc < -45) return '#1e90ff';
  if (bc < -15) return '#ffa502';
  return '#2ed573';
}

function formatDistVoice(m: number): string {
  return m >= 1000
    ? `${(Math.round(m / 100) / 10).toFixed(1)}キロメートル`
    : `${Math.round(m / 50) * 50}メートル`;
}

export default function NavigationScreen({ route, navigation }: Props) {
  const { routeId, isReverse = false } = route.params;
  const mapRef = useRef<MapView>(null);
  useKeepAwake();

  const [routeData, setRouteData]     = useState<Route | null>(null);
  const [position, setPosition]       = useState<Position | null>(null);
  const [trkIdx, setTrkIdx]           = useState(0);
  const [isFollowing, setIsFollowing] = useState(true);
  const [elapsed, setElapsed]         = useState(0);
  const [simulating, setSimulating]   = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [gradeEnabled, setGradeEnabled] = useState(false);
  const [grads, setGrads]             = useState<GradePoint[]>([]);

  // ── ナビゲーション状態（表示用 state）──────────────────
  const [navTarget, setNavTarget]       = useState<{ wp: Waypoint; dist: number } | null>(null);
  const [navAfterNext, setNavAfterNext] = useState<Waypoint | null>(null);
  const [remainingDist, setRemainingDist] = useState(0);
  const [gradeWidth, setGradeWidth]       = useState(0);
  const [goalReached, setGoalReached]     = useState(false);

  // ── ウェイポイントナビ（オリジナルと同じ refs）─────────
  const waypointsRef      = useRef<Waypoint[]>([]);
  const targetIdxRef      = useRef(0);
  const nearTargetRef     = useRef(false);
  const minDistTargetRef  = useRef(Infinity);
  const prevDistNextRef   = useRef(Infinity);
  const phasesBuiltRef    = useRef(false);

  // ── 音声系 refs ────────────────────────────────────────
  const voiceEnabledRef   = useRef(true);
  const phasesRef         = useRef<Phase[]>([]);
  const goalAnnouncedRef  = useRef(false);
  const startAnnouncedRef = useRef(false);
  const jaVoiceIdRef      = useRef<string>('com.apple.voice.enhanced.ja-JP.Kyoko');

  // ── その他 refs ────────────────────────────────────────
  const [simRegion, setSimRegion]       = useState<Region | undefined>(undefined);
  const startTimeRef    = useRef(Date.now());
  const trkIdxRef       = useRef(0);
  const routeDataRef    = useRef<Route | null>(null);
  const isFollowingRef  = useRef(true);
  const simIntervalRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const simPtIdxRef     = useRef(0);
  const simulatingRef   = useRef(false);
  const positionRef         = useRef<Position | null>(null);
  const trackCumDistsRef    = useRef<number[]>([]);

  // ── 音声 ───────────────────────────────────────────────
  function speak(text: string) {
    if (!voiceEnabledRef.current) return;
    Speech.speak(text, { voice: jaVoiceIdRef.current });
  }

  useEffect(() => {
    Speech.getAvailableVoicesAsync().then(voices => {
      const ja = voices.filter(v => v.language.startsWith('ja'));
      const selected = ja.find(v => v.identifier === 'com.apple.voice.enhanced.ja-JP.Kyoko')
        ?? ja.find(v => v.quality === Speech.VoiceQuality.Enhanced)
        ?? ja[0];
      if (selected) jaVoiceIdRef.current = selected.identifier;
    });
  }, []);

  function buildPhases(d0: number, name: string, isGoal: boolean = false) {
    const label    = name || (isGoal ? '目的地' : '交差点');
    const closeSuffix = /やや[右左]/.test(label) ? 'に曲がってください' : 'してください';
    const closeTxt = isGoal ? `${label}です` : `${label}${closeSuffix}`;
    const nearTxt  = `まもなく${label}です`;
    let farText: string;
    let phases: Phase[];
    if (d0 > 300) {
      farText = `約${formatDistVoice(d0)}先、${label}です`;
      phases  = [
        { trigger: '300', text: `約300メートル先、${label}です` },
        { trigger: '100', text: nearTxt },
        { trigger: '25',  text: closeTxt },
      ];
    } else if (d0 > 100) {
      farText = `約${formatDistVoice(d0)}先、${label}です`;
      phases  = [{ trigger: '100', text: nearTxt }, { trigger: '25', text: closeTxt }];
    } else if (d0 > 25) {
      farText = nearTxt;
      phases  = [{ trigger: '25', text: closeTxt }];
    } else {
      farText = closeTxt;
      phases  = [];
    }
    phasesRef.current = phases;
    setTimeout(() => speak(farText), 2000);
  }

  function checkPhases(dist: number) {
    const phases = phasesRef.current;
    if (!phases.length) return;
    const phase = phases[0];
    const fire =
      (phase.trigger === '300' && dist <= 300) ||
      (phase.trigger === '100' && dist <= 100) ||
      (phase.trigger === '25'  && dist <= 25);
    if (fire) {
      speak(phase.text);
      phasesRef.current = phases.slice(1);
      checkPhases(dist);
    }
  }

  function toggleVoice() {
    const next = !voiceEnabledRef.current;
    voiceEnabledRef.current = next;
    setVoiceEnabled(next);
    if (next) speak('音声案内を開始します');
    else Speech.stop();
  }

  // ── ズーム操作 ─────────────────────────────────────────
  async function zoomIn() {
    const cam = await mapRef.current?.getCamera();
    if (!cam) return;
    // iOS: altitude ベース（低いほど拡大）、Android: zoom ベース
    if (cam.altitude != null) mapRef.current?.animateCamera({ altitude: cam.altitude / 2 }, { duration: 300 });
    else if (cam.zoom != null) mapRef.current?.animateCamera({ zoom: cam.zoom + 1 }, { duration: 300 });
  }
  async function zoomOut() {
    const cam = await mapRef.current?.getCamera();
    if (!cam) return;
    if (cam.altitude != null) mapRef.current?.animateCamera({ altitude: cam.altitude * 2 }, { duration: 300 });
    else if (cam.zoom != null) mapRef.current?.animateCamera({ zoom: cam.zoom - 1 }, { duration: 300 });
  }

  // ── ウェイポイントナビ（オリジナルのロジック移植）──────

  function updateNavDisplay(lat: number, lon: number) {
    const wps = waypointsRef.current;
    const idx = targetIdxRef.current;
    if (idx >= wps.length) { setNavTarget(null); setNavAfterNext(null); return; }
    const wp   = wps[idx];
    const dist = geoDist(lat, lon, wp.lat, wp.lon);
    setNavTarget({ wp, dist });
    setNavAfterNext(wps[idx + 1] ?? null);

    // オリジナルの updateInfoOverlay と同じ式
    const cum = trackCumDistsRef.current;
    if (cum.length > 1) {
      const routeFromWp = wp.isGoal
        ? 0
        : Math.max(0, cum[cum.length - 1] - (cum[wp.trkIdx] ?? 0));
      setRemainingDist(Math.max(0, dist + routeFromWp));
    }
  }

  // オリジナルの checkWaypointAdvance をそのまま移植
  function checkWaypointAdvance(lat: number, lon: number) {
    const wps = waypointsRef.current;
    let changed = false;

    while (targetIdxRef.current < wps.length) {
      const wp   = wps[targetIdxRef.current];
      const dist = geoDist(lat, lon, wp.lat, wp.lon);
      const next = wps[targetIdxRef.current + 1];

      // ゴールは 30m 以内で確定
      if (wp.isGoal) {
        if (dist < 30) {
          if (!goalAnnouncedRef.current) {
            goalAnnouncedRef.current = true;
            phasesRef.current = [];
            setGoalReached(true);
            speak(`${wp.name}です。お疲れ様でした！`);
          }
          targetIdxRef.current++;
          changed = true;
        }
        break;
      }

      // ① 最近接距離更新・接近フラグ
      if (dist < minDistTargetRef.current) minDistTargetRef.current = dist;
      if (dist < NEAR_DIST) nearTargetRef.current = true;

      if (!nearTargetRef.current) {
        if (next && !next.isGoal) prevDistNextRef.current = geoDist(lat, lon, next.lat, next.lon);
        break;
      }

      // ② 遠ざかりチェック
      if (dist <= minDistTargetRef.current + DEPART_M) {
        if (next && !next.isGoal) prevDistNextRef.current = geoDist(lat, lon, next.lat, next.lon);
        break;
      }

      // ③ 次wpt への接近チェック（次がゴールなら省略）
      if (next && !next.isGoal) {
        const dn = geoDist(lat, lon, next.lat, next.lon);
        if (dn > prevDistNextRef.current + 5) { prevDistNextRef.current = dn; break; }
        prevDistNextRef.current = dn;
      }

      // ①②③ 全クリア → 通過確定
      targetIdxRef.current++;
      nearTargetRef.current    = false;
      minDistTargetRef.current = Infinity;
      prevDistNextRef.current  = Infinity;
      changed = true;
    }

    return changed;
  }

  // GPS 更新ごとにまとめて実行する処理
  function onPositionUpdate(lat: number, lon: number, _heading: number) {
    // trkIdx 更新（勾配マップ用）
    const rd = routeDataRef.current;
    if (rd?.track?.length) {
      const newIdx = updateTrkIdx(rd.track, trkIdxRef.current, lat, lon);
      setTrkIdx(newIdx);
      trkIdxRef.current = newIdx;
    }

    const wps = waypointsRef.current;
    if (!wps.length) return;

    // 初回 GPS 取得でフェーズ構築（オリジナルの phasesBuilt フラグに相当）
    if (!phasesBuiltRef.current) {
      phasesBuiltRef.current = true;
      const wp = wps[targetIdxRef.current];
      if (wp) buildPhases(geoDist(lat, lon, wp.lat, wp.lon), wp.name, wp.isGoal);
    }

    // 音声フェーズチェック（現在ターゲットへの距離で判定）
    const curWp = wps[targetIdxRef.current];
    if (curWp) checkPhases(geoDist(lat, lon, curWp.lat, curWp.lon));

    // ウェイポイント通過判定
    const prevIdx = targetIdxRef.current;
    checkWaypointAdvance(lat, lon);

    // ターゲットが変わったら新フェーズ構築
    if (targetIdxRef.current !== prevIdx) {
      const newWp = wps[targetIdxRef.current];
      if (newWp) buildPhases(geoDist(lat, lon, newWp.lat, newWp.lon), newWp.name, newWp.isGoal);
    }

    updateNavDisplay(lat, lon);

    if (isFollowingRef.current) {
      mapRef.current?.animateCamera(
        { center: { latitude: lat, longitude: lon }, zoom: 17 },
        { duration: 500 },
      );
    }
  }

  // ── シミュレーション ────────────────────────────────────
  function stopSimulation() {
    if (simIntervalRef.current) { clearInterval(simIntervalRef.current); simIntervalRef.current = null; }
    simulatingRef.current = false;
    setSimulating(false);
    setSimRegion(undefined);
  }

  function startSimulation() {
    const track = routeDataRef.current?.track;
    if (!track?.length) return;
    stopSimulation();
    simulatingRef.current = true;
    simPtIdxRef.current   = 0;
    trkIdxRef.current     = 0;
    targetIdxRef.current  = 0;
    nearTargetRef.current    = false;
    minDistTargetRef.current = Infinity;
    prevDistNextRef.current  = Infinity;
    phasesBuiltRef.current   = false;
    setTrkIdx(0);
    setSimulating(true);
    setIsFollowing(true);
    isFollowingRef.current = true;

    const first    = track[0];
    const heading0 = track.length > 1 ? calcBearing(first.lat, first.lon, track[1].lat, track[1].lon) : 0;
    setPosition({ latitude: first.lat, longitude: first.lon, heading: heading0 });
    setSimRegion({ latitude: first.lat, longitude: first.lon, latitudeDelta: 0.004, longitudeDelta: 0.004 });

    simIntervalRef.current = setInterval(() => {
      const t = routeDataRef.current?.track;
      if (!t) return;
      const i = simPtIdxRef.current;
      if (i >= t.length - 1) { stopSimulation(); return; }
      const pt      = t[i];
      const nextPt  = t[i + 1];
      const heading = calcBearing(pt.lat, pt.lon, nextPt.lat, nextPt.lon);
      setPosition({ latitude: pt.lat, longitude: pt.lon, heading });
      if (isFollowingRef.current)
        setSimRegion({ latitude: pt.lat, longitude: pt.lon, latitudeDelta: 0.004, longitudeDelta: 0.004 });
      onPositionUpdate(pt.lat, pt.lon, heading);
      simPtIdxRef.current = i + 1;
    }, 600);
  }

  // ── refs 同期 ───────────────────────────────────────────
  useEffect(() => { routeDataRef.current = routeData; }, [routeData]);
  useEffect(() => { trkIdxRef.current = trkIdx; }, [trkIdx]);
  useEffect(() => { isFollowingRef.current = isFollowing; }, [isFollowing]);
  useEffect(() => { positionRef.current = position; }, [position]);

  // ── ルートロード ────────────────────────────────────────
  useEffect(() => {
    getAllRoutes().then(routes => {
      const r = routes.find(r => r.id === routeId) ?? null;
      if (!r || !isReverse || !r.track || !r.turns) { setRouteData(r); return; }
      const n = r.track.length;
      setRouteData({
        ...r,
        track: [...r.track].reverse(),
        turns: [...r.turns].reverse().map(t => ({
          ...t,
          trkIdx: n - 1 - t.trkIdx,
          name: reverseTurnName(t.name),
          bearingChange: isNaN(t.bearingChange) ? t.bearingChange : -t.bearingChange,
        })),
      });
    });
  }, [routeId, isReverse]);

  // ルートロード時にウェイポイント配列を構築（オリジナルの startNavigation に相当）
  useEffect(() => {
    if (!routeData?.turns || routeData.turns.length < 2) return;
    const navTurns = routeData.turns.slice(1, -1);
    const goalWpt  = routeData.turns[routeData.turns.length - 1];
    waypointsRef.current = [
      ...navTurns.map(t => ({ lat: t.lat, lon: t.lon, name: t.name, bearingChange: t.bearingChange, isGoal: false, trkIdx: t.trkIdx })),
      { lat: goalWpt.lat, lon: goalWpt.lon, name: goalWpt.name, bearingChange: 0, isGoal: true, trkIdx: goalWpt.trkIdx },
    ];

    // オリジナルの startNavigation と同じ累積距離を事前計算
    if (routeData.track?.length) {
      const cum: number[] = [0];
      for (let i = 1; i < routeData.track.length; i++)
        cum.push(cum[i - 1] + geoDist(
          routeData.track[i - 1].lat, routeData.track[i - 1].lon,
          routeData.track[i].lat,     routeData.track[i].lon,
        ));
      trackCumDistsRef.current = cum;
      setRemainingDist(cum[cum.length - 1]);  // GPS取得前の初期値 = 全ルート距離
    }

    targetIdxRef.current     = 0;
    nearTargetRef.current    = false;
    minDistTargetRef.current = Infinity;
    prevDistNextRef.current  = Infinity;
    phasesBuiltRef.current   = false;
    goalAnnouncedRef.current = false;
    setGoalReached(false);

    // 既に GPS 取得済みなら即表示更新
    const pos = positionRef.current;
    if (pos) updateNavDisplay(pos.latitude, pos.longitude);
  }, [routeData]);

  // 標高プロファイル計算
  useEffect(() => {
    if (routeData?.track) setGrads(buildTrackGrads(routeData.track));
  }, [routeData]);

  // 地図をルートに合わせる
  useEffect(() => {
    if (!routeData?.track?.length) return;
    const coords = routeData.track.map(p => ({ latitude: p.lat, longitude: p.lon }));
    mapRef.current?.fitToCoordinates(coords, {
      edgePadding: { top: 60, right: 40, bottom: 320, left: 40 },
      animated: true,
    });
  }, [routeData]);

  // ナビ開始アナウンス
  useEffect(() => {
    if (!routeData || startAnnouncedRef.current) return;
    startAnnouncedRef.current = true;
    speak('ナビを開始します');
  }, [routeData]);

  // ── GPS 追跡 ────────────────────────────────────────────
  useEffect(() => {
    let sub: Location.LocationSubscription | null = null;
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') return;
      sub = await Location.watchPositionAsync(
        { accuracy: Location.Accuracy.BestForNavigation, timeInterval: 1000, distanceInterval: 5 },
        loc => {
          if (simulatingRef.current) return;
          const { latitude, longitude, heading } = loc.coords;
          setPosition({ latitude, longitude, heading: heading ?? 0 });
          onPositionUpdate(latitude, longitude, heading ?? 0);
        },
      );
    })();
    return () => { sub?.remove(); };
  }, []);

  useEffect(() => {
    return () => {
      if (simIntervalRef.current) clearInterval(simIntervalRef.current);
      simulatingRef.current = false;
    };
  }, []);

  // 経過時間タイマー
  useEffect(() => {
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // ── 派生値 ──────────────────────────────────────────────
  const coordinates = useMemo(
    () => (routeData?.track ?? []).map(p => ({ latitude: p.lat, longitude: p.lon })),
    [routeData],
  );



  const startCoord = coordinates[0];
  const endCoord   = coordinates[coordinates.length - 1];

  // ── 復帰ボタン（オリジナルの rerouteToNearest 移植）─────
  function rerouteToNearest() {
    const wps   = waypointsRef.current;
    const track = routeDataRef.current?.track;
    const pos   = positionRef.current;
    if (!wps.length || !track || !pos) return;

    // ① 直前に通過したwptのtrkIdx+1以降で最近傍trkptを求める
    const prevTrkIdx = targetIdxRef.current > 0
      ? (wps[targetIdxRef.current - 1].trkIdx + 1) : 0;
    let nearestTrkIdx = prevTrkIdx, bestD = Infinity;
    for (let i = prevTrkIdx; i < track.length; i++) {
      const d = geoDist(pos.latitude, pos.longitude, track[i].lat, track[i].lon);
      if (d < bestD) { bestD = d; nearestTrkIdx = i; }
    }

    // ② nearestTrkIdx 以降で最初の wpt を選ぶ
    let bestWpIdx = wps.length - 1;
    for (let wi = 0; wi < wps.length; wi++) {
      if (wps[wi].isGoal) break;
      if (wps[wi].trkIdx >= nearestTrkIdx) { bestWpIdx = wi; break; }
    }

    targetIdxRef.current     = bestWpIdx;
    nearTargetRef.current    = false;
    minDistTargetRef.current = Infinity;
    prevDistNextRef.current  = Infinity;
    phasesBuiltRef.current   = false;
    updateNavDisplay(pos.latitude, pos.longitude);
  }

  // ── レンダリング ─────────────────────────────────────────
  return (
    <View style={styles.container}>
      {/* Map area wrapper — overlays position: absolute are relative to this View */}
      <View style={{ flex: 1 }}>
      <MapView
        ref={mapRef}
        style={styles.map}
        region={simRegion}
        showsUserLocation={false}
        showsCompass
        onPanDrag={() => { setIsFollowing(false); isFollowingRef.current = false; }}
      >
        {coordinates.length > 0 && (
          <Polyline coordinates={coordinates} strokeColor="#4A9EFF" strokeWidth={4} />
        )}
        {/* ターンポイント（スタート・ゴールを除く中間） */}
        {routeData?.turns?.slice(1, -1).map((t, i) => (
          <Circle
            key={i}
            center={{ latitude: t.lat, longitude: t.lon }}
            radius={8}
            strokeColor={getTurnColor(t.bearingChange)}
            fillColor={getTurnColor(t.bearingChange)}
            strokeWidth={2}
          />
        ))}
        {startCoord && (
          <Marker coordinate={startCoord} anchor={{ x: 0.5, y: 0.5 }}>
            <Text style={styles.mapEmoji}>🟢</Text>
          </Marker>
        )}
        {endCoord && startCoord !== endCoord && (
          <Marker coordinate={endCoord} anchor={{ x: 0.5, y: 0.5 }}>
            <Text style={styles.mapEmoji}>🏁</Text>
          </Marker>
        )}
        {position && (
          <Marker coordinate={position} anchor={{ x: 0.5, y: 0.5 }} flat>
            <View style={{ transform: [{ rotate: `${position.heading ?? 0}deg` }] }}>
              <Svg width={38} height={38} viewBox="0 0 32 32">
                <Polygon
                  points="16,2 26,28 16,22 6,28"
                  fill="#1a52d5"
                  stroke="#ffffff"
                  strokeWidth={3}
                  strokeLinejoin="round"
                />
              </Svg>
            </View>
          </Marker>
        )}
      </MapView>

      {/* 現在地追跡ボタン（地図右上）— オリジナルの recenter() に相当 */}
      {!isFollowing && (
        <TouchableOpacity
          style={styles.recenterBtn}
          onPress={() => {
            setIsFollowing(true);
            isFollowingRef.current = true;
            const pos = positionRef.current;
            if (pos) mapRef.current?.animateCamera(
              { center: pos, zoom: 17 }, { duration: 500 },
            );
          }}
        >
          <Svg width={22} height={22} viewBox="0 0 32 32">
            <Polygon
              points="16,2 26,28 16,22 6,28"
              fill="#1a52d5"
              stroke="#ffffff"
              strokeWidth={3}
              strokeLinejoin="round"
            />
          </Svg>
        </TouchableOpacity>
      )}

      {/* 勾配マップオーバーレイ — 常にレンダリングして幅を計測、内容は gradeEnabled 時のみ */}
      <View
        style={styles.gradeOverlay}
        onLayout={e => setGradeWidth(e.nativeEvent.layout.width)}
      >
        {gradeEnabled && grads.length > 0 && gradeWidth > 0 && (
          <GradeChart grads={grads} trkIdx={trkIdx} width={gradeWidth} />
        )}
      </View>

      {/* ズームボタン（stats overlay の真上） */}
      <View style={styles.zoomBtns}>
        <TouchableOpacity style={styles.zoomBtn} onPress={zoomIn}>
          <Text style={styles.zoomBtnText}>＋</Text>
        </TouchableOpacity>
        <View style={styles.zoomDivider} />
        <TouchableOpacity style={styles.zoomBtn} onPress={zoomOut}>
          <Text style={styles.zoomBtnText}>－</Text>
        </TouchableOpacity>
      </View>

      {/* Stats overlay */}
      <View style={styles.statsOverlay}>
        <Text style={styles.statLabel}>残り距離</Text>
        <Text style={styles.statValue}>{formatDist(remainingDist)}</Text>
        <Text style={[styles.statLabel, { marginTop: 6 }]}>経過時間</Text>
        <Text style={styles.statValue}>{formatTime(elapsed)}</Text>
      </View>
      </View>{/* end map area wrapper */}

      {/* Bottom panel */}
      <View style={styles.panel}>
        {/* Turn instruction */}
        <View style={styles.turnRow}>
          {navTarget ? (
            <>
              <Text style={styles.turnArrow}>
                {navTarget.wp.isGoal && goalReached ? '🏁' : getTurnArrow(navTarget.wp.bearingChange)}
              </Text>
              <View style={styles.turnInfo}>
                <Text style={styles.turnDist}>{formatDist(navTarget.dist)}</Text>
                <Text style={styles.turnName}>{navTarget.wp.name}</Text>
              </View>
            </>
          ) : (
            <Text style={styles.turnName}>GPS待機中...</Text>
          )}
        </View>

        {/* After-next turn */}
        {navAfterNext && navTarget && !navTarget.wp.isGoal && (
          <Text style={styles.afterNext}>
            次：{navAfterNext.isGoal
              ? `このターン後${navAfterNext.name}へ直進`
              : `${getTurnArrow(navAfterNext.bearingChange)} ${navAfterNext.name}`}
          </Text>
        )}

        {/* Buttons */}
        <View style={styles.buttonRow}>
          <TouchableOpacity style={styles.btn} onPress={() => navigation.goBack()}>
            <Text style={styles.btnText}>← 一覧</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.btn, voiceEnabled && styles.btnActive]}
            onPress={toggleVoice}
          >
            <Text style={[styles.btnText, voiceEnabled && styles.btnTextActive]}>
              {voiceEnabled ? '🔊 音声' : '🔇 音声'}
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.btn}
            onPress={() => {
              rerouteToNearest();
              setIsFollowing(true);
              isFollowingRef.current = true;
              const pos = positionRef.current;
              if (pos) mapRef.current?.animateCamera(
                { center: pos, heading: pos.heading, zoom: 17 }, { duration: 500 },
              );
            }}
          >
            <Text style={styles.btnText}>🔄 復帰</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.btn, gradeEnabled && styles.btnActive]}
            onPress={() => setGradeEnabled(v => !v)}
          >
            <Text style={[styles.btnText, gradeEnabled && styles.btnTextActive]}>⛰ 勾配</Text>
          </TouchableOpacity>
          {__DEV__ && (
            <TouchableOpacity
              style={[styles.btn, simulating && styles.btnSim]}
              onPress={simulating ? stopSimulation : startSimulation}
            >
              <Text style={[styles.btnText, simulating && styles.btnTextActive]}>
                {simulating ? '■ 停止' : '▶ テスト'}
              </Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1a1a2e' },
  map: { flex: 1 },

  mapEmoji: { fontSize: 22 },

  recenterBtn: {
    position: 'absolute', top: 56, right: 12,
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: 10, padding: 10,
  },

  gradeOverlay: { position: 'absolute', bottom: 12, left: 12, right: 130 },

  statsOverlay: {
    position: 'absolute', bottom: 12, right: 12,
    backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: 10,
    paddingHorizontal: 12, alignItems: 'flex-end',
    width: 110, height: 90, justifyContent: 'center',
  },
  statLabel: { color: '#bbb', fontSize: 10 },
  statValue: { color: '#fff', fontSize: 17, fontWeight: 'bold' },

  zoomBtns: {
    position: 'absolute', bottom: 110, right: 12,
    width: 55,
    backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: 10, overflow: 'hidden',
  },
  zoomBtn: { alignItems: 'center', justifyContent: 'center', paddingVertical: 11 },
  zoomBtnText: { color: '#fff', fontSize: 22, fontWeight: '300', lineHeight: 24 },
  zoomDivider: { height: 1, backgroundColor: 'rgba(255,255,255,0.15)' },

  panel: { backgroundColor: '#1a1a2e', paddingBottom: 34 },

  turnRow: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 20, paddingVertical: 12 },
  turnArrow: { fontSize: 48, color: '#fff', marginRight: 16 },
  turnInfo: { flex: 1 },
  turnDist: { color: '#4A9EFF', fontSize: 32, fontWeight: 'bold' },
  turnName: { color: '#fff', fontSize: 18 },

  afterNext: { color: '#aaa', fontSize: 13, paddingHorizontal: 20, marginBottom: 8 },

  buttonRow: { flexDirection: 'row', paddingHorizontal: 16, gap: 10, marginTop: 4 },
  btn: { flex: 1, backgroundColor: '#2a2a4a', paddingVertical: 12, borderRadius: 8, alignItems: 'center' },
  btnActive: { backgroundColor: '#2a5298' },
  btnSim: { backgroundColor: '#2a4a2a' },
  btnText: { color: '#aaa', fontSize: 15 },
  btnTextActive: { color: '#fff', fontWeight: 'bold' },
});
