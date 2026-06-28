import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Modal,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { supabase } from '../lib/supabaseClient';
import { parseGPX } from '../lib/gpxParser';
import { geoDist, buildWptTrkMapping } from '../lib/geoCalc';
import { saveRoute, Route } from '../storage/routeStorage';

type NetworkRoute = {
  file_key: string;
  display_name: string;
  distance_m: number | null;
  elevation_gain_m: number | null;
};

type Props = {
  visible: boolean;
  onClose: () => void;
  onImported: () => void;
};

export default function RoutePickerModal({ visible, onClose, onImported }: Props) {
  const [routes, setRoutes] = useState<NetworkRoute[]>([]);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!visible) return;
    setLoading(true);
    setError(null);
    supabase
      .from('route_files')
      .select('*')
      .order('created_at', { ascending: false })
      .then(({ data, error: err }) => {
        if (err) setError(err.message);
        else setRoutes(data ?? []);
        setLoading(false);
      });
  }, [visible]);

  async function handleSelect(r: NetworkRoute) {
    setDownloading(r.file_key);
    try {
      const { data: signed, error: signErr } = await supabase
        .storage.from('gpx_routes')
        .createSignedUrl(r.file_key, 60);
      if (signErr) throw new Error(signErr.message);

      const response = await fetch(signed.signedUrl);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const gpxText = await response.text();
      const parsed = parseGPX(gpxText);
      if (parsed.track.length < 2) throw new Error('GPXの解析に失敗しました');

      const turns = buildWptTrkMapping(parsed.track, parsed.turns);

      let totalDist = 0;
      for (let i = 1; i < parsed.track.length; i++) {
        totalDist += geoDist(
          parsed.track[i - 1].lat, parsed.track[i - 1].lon,
          parsed.track[i].lat, parsed.track[i].lon,
        );
      }

      const route: Route = {
        id: r.file_key,
        name: parsed.name || r.display_name,
        distKm: (totalDist / 1000).toFixed(1),
        fileKey: r.file_key,
        importedAt: Date.now(),
        track: parsed.track,
        turns,
      };

      await saveRoute(route);
      onImported();
      onClose();
    } catch (e: any) {
      Alert.alert('エラー', `読み込み失敗: ${e.message}`);
    } finally {
      setDownloading(null);
    }
  }

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>ネットワークルート</Text>
          <TouchableOpacity onPress={onClose} disabled={!!downloading}>
            <Text style={[styles.closeText, downloading ? styles.disabled : null]}>閉じる</Text>
          </TouchableOpacity>
        </View>

        {loading && <ActivityIndicator style={styles.center} color="#7ec8e3" size="large" />}
        {error && <Text style={styles.error}>⚠️ {error}</Text>}

        {!loading && !error && (
          <FlatList
            data={routes}
            keyExtractor={item => item.file_key}
            renderItem={({ item }) => {
              const dist = item.distance_m != null ? `${(item.distance_m / 1000).toFixed(1)} km` : '---';
              const gain = item.elevation_gain_m != null ? `${Math.round(item.elevation_gain_m)} m↑` : '---';
              const isDownloading = downloading === item.file_key;
              return (
                <TouchableOpacity
                  style={styles.item}
                  onPress={() => handleSelect(item)}
                  disabled={!!downloading}
                >
                  <View style={{ flex: 1 }}>
                    <Text style={styles.itemName}>{item.display_name}</Text>
                    <Text style={styles.itemMeta}>{dist}　|　{gain}</Text>
                  </View>
                  {isDownloading
                    ? <ActivityIndicator color="#7ec8e3" />
                    : <Text style={styles.chevron}>›</Text>
                  }
                </TouchableOpacity>
              );
            }}
            ListEmptyComponent={<Text style={styles.empty}>ルートがありません</Text>}
          />
        )}
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1a1a2e' },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 16, borderBottomWidth: 1, borderBottomColor: '#333' },
  title: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  closeText: { color: '#7ec8e3', fontSize: 16 },
  disabled: { opacity: 0.4 },
  center: { flex: 1 },
  error: { color: '#ff6348', margin: 24, textAlign: 'center' },
  item: { flexDirection: 'row', alignItems: 'center', padding: 16, borderBottomWidth: 1, borderBottomColor: '#2a2a4a' },
  itemName: { color: '#fff', fontSize: 16, marginBottom: 4 },
  itemMeta: { color: '#888', fontSize: 13 },
  chevron: { color: '#888', fontSize: 22, marginLeft: 8 },
  empty: { color: '#888', textAlign: 'center', marginTop: 40 },
});
