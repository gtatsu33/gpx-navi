import { useCallback, useState } from 'react';
import { Alert, FlatList, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import * as DocumentPicker from 'expo-document-picker';
import { RootStackParamList } from '../../App';
import RoutePickerModal from '../components/RoutePickerModal';
import { deleteRoute, getAllRoutes, Route } from '../storage/routeStorage';
import { parseGPX } from '../lib/gpxParser';
import { geoDist, buildWptTrkMapping } from '../lib/geoCalc';
import { saveRoute } from '../storage/routeStorage';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

export default function HomeScreen({ navigation }: Props) {
  const [routes, setRoutes] = useState<Route[]>([]);
  const [pickerVisible, setPickerVisible] = useState(false);
  const [reverseMap, setReverseMap] = useState<Record<string, boolean>>({});

  const loadRoutes = useCallback(() => {
    getAllRoutes().then(all => setRoutes([...all].reverse()));
  }, []);

  useFocusEffect(useCallback(() => { loadRoutes(); }, [loadRoutes]));

  async function handleLocalImport() {
    try {
      const result = await DocumentPicker.getDocumentAsync({ type: '*/*', copyToCacheDirectory: true });
      if (result.canceled || !result.assets?.[0]) return;
      const asset = result.assets[0];
      const text = await fetch(asset.uri).then(r => r.text());
      const parsed = parseGPX(text);
      if (parsed.track.length < 2) { Alert.alert('エラー', 'GPXの解析に失敗しました'); return; }
      const turns = buildWptTrkMapping(parsed.track, parsed.turns);
      let totalDist = 0;
      for (let i = 1; i < parsed.track.length; i++)
        totalDist += geoDist(parsed.track[i-1].lat, parsed.track[i-1].lon, parsed.track[i].lat, parsed.track[i].lon);
      const route: Route = {
        id: `local_${Date.now()}`,
        name: parsed.name || asset.name || 'ルート',
        distKm: (totalDist / 1000).toFixed(1),
        importedAt: Date.now(),
        track: parsed.track,
        turns,
      };
      await saveRoute(route);
      loadRoutes();
      Alert.alert('完了', `「${route.name}」を追加しました`);
    } catch (e: any) {
      Alert.alert('エラー', `読み込み失敗: ${e.message}`);
    }
  }

  async function handleDelete(id: string, name: string) {
    Alert.alert('削除', `「${name}」を削除しますか？`, [
      { text: 'キャンセル', style: 'cancel' },
      { text: '削除', style: 'destructive', onPress: async () => { await deleteRoute(id); loadRoutes(); } },
    ]);
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.appTitle}>GPX Navi</Text>
        <View style={styles.headerButtons}>
          <TouchableOpacity style={styles.iconButton} onPress={handleLocalImport}>
            <Text style={styles.iconButtonText}>📂</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.addButton} onPress={() => setPickerVisible(true)}>
            <Text style={styles.addButtonText}>☁️ ルート追加</Text>
          </TouchableOpacity>
        </View>
      </View>

      <FlatList
        data={routes}
        keyExtractor={item => item.id}
        contentContainerStyle={routes.length === 0 ? styles.emptyContainer : undefined}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.card}
            onPress={() => navigation.navigate('Navigation', { routeId: item.id, routeName: item.name, isReverse: !!reverseMap[item.id] })}
          >
            <Text style={styles.routeIcon}>🚴</Text>
            <View style={{ flex: 1 }}>
              <Text style={styles.routeName}>{item.name}</Text>
              <Text style={styles.routeMeta}>{item.distKm} km</Text>
            </View>
            <TouchableOpacity
              style={[styles.reverseBtn, reverseMap[item.id] && styles.reverseBtnOn]}
              onPress={() => setReverseMap(m => ({ ...m, [item.id]: !m[item.id] }))}
            >
              <Text style={[styles.reverseBtnText, reverseMap[item.id] && styles.reverseBtnTextOn]}>逆走</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => handleDelete(item.id, item.name)}>
              <Text style={styles.deleteText}>削除</Text>
            </TouchableOpacity>
          </TouchableOpacity>
        )}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyIcon}>🗺️</Text>
            <Text style={styles.emptyText}>ルートがありません{'\n'}「＋ ルート追加」からネットワークルートを追加してください</Text>
          </View>
        }
      />

      <RoutePickerModal
        visible={pickerVisible}
        onClose={() => setPickerVisible(false)}
        onImported={loadRoutes}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1a1a2e' },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 16, paddingTop: 60 },
  appTitle: { color: '#fff', fontSize: 22, fontWeight: 'bold' },
  headerButtons: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  iconButton: { backgroundColor: '#2a2a4a', paddingVertical: 8, paddingHorizontal: 12, borderRadius: 8 },
  iconButtonText: { fontSize: 18 },
  addButton: { backgroundColor: '#2a5298', paddingVertical: 8, paddingHorizontal: 16, borderRadius: 8 },
  addButtonText: { color: '#fff', fontSize: 14, fontWeight: 'bold' },
  card: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#22223a', marginHorizontal: 16, marginBottom: 10, padding: 16, borderRadius: 12 },
  routeIcon: { fontSize: 28, marginRight: 12 },
  routeName: { color: '#fff', fontSize: 16, fontWeight: 'bold', marginBottom: 4 },
  routeMeta: { color: '#888', fontSize: 13 },
  reverseBtn: { borderWidth: 1, borderColor: '#555', borderRadius: 6, paddingVertical: 4, paddingHorizontal: 8, marginRight: 8 },
  reverseBtnOn: { borderColor: '#ffa502', backgroundColor: '#2a1e08' },
  reverseBtnText: { color: '#666', fontSize: 12 },
  reverseBtnTextOn: { color: '#ffa502' },
  deleteText: { color: '#ff6348', fontSize: 13 },
  emptyContainer: { flex: 1, justifyContent: 'center' },
  empty: { alignItems: 'center', padding: 40 },
  emptyIcon: { fontSize: 48, marginBottom: 16 },
  emptyText: { color: '#888', fontSize: 15, textAlign: 'center', lineHeight: 24 },
});
