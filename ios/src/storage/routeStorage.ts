import AsyncStorage from '@react-native-async-storage/async-storage';
import { TrackPoint } from '../lib/gpxParser';
import { Turn } from '../lib/geoCalc';

export type Route = {
  id: string;
  name: string;
  distKm: string;
  fileKey?: string;
  importedAt: number;
  track?: TrackPoint[];
  turns?: Turn[];
};

const STORAGE_KEY = 'routes_v1';

async function loadAll(): Promise<Route[]> {
  const json = await AsyncStorage.getItem(STORAGE_KEY);
  return json ? JSON.parse(json) : [];
}

export async function getAllRoutes(): Promise<Route[]> {
  return loadAll();
}

export async function saveRoute(route: Route): Promise<void> {
  const routes = await loadAll();
  const idx = routes.findIndex(r => r.id === route.id);
  if (idx >= 0) {
    routes[idx] = route; // update if already exists (re-import upgrades metadata)
  } else {
    routes.push(route);
  }
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(routes));
}

export async function deleteRoute(id: string): Promise<void> {
  const routes = await loadAll();
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(routes.filter(r => r.id !== id)));
}
