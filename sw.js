// GPX ナビ Service Worker
// アプリシェルとマップタイルをキャッシュしてオフライン動作を実現する

const SHELL_CACHE = 'gpxnav-shell-v4';
const TILE_CACHE  = 'gpxnav-tiles-v1';

const SHELL_ASSETS = [
  './',
  './index.html',
  './manifest.json',
  './gpx-navi_180x180.png',
  './gpx-navi_192x192.png',
  './gpx-navi_512x512.png',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
  'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
  'https://fonts.googleapis.com/css2?family=Barlow:wght@400;600;700;900&family=Noto+Sans+JP:wght@400;700&display=swap',
];

// ── インストール: アプリシェルをキャッシュ ──
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(SHELL_CACHE)
      .then(cache => cache.addAll(SHELL_ASSETS).catch(() => {}))
      .then(() => self.skipWaiting())
  );
});

// ── アクティベート: 古いキャッシュを削除 ──
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(k => k !== SHELL_CACHE && k !== TILE_CACHE)
          .map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

// ── フェッチ: キャッシュファースト戦略 ──
self.addEventListener('fetch', event => {
  const url = event.request.url;

  // OSM タイルはタイルキャッシュを使用
  if (url.includes('tile.openstreetmap.org')) {
    event.respondWith(tileFirst(event.request));
    return;
  }

  // フォント・その他CDNはキャッシュファースト
  if (url.includes('fonts.googleapis.com') ||
      url.includes('fonts.gstatic.com') ||
      url.includes('unpkg.com')) {
    event.respondWith(cacheFirst(event.request, SHELL_CACHE));
    return;
  }

  // アプリシェル（同一オリジン）
  if (url.startsWith(self.location.origin)) {
    event.respondWith(cacheFirst(event.request, SHELL_CACHE));
    return;
  }
});

// タイル用: キャッシュ優先、なければネットワーク取得してキャッシュ
async function tileFirst(request) {
  const cache  = await caches.open(TILE_CACHE);
  const cached = await cache.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    // オフライン時はグレーのプレースホルダを返す
    return new Response(
      '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="#2a2a3e"/></svg>',
      { headers: { 'Content-Type': 'image/svg+xml' } }
    );
  }
}

// 汎用: キャッシュ優先
async function cacheFirst(request, cacheName) {
  const cache  = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return new Response('Offline', { status: 503 });
  }
}
