/** テスト時に差し替え可能なsleep実装（フェイクタイマー対応のため注入式にする）。 */
export function defaultSleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * タイムアウト付きfetch。fetchImplを注入できるようにし、モックテストを可能にする。
 * implement.txt 14章の共通fetchラッパー。
 */
export async function fetchWithTimeout(url, { timeoutMs = 30000, fetchImpl = fetch, ...options } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetchImpl(url, { ...options, signal: controller.signal })
  } finally {
    clearTimeout(timer)
  }
}
