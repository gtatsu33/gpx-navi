import { useRef, useState } from 'react'
import { matchRoute } from '../lib/mapMatching.js'

function initialState() {
  return { status: 'idle', chunkIdx: 0, totalChunks: 0, nSnapped: 0, error: null }
}

/**
 * マップマッチングの進捗・キャンセルをReact状態として管理する薄いラッパー。
 * チャンク処理の実体はlib/mapMatching.jsのmatchRoute（React非依存）。
 */
export function useMapMatching() {
  const [state, setState] = useState(initialState())
  const cancelRequestedRef = useRef(false)

  function cancel() {
    cancelRequestedRef.current = true
  }

  async function run(points) {
    cancelRequestedRef.current = false
    return matchRoute(points, {
      shouldCancel: () => cancelRequestedRef.current,
      onProgress: (progress) => setState(progress),
    })
  }

  function reset() {
    cancelRequestedRef.current = false
    setState(initialState())
  }

  return { state, run, cancel, reset }
}
