/**
 * マップマッチング進捗ダイアログ。spec.txt 10-2章。
 * 実走行データ読込時の一度きりの処理であり、標高のリアルタイム背景取得
 * (13-2章)とは性質が異なるため、モーダルダイアログのままでよい
 * （implement.txt 8章）。
 */
export default function MapMatchingDialog({ state, onCancel }) {
  if (state.status !== 'running') return null

  const progress = state.totalChunks ? Math.round((state.chunkIdx / state.totalChunks) * 100) : 0

  return (
    <div className="modal-overlay">
      <div className="modal-box">
        <h3>🗺️ マップマッチング中</h3>
        <div className="progress-bar-track">
          <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
        </div>
        <p>
          {state.chunkIdx}/{state.totalChunks} チャンク
        </p>
        <button type="button" onClick={onCancel}>
          ⏹ キャンセル
        </button>
      </div>
    </div>
  )
}
