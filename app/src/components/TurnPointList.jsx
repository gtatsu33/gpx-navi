import { wptStyle } from '../lib/turns.js'

/**
 * ターンポイント一覧パネル。spec.txt 15章。
 */
export default function TurnPointList({ routePoints, routeModified, canUndo, dispatch, onDetectTurns, onFocus }) {
  const currentWpts = routePoints
    .map((p, i) => (p.wpt ? { trkptIdx: i, p } : null))
    .filter(Boolean)

  return (
    <div className="turn-point-list">
      <h3>📋 ターンポイント一覧　({currentWpts.length}件)</h3>
      <div className="tpl-toolbar">
        <button type="button" onClick={onDetectTurns}>
          🔍 ターンポイント検出
        </button>
        <button type="button" disabled={!canUndo} onClick={() => dispatch({ type: 'UNDO' })}>
          ↩ 戻す
        </button>
      </div>
      {routeModified && currentWpts.length > 0 && (
        <p className="tpl-warning">⚠️ ルートが変更されています。ターンポイント検出を実行してください。</p>
      )}
      <div className="tpl-body">
        {currentWpts.length === 0 && <p>ターンポイントがありません。</p>}
        {currentWpts.map(({ trkptIdx, p }, listIdx) => {
          const [arrow] = wptStyle(p.wpt)
          const badge = p.wpt.delta !== null && p.wpt.delta !== undefined ? `${p.wpt.delta.toFixed(1)}°` : '手動'
          return (
            <div className="tpl-row" key={trkptIdx}>
              <button type="button" className="tpl-center-btn" onClick={() => onFocus({ lat: p.lat, lng: p.lon })}>
                {listIdx + 1} | {arrow} | trkpt: {trkptIdx} | {badge}
              </button>
              <input
                type="text"
                value={p.wpt.name}
                onChange={(e) => dispatch({ type: 'RENAME_WPT', payload: { trkptIndex: trkptIdx, name: e.target.value } })}
              />
              <button
                type="button"
                title="削除"
                onClick={() => dispatch({ type: 'DELETE_WPT', payload: { trkptIndex: trkptIdx } })}
              >
                🗑
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
