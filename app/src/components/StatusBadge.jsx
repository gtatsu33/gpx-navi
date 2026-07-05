/**
 * 標高取得状況などの控えめなステータス表示。spec.txt 16-1章／implement.txt 2-6章。
 * モーダルは使わず、小さなインライン表示のみ行う。
 */
export default function StatusBadge({ status, onRetry }) {
  if (status.state === 'out_of_japan') {
    return <span className="status-badge">対象外のルートです（国内ルートのみ）</span>
  }
  if (status.state === 'running') {
    return (
      <span className="status-badge">
        ⛰️ 標高取得中… 確定{status.done}/全{status.total}点
      </span>
    )
  }
  if (status.state === 'circuit_open') {
    return <span className="status-badge warn">⚠️ 標高取得が一時停止中です</span>
  }
  if (status.unavailable > 0) {
    return (
      <button type="button" className="status-badge warn status-badge-btn" onClick={onRetry}>
        ⚠️ {status.unavailable}点は取得できませんでした（クリックで再試行）
      </button>
    )
  }
  if (status.state === 'done') {
    return <span className="status-badge">⛰️ 標高取得完了</span>
  }
  return null
}
