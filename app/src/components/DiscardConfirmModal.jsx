// spec.txt 4章・implement.txt 1章: 破棄確認モーダル。
function DiscardConfirmModal({ onDiscard, onCancel }) {
  return (
    <div className="modal-overlay">
      <div className="modal-box">
        <p>編集中の内容は破棄されます。スタート画面に戻りますか？</p>
        <div className="save-dialog-buttons">
          <button type="button" className="btn-secondary" onClick={onCancel}>
            ✏️ 編集を続ける
          </button>
          <button type="button" className="btn-danger" onClick={onDiscard}>
            🏠 スタート画面に戻る
          </button>
        </div>
      </div>
    </div>
  )
}

export default DiscardConfirmModal
