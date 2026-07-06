import { useEffect, useRef, useState } from 'react'
import { buildGpx } from '../lib/gpx.js'
import { cleanElevationSpikes, computeGradeStats, fetchElevationsForIndices } from '../lib/elevation.js'
import { isSupabaseConfigured, uploadGpx } from '../lib/supabase.js'

function isOrgComplete(routePoints) {
  return routePoints.length > 0 && routePoints.every((p) => p.eleOrg !== null && p.eleOrg !== undefined)
}

function isAscii(str) {
  return /^[\x00-\x7F]*$/.test(str)
}

function recommendChoice(gradeOrg, gradeFix, orgOk, fixOk) {
  if (orgOk && fixOk) {
    const so = gradeOrg.max + Math.abs(gradeOrg.min)
    const sf = gradeFix.max + Math.abs(gradeFix.min)
    return sf <= so ? 'fix' : 'org'
  }
  if (fixOk) return 'fix'
  return 'org'
}

/**
 * 保存画面。spec.txt 16章。
 * 16-2: 開いた直後に標高整合性チェック（ele_fixが未確定の点だけ取得＋
 *       スパイク除去。org/fixそれぞれのデータが全点揃っているかだけで
 *       選択肢の有効/無効を決める）
 * 16-3: GPXビルド
 * 16-4: ファイル名・クラウド保存・ダウンロード
 */
export default function SaveDialog({
  routePoints,
  gradeOrg,
  eleChoice,
  routeModified,
  dispatch,
  rawGpxString,
  defaultFilename,
  totalDistKm,
  gainM,
  onClose,
  isLoggedIn,
}) {
  const [checkStatus, setCheckStatus] = useState({ phase: 'checking', done: 0, total: 0 })
  const [filename, setFilename] = useState(defaultFilename)
  const [uploadToCloud, setUploadToCloud] = useState(false)
  const [supabaseFilename, setSupabaseFilename] = useState('')
  const [uploadResult, setUploadResult] = useState(null)
  const [uploading, setUploading] = useState(false)
  const ranRef = useRef(false)

  const orgAvailable = isOrgComplete(routePoints)

  useEffect(() => {
    if (ranRef.current) return
    ranRef.current = true

    async function runReconciliation() {
      const points = routePoints.map((p) => [p.lat, p.lon])
      // spec.txt 16-2章: ele_fixが未確定(null)の点だけ取得する。既に確定
      // している点（5-4章のeleSource=gsiフラグにより読込時にele_orgから
      // 複製済みの点を含む）は再取得しない。
      const indices = routePoints.map((p, i) => (p.eleFix === null ? i : null)).filter((i) => i !== null)

      const currentFix = routePoints.map((p) => p.eleFix)

      if (indices.length) {
        setCheckStatus({ phase: 'fetching', done: 0, total: indices.length })
        const assignments = await fetchElevationsForIndices(points, indices, {
          onProgress: (p) => setCheckStatus({ phase: 'fetching', done: p.done, total: p.total }),
        })
        assignments.forEach(({ trkptIndex, value }) => {
          currentFix[trkptIndex] = value
        })
      }

      const { cleaned } = cleanElevationSpikes(points, currentFix)
      const gradeFix = computeGradeStats(points, cleaned)
      const fixAvailable = cleaned.every((v) => v !== null)
      const choice = recommendChoice(gradeOrg, gradeFix, orgAvailable, fixAvailable)

      dispatch({ type: 'FINALIZE_SAVE_ELEVATION', payload: { fixValues: cleaned, gradeFix, choice } })
      setCheckStatus({ phase: 'done', done: indices.length, total: indices.length })
    }

    runReconciliation()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const checking = checkStatus.phase !== 'done'
  const needsAsciiName = uploadToCloud && !isAscii(filename)
  const downloadDisabled = checking || (needsAsciiName && !supabaseFilename)

  async function handleDownload() {
    const xml = buildGpx({
      baseXmlString: rawGpxString,
      routePoints,
      eleChoice,
      routeName: filename,
    })

    const blob = new Blob([xml], { type: 'application/gpx+xml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${filename}_gne.gpx`
    a.click()
    URL.revokeObjectURL(url)

    if (uploadToCloud) {
      setUploading(true)
      const supabaseName = needsAsciiName ? supabaseFilename : filename
      const result = await uploadGpx(xml, `${supabaseName}_gne`, {
        displayName: filename,
        distanceM: Math.round(totalDistKm * 1000),
        elevationGainM: gainM !== null ? Math.round(gainM) : null,
      })
      setUploading(false)
      setUploadResult(result)
      if (!result.ok) return
    }
    onClose()
  }

  return (
    <div className="modal-overlay">
      <div className="modal-box save-dialog">
        <h2>🚴 gpx-editor</h2>
        <div className="save-dialog-inner">
          <h3>💾 GPXを保存</h3>

          {checking ? (
            <>
              <p>⛰️ 標高データを確認・取得中… {checkStatus.done}/{checkStatus.total}点</p>
              <div className="progress-bar-track">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${checkStatus.total ? (checkStatus.done / checkStatus.total) * 100 : 100}%` }}
                />
              </div>
            </>
          ) : (
            <>
              <p>
                ターンポイント: {routeModified ? '⚠️ 未確定（ルート変更後に再検出を推奨）' : `✅ 設定済み`}
              </p>

              <EleChoiceRadio
                gradeOrg={gradeOrg}
                eleChoice={eleChoice}
                dispatch={dispatch}
                routePoints={routePoints}
                orgAvailable={orgAvailable}
              />

              <hr />

              <label>
                ファイル名
                <input type="text" className="text-input" value={filename} onChange={(e) => setFilename(e.target.value)} />
              </label>
              <p className="save-filename-preview">保存ファイル名: {filename}_gne.gpx</p>

              <label className={!isSupabaseConfigured() || !isLoggedIn ? 'save-dialog-label-disabled' : undefined}>
                <input
                  type="checkbox"
                  checked={uploadToCloud}
                  onChange={(e) => setUploadToCloud(e.target.checked)}
                  disabled={!isSupabaseConfigured() || !isLoggedIn}
                  title={!isLoggedIn ? '招待ユーザー限定の機能です' : undefined}
                />
                ☁️ クラウドにも保存
                {!isSupabaseConfigured() && <span className="ele-forced-note"> （Supabase未設定のため利用できません）</span>}
                {isSupabaseConfigured() && !isLoggedIn && (
                  <span className="ele-forced-note"> （招待ユーザー限定の機能です）</span>
                )}
              </label>

              {needsAsciiName && (
                <div>
                  <p>⚠️ ファイル名に2byte文字が含まれています。半角英数字のみでファイル名を付け直してください。</p>
                  <input
                    type="text"
                    className="text-input"
                    placeholder="例: osanpo_14km"
                    value={supabaseFilename}
                    onChange={(e) => setSupabaseFilename(e.target.value)}
                  />
                </div>
              )}

              {uploadResult && !uploadResult.ok && <p className="error">⚠️ {uploadResult.message}</p>}

              <div className="save-dialog-buttons">
                <button type="button" className="btn-secondary" onClick={onClose}>
                  キャンセル
                </button>
                <button type="button" className="btn-primary" onClick={handleDownload} disabled={downloadDisabled || uploading}>
                  {uploading ? 'アップロード中…' : '💾 保存'}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function EleChoiceRadio({ gradeOrg, eleChoice, dispatch, routePoints, orgAvailable }) {
  const fixOk = routePoints.length > 0 && routePoints.every((p) => p.eleFix !== null && p.eleFix !== undefined)
  const gradeFixDisplay = fixOk ? computeGradeStatsDisplay(routePoints) : null

  return (
    <div className="ele-choice">
      <p>標高データを選択:</p>
      <label>
        <input
          type="radio"
          name="eleChoice"
          checked={eleChoice === 'org'}
          disabled={!orgAvailable}
          onChange={() => dispatch({ type: 'SET_ELE_CHOICE', payload: { choice: 'org' } })}
        />
        元データ{orgAvailable && gradeOrg ? `　上り ${gradeOrg.max.toFixed(1)}%　下り ${gradeOrg.min.toFixed(1)}%` : ''}
        {!orgAvailable && '（このルートでは利用できません）'}
      </label>
      <label>
        <input
          type="radio"
          name="eleChoice"
          checked={eleChoice === 'fix'}
          disabled={!fixOk}
          onChange={() => dispatch({ type: 'SET_ELE_CHOICE', payload: { choice: 'fix' } })}
        />
        国土地理院補正{gradeFixDisplay ? `　上り ${gradeFixDisplay.max.toFixed(1)}%　下り ${gradeFixDisplay.min.toFixed(1)}%` : '（データなし）'}
      </label>
    </div>
  )
}

function computeGradeStatsDisplay(routePoints) {
  const points = routePoints.map((p) => [p.lat, p.lon])
  const fixVals = routePoints.map((p) => p.eleFix)
  return computeGradeStats(points, fixVals)
}
