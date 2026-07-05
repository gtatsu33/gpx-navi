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
    return sf < so ? 'fix' : 'org'
  }
  if (fixOk) return 'fix'
  return 'org'
}

/**
 * 保存画面。spec.txt 16章。
 * 16-2: 開いた直後に標高整合性チェック（org完全なら全点fix再取得＋比較、
 *       編集済みならfix未確定点を取得してfixに統一・選択UI無し）
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
}) {
  const [checkStatus, setCheckStatus] = useState({ phase: 'checking', done: 0, total: 0 })
  const [filename, setFilename] = useState(defaultFilename)
  const [uploadToCloud, setUploadToCloud] = useState(false)
  const [supabaseFilename, setSupabaseFilename] = useState('')
  const [uploadResult, setUploadResult] = useState(null)
  const [uploading, setUploading] = useState(false)
  const ranRef = useRef(false)

  const orgComplete = isOrgComplete(routePoints)

  useEffect(() => {
    if (ranRef.current) return
    ranRef.current = true

    async function runReconciliation() {
      const points = routePoints.map((p) => [p.lat, p.lon])
      const indices = orgComplete
        ? routePoints.map((_, i) => i)
        : routePoints.map((p, i) => (p.eleFix === null ? i : null)).filter((i) => i !== null)

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
      const fixOk = cleaned.every((v) => v !== null)
      const choice = orgComplete ? recommendChoice(gradeOrg, gradeFix, orgComplete, fixOk) : 'fix'

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

            {orgComplete ? (
              <EleChoiceRadio gradeOrg={gradeOrg} eleChoice={eleChoice} dispatch={dispatch} routePoints={routePoints} />
            ) : (
              <p className="ele-forced-note">
                ⛰️ ルートが編集されているため、国土地理院データに統一します（元データとの混在は行いません）。
              </p>
            )}

            <hr />

            <label>
              ファイル名
              <input type="text" value={filename} onChange={(e) => setFilename(e.target.value)} />
            </label>
            <p className="save-filename-preview">保存ファイル名: {filename}_gne.gpx</p>

            <label>
              <input type="checkbox" checked={uploadToCloud} onChange={(e) => setUploadToCloud(e.target.checked)} disabled={!isSupabaseConfigured()} />
              ☁️ ネットワーク上にも保存
              {!isSupabaseConfigured() && <span className="ele-forced-note"> （Supabase未設定のため利用できません）</span>}
            </label>

            {needsAsciiName && (
              <div>
                <p>⚠️ ファイル名に2byte文字が含まれています。半角英数字のみでファイル名を付け直してください。</p>
                <input
                  type="text"
                  placeholder="例: osanpo_14km"
                  value={supabaseFilename}
                  onChange={(e) => setSupabaseFilename(e.target.value)}
                />
              </div>
            )}

            {uploadResult && !uploadResult.ok && <p className="error">⚠️ {uploadResult.message}</p>}

            <div className="save-dialog-buttons">
              <button type="button" onClick={onClose}>
                キャンセル
              </button>
              <button type="button" onClick={handleDownload} disabled={downloadDisabled || uploading}>
                {uploading ? 'アップロード中…' : '⬇️ ダウンロード'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function EleChoiceRadio({ gradeOrg, eleChoice, dispatch, routePoints }) {
  const fixOk = routePoints.length > 0 && routePoints.every((p) => p.eleFix !== null && p.eleFix !== undefined)
  const gradeFixDisplay = fixOk ? computeGradeStatsDisplay(routePoints) : null

  return (
    <div className="ele-choice">
      <p>標高データを選択:</p>
      <label>
        <input type="radio" name="eleChoice" checked={eleChoice === 'org'} onChange={() => dispatch({ type: 'SET_ELE_CHOICE', payload: { choice: 'org' } })} />
        元データ{gradeOrg ? `　上り ${gradeOrg.max.toFixed(1)}%　下り ${gradeOrg.min.toFixed(1)}%` : ''}
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
