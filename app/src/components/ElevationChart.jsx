import { useRef } from 'react'
import { cumulativeDistances } from '../lib/elevation.js'
import { nearestIndexAtDistance } from '../lib/mapInteractions.js'

const WIDTH = 1000
const HEIGHT = 150
const MARGIN = { left: 50, right: 10, top: 10, bottom: 30 }
const PLOT_WIDTH = WIDTH - MARGIN.left - MARGIN.right
const PLOT_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom

function fillSeries(vals, eleMin) {
  return vals.map((v) => (v === null || v === undefined ? eleMin : v))
}

/**
 * 標高プロファイルグラフ（自前SVG）。spec.txt 4章・13-4章・7-3章に対応。
 * 外部チャートライブラリは使わない（implement.txt 10章）。
 */
export default function ElevationChart({ routePoints, hoveredKm, onHoverKm, onClickKm }) {
  const svgRef = useRef(null)

  if (routePoints.length < 2) return null

  const cumDistsM = cumulativeDistances(routePoints.map((p) => [p.lat, p.lon]))
  const cumKm = cumDistsM.map((d) => d / 1000)
  const totalKm = cumKm[cumKm.length - 1]

  const orgVals = routePoints.map((p) => p.eleOrg)
  const fixVals = routePoints.map((p) => p.eleFix)
  const definedVals = [...orgVals, ...fixVals].filter((v) => v !== null && v !== undefined)
  let eleMin = 0
  let eleMax = 1
  if (definedVals.length) {
    eleMin = Math.min(...definedVals)
    eleMax = Math.max(...definedVals)
    if (eleMin === eleMax) eleMax = eleMin + 1
  }

  const xScale = (km) => MARGIN.left + (km / totalKm) * PLOT_WIDTH
  const yScale = (ele) => MARGIN.top + (1 - (ele - eleMin) / (eleMax - eleMin)) * PLOT_HEIGHT

  const buildPath = (vals) =>
    fillSeries(vals, eleMin)
      .map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(cumKm[i]).toFixed(1)},${yScale(v).toFixed(1)}`)
      .join(' ')

  const hasOrg = orgVals.some((v) => v !== null && v !== undefined)
  const hasFix = fixVals.some((v) => v !== null && v !== undefined)

  const tickStep = totalKm < 50 ? 5 : 10
  const ticks = []
  for (let km = tickStep; km < totalKm; km += tickStep) ticks.push(km)

  const wptKms = routePoints.map((p, i) => (p.wpt ? cumKm[i] : null)).filter((v) => v !== null)

  function kmFromClientX(clientX) {
    const rect = svgRef.current.getBoundingClientRect()
    const relX = ((clientX - rect.left) / rect.width) * WIDTH
    const km = ((relX - MARGIN.left) / PLOT_WIDTH) * totalKm
    return Math.min(Math.max(km, 0), totalKm)
  }

  let hoverElevation = null
  if (hoveredKm !== null && hoveredKm !== undefined) {
    const idx = nearestIndexAtDistance(cumDistsM, hoveredKm * 1000)
    const v = fixVals[idx] !== null && fixVals[idx] !== undefined ? fixVals[idx] : orgVals[idx]
    if (v !== null && v !== undefined) hoverElevation = v
  }

  return (
    <svg
      ref={svgRef}
      className="elevation-chart"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      width="100%"
      height={HEIGHT}
      onMouseMove={(e) => onHoverKm(kmFromClientX(e.clientX))}
      onMouseLeave={() => onHoverKm(null)}
      onClick={(e) => onClickKm(kmFromClientX(e.clientX))}
    >
      <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="white" />

      {eleMin <= 0 && 0 <= eleMax && (
        <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={yScale(0)} y2={yScale(0)} stroke="gray" strokeWidth={1} />
      )}

      {wptKms.map((km, i) => (
        <line
          key={i}
          x1={xScale(km)}
          x2={xScale(km)}
          y1={MARGIN.top}
          y2={HEIGHT - MARGIN.bottom}
          stroke="rgba(100,100,200,0.5)"
          strokeWidth={1}
        />
      ))}

      {ticks.map((km) => (
        <g key={km}>
          <line
            x1={xScale(km)}
            x2={xScale(km)}
            y1={HEIGHT - MARGIN.bottom}
            y2={HEIGHT - MARGIN.bottom + 4}
            stroke="#888"
          />
          <text x={xScale(km)} y={HEIGHT - MARGIN.bottom + 16} fontSize="9" textAnchor="middle" fill="#888">
            {km}km
          </text>
        </g>
      ))}

      {hasOrg && <path d={buildPath(orgVals)} fill="none" stroke="black" strokeWidth={1.5} strokeDasharray="4,3" />}
      {hasFix && <path d={buildPath(fixVals)} fill="none" stroke="black" strokeWidth={1.5} />}

      {hoveredKm !== null && hoveredKm !== undefined && (
        <>
          <line
            x1={xScale(hoveredKm)}
            x2={xScale(hoveredKm)}
            y1={MARGIN.top}
            y2={HEIGHT - MARGIN.bottom}
            stroke="#e67e22"
            strokeWidth={1}
          />
          {hoverElevation !== null && (
            <text
              x={Math.min(Math.max(xScale(hoveredKm), MARGIN.left + 20), WIDTH - MARGIN.right - 20)}
              y={MARGIN.top + 10}
              fontSize="22"
              fontWeight="bold"
              textAnchor="middle"
              fill="#e67e22"
            >
              {hoverElevation.toFixed(0)}m
            </text>
          )}
        </>
      )}

      <text x={WIDTH - MARGIN.right} y={HEIGHT - 4} fontSize="9" textAnchor="end" fill="#666">
        {hasOrg && '- - 元データ標高　'}
        {hasFix && '— 国土地理院補正標高'}
      </text>
    </svg>
  )
}
