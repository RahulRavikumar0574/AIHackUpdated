import React from 'react';

// A simple semicircle gauge showing risk vs threshold
// Props: risk (0..1), threshold (0..1)
const clamp01 = (v) => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));

const RiskGauge = ({ risk = 0.5, threshold = 0.5, width = 320, height = 180 }) => {
  const r = Math.min(width / 2 - 10, height - 20);
  const cx = width / 2;
  const cy = height - 10;

  const val = clamp01(risk);
  const th = clamp01(threshold);

  // Map [0,1] to angle [180deg..0deg]
  const toAngle = (x) => Math.PI - x * Math.PI;
  const angleVal = toAngle(val);
  const angleTh = toAngle(th);

  const xVal = cx + r * Math.cos(angleVal);
  const yVal = cy + r * Math.sin(angleVal);

  const xTh = cx + r * Math.cos(angleTh);
  const yTh = cy + r * Math.sin(angleTh);

  // Arc path for background
  const startX = cx - r;
  const startY = cy;
  const endX = cx + r;
  const endY = cy;

  return (
    <div className="w-full flex flex-col items-center">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {/* Arc background */}
        <path d={`M ${startX} ${startY} A ${r} ${r} 0 0 1 ${endX} ${endY}`} fill="none" stroke="#e5e7eb" strokeWidth="16" />

        {/* Threshold tick */}
        <line x1={xTh} y1={yTh} x2={cx} y2={cy} stroke="#f59e0b" strokeWidth="3" opacity="0.6" />

        {/* Value needle */}
        <line x1={cx} y1={cy} x2={xVal} y2={yVal} stroke={val < th ? '#10b981' : '#ef4444'} strokeWidth="5" />

        {/* Center cap */}
        <circle cx={cx} cy={cy} r="6" fill="#374151" />

        {/* Labels */}
        <text x={cx - r} y={cy + 16} fill="#6b7280" fontSize="12">0%</text>
        <text x={cx} y={cy - r - 5} fill="#6b7280" fontSize="12" textAnchor="middle">Risk</text>
        <text x={cx + r - 24} y={cy + 16} fill="#6b7280" fontSize="12">100%</text>
      </svg>
      <div className="mt-2 text-sm text-gray-700">
        <span className="font-semibold">Risk:</span> {(val * 100).toFixed(1)}% {' '}
        <span className="mx-1">|</span>
        <span className="font-semibold">Threshold:</span> {(th * 100).toFixed(1)}%
      </div>
    </div>
  );
};

export default RiskGauge;
