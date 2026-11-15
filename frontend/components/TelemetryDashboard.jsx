/**
 * ============================================
 * SuperStream AI — Telemetry Dashboard
 * ============================================
 * Real-time performance metrics HUD displaying:
 * - FPS (frames per second)
 * - Inference time (ms per frame)
 * - Bandwidth saved (percentage)
 * - Model info & resolution
 * - Total frames processed
 * - Uptime
 *
 * Features a rolling sparkline chart for FPS
 * history and animated metric cards.
 * ============================================
 */

import { useEffect, useRef, useState, useCallback } from "react";

// ── Maximum history points for sparkline ──
const SPARKLINE_POINTS = 60;

/**
 * Sparkline — Tiny inline chart for FPS history.
 * Renders an SVG polyline with gradient fill.
 */
function Sparkline({ data, color, height = 32 }) {
  if (!data || data.length < 2) return null;

  const width = 160;
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const padding = 2;

  const points = data
    .map((val, i) => {
      const x = (i / (data.length - 1)) * (width - padding * 2) + padding;
      const y =
        height - padding - ((val - min) / range) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  // Create filled area path
  const firstX = padding;
  const lastX =
    ((data.length - 1) / (data.length - 1)) * (width - padding * 2) + padding;
  const fillPoints = `${firstX},${height} ${points} ${lastX},${height}`;

  const gradientId = `sparkline-${color.replace("#", "")}`;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="opacity-80"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <polygon points={fillPoints} fill={`url(#${gradientId})`} />
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * MetricCard — Individual metric display with label,
 * value, unit, and optional sparkline.
 */
function MetricCard({
  label,
  value,
  unit,
  color,
  icon,
  sparklineData,
  subtext,
  animateId,
}) {
  return (
    <div
      id={animateId}
      className="glass-card neon-border p-4 lg:p-5 flex flex-col justify-between min-h-[110px] transition-all duration-300 hover:scale-[1.02]"
    >
      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-base opacity-70">{icon}</span>
          <span className="font-mono text-xs uppercase tracking-widest text-gray-500">
            {label}
          </span>
        </div>
        {sparklineData && (
          <Sparkline data={sparklineData} color={color} height={28} />
        )}
      </div>

      {/* ── Value ── */}
      <div className="flex items-baseline gap-1.5">
        <span
          className="font-display font-bold text-3xl lg:text-4xl tracking-tight"
          style={{ color }}
        >
          {value}
        </span>
        <span className="font-mono text-sm text-gray-500">{unit}</span>
      </div>

      {/* ── Subtext ── */}
      {subtext && (
        <p className="font-mono text-xs text-gray-600 mt-1">{subtext}</p>
      )}
    </div>
  );
}

/**
 * TelemetryDashboard — Main telemetry HUD component.
 * Receives metrics from the VideoPlayer via props.
 */
export default function TelemetryDashboard({ metrics }) {
  const [fpsHistory, setFpsHistory] = useState([]);
  const [inferenceHistory, setInferenceHistory] = useState([]);

  // ── Update sparkline history when metrics change ──
  useEffect(() => {
    if (!metrics) return;

    setFpsHistory((prev) => {
      const next = [...prev, metrics.fps || 0];
      return next.slice(-SPARKLINE_POINTS);
    });

    setInferenceHistory((prev) => {
      const next = [...prev, metrics.inference_ms || 0];
      return next.slice(-SPARKLINE_POINTS);
    });
  }, [metrics]);

  // ── Derive display values ──
  const fps = metrics?.fps?.toFixed(1) ?? "—";
  const inferenceMs = metrics?.inference_ms?.toFixed(1) ?? "—";
  const bandwidthSaved = metrics?.bandwidth_saved_pct?.toFixed(1) ?? "—";
  const totalFrames = metrics?.total_frames?.toLocaleString() ?? "—";
  const modelName = metrics?.model_name ?? "—";
  const upscaleFactor = metrics?.upscale_factor ?? "—";
  const inputRes = metrics?.input_resolution ?? "—";
  const outputRes = metrics?.output_resolution ?? "—";
  const uptime = metrics?.uptime_seconds
    ? formatUptime(metrics.uptime_seconds)
    : "—";
  const avgLrKb = metrics?.avg_lr_size_kb?.toFixed(1) ?? "—";
  const avgHrKb = metrics?.avg_hr_size_kb?.toFixed(1) ?? "—";

  return (
    <div className="w-full">
      {/* ── Section Header ── */}
      <div className="flex items-center gap-3 mb-5">
        <div className="h-px flex-1 bg-gradient-to-r from-neon-magenta/30 to-transparent" />
        <h2 className="font-display font-semibold text-sm tracking-[0.2em] uppercase text-gray-400">
          Telemetry
        </h2>
        <div className="h-px flex-1 bg-gradient-to-l from-neon-cyan/30 to-transparent" />
      </div>

      {/* ── Primary Metrics Grid ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-4 mb-4">
        <MetricCard
          animateId="metric-fps"
          label="FPS"
          value={fps}
          unit="frames/s"
          color="#00f0ff"
          icon="⚡"
          sparklineData={fpsHistory}
          subtext="Real-time throughput"
        />
        <MetricCard
          animateId="metric-inference"
          label="Inference"
          value={inferenceMs}
          unit="ms"
          color="#ff2d7b"
          icon="🧠"
          sparklineData={inferenceHistory}
          subtext="Per-frame latency"
        />
        <MetricCard
          animateId="metric-bandwidth"
          label="BW Saved"
          value={bandwidthSaved}
          unit="%"
          color="#ffe156"
          icon="📡"
          subtext={`LR: ${avgLrKb} KB → HR: ${avgHrKb} KB`}
        />
        <MetricCard
          animateId="metric-frames"
          label="Processed"
          value={totalFrames}
          unit="frames"
          color="#b24dff"
          icon="🎬"
          subtext={`Uptime: ${uptime}`}
        />
      </div>

      {/* ── Secondary Info Bar ── */}
      <div className="glass-card p-3 lg:p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 lg:gap-6">
          {/* ── Model Info ── */}
          <div className="flex items-center gap-4">
            <InfoChip
              label="MODEL"
              value={modelName}
              color="#00f0ff"
            />
            <InfoChip
              label="SCALE"
              value={`${upscaleFactor}×`}
              color="#ff2d7b"
            />
          </div>

          {/* ── Resolution Pipeline ── */}
          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-neon-magenta">{inputRes}</span>
            <svg
              className="w-4 h-4 text-gray-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M14 5l7 7m0 0l-7 7m7-7H3"
              />
            </svg>
            <span className="text-gray-500 font-semibold">AI</span>
            <svg
              className="w-4 h-4 text-gray-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M14 5l7 7m0 0l-7 7m7-7H3"
              />
            </svg>
            <span className="text-neon-cyan">{outputRes}</span>
          </div>

          {/* ── Uptime ── */}
          <div className="flex items-center gap-2">
            <span className="text-gray-600 text-xs font-mono">UPTIME</span>
            <span className="text-gray-400 text-xs font-mono font-semibold">
              {uptime}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * InfoChip — Small labeled value badge.
 */
function InfoChip({ label, value, color }) {
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-xs text-gray-600 tracking-wider">
        {label}
      </span>
      <span
        className="font-mono text-sm font-bold px-2 py-0.5 rounded"
        style={{
          color,
          backgroundColor: `${color}10`,
          border: `1px solid ${color}25`,
        }}
      >
        {value}
      </span>
    </div>
  );
}

/**
 * Format seconds into HH:MM:SS display.
 */
function formatUptime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);

  if (h > 0) {
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }
  return `${m}:${String(s).padStart(2, "0")}`;
}
