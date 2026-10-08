import React, { useEffect, useState, useCallback } from "react";

export const pct = (v) => (v === null || v === undefined ? "–" : (v * 100).toFixed(1) + "%");
export const COLORS = ["#4f46e5", "#16a34a", "#d97706", "#dc2626", "#0891b2", "#9333ea"];

// Loads data from the backend (optionally polling) and exposes loading / error state.
export function useApi(fn, deps = [], pollMs = 0) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const load = useCallback(() => fn().then((d) => { setData(d); setError(null); }).catch((e) => setError(e.message)), deps);
  useEffect(() => {
    load();
    if (!pollMs) return;
    const t = setInterval(load, pollMs);
    return () => clearInterval(t);
  }, [load, pollMs]);
  return { data, error, reload: load };
}

export const Stat = ({ label, value, sub }) => (
  <div className="card stat"><div className="label">{label}</div><div className="value">{value}</div>{sub && <div className="sub" style={{margin:0}}>{sub}</div>}</div>
);
export const ErrorBox = ({ error }) => (error ? <div className="error">{error}</div> : null);
export const Empty = ({ children }) => <div className="card empty">{children}</div>;
export const DemoNotice = ({ dataset }) =>
  dataset && dataset.is_demo ? (
    <div className="notice">Results use a <b>synthetic demo dataset</b> (drawn images). They show that the pipeline works, not real factory performance.</div>
  ) : null;
export const Progress = ({ value }) => <div className="bar"><div style={{ width: `${Math.round((value || 0) * 100)}%` }} /></div>;

// Simple SVG line chart. series = [{name, points:[number|null]}]
export function LineChart({ series, labels, height = 180, yMax = 1 }) {
  const w = 520, pad = 28;
  const n = Math.max(1, labels.length - 1);
  const x = (i) => pad + (i * (w - 2 * pad)) / n;
  const y = (v) => height - pad - (v / yMax) * (height - 2 * pad);
  return (
    <div>
      <svg viewBox={`0 0 ${w} ${height}`} style={{ width: "100%" }}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}><line x1={pad} x2={w - pad} y1={y(t * yMax)} y2={y(t * yMax)} stroke="#e4e7ef" />
            <text x={2} y={y(t * yMax) + 4} fontSize="10" fill="#6b7385">{Math.round(t * 100)}%</text></g>
        ))}
        {labels.map((l, i) => <text key={i} x={x(i)} y={height - 8} fontSize="10" textAnchor="middle" fill="#6b7385">{l}</text>)}
        {series.map((s, si) => {
          const pts = s.points.map((v, i) => (v === null || v === undefined ? null : [x(i), y(v)]));
          const d = pts.filter(Boolean).map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" ");
          return <g key={s.name}><path d={d} fill="none" stroke={COLORS[si % COLORS.length]} strokeWidth="2.5" />
            {pts.filter(Boolean).map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="3.5" fill={COLORS[si % COLORS.length]} />)}</g>;
        })}
      </svg>
      <div className="legend">{series.map((s, i) => <span key={s.name}><i className="dot" style={{ background: COLORS[i % COLORS.length] }} />{s.name}</span>)}</div>
    </div>
  );
}

// Horizontal grouped bars. rows = [{label, values:[{name,value}]}]
export function BarGroup({ rows, names }) {
  return (
    <div>
      <div className="legend">{names.map((n, i) => <span key={n}><i className="dot" style={{ background: COLORS[i] }} />{n}</span>)}</div>
      {rows.map((r) => (
        <div key={r.label} style={{ margin: "10px 0" }}>
          <div style={{ fontSize: 13, marginBottom: 3 }}>{r.label}</div>
          {r.values.map((v, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
              <div className="bar" style={{ flex: 1 }}><div style={{ width: `${(v ?? 0) * 100}%`, background: COLORS[i] }} /></div>
              <span style={{ width: 52, fontSize: 12 }}>{pct(v)}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
