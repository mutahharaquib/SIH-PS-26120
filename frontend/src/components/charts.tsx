import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";

const COLORS = ["#38bdf8", "#f59e0b", "#a78bfa", "#34d399", "#f87171", "#e879f9"];

export function TimeChart({ rows, series, height = 200, refLines = [], yLabel }: {
  rows: any[]; series: { key: string; name?: string; color?: string; dash?: boolean }[]; height?: number;
  refLines?: { y: number; label: string; color?: string }[]; yLabel?: string;
}) {
  const data = rows.map((r) => ({ ...r, tt: typeof r.t === "string" ? r.t.slice(5, 13).replace("T", " ") : r.t }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#1e293b" />
        <XAxis dataKey="tt" tick={{ fontSize: 10, fill: "#94a3b8" }} minTickGap={40} />
        <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} width={48}
          label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 10 } : undefined} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {refLines.map((r) => (
          <ReferenceLine key={r.label} y={r.y} stroke={r.color ?? "#ef4444"} strokeDasharray="4 4"
            label={{ value: r.label, fill: r.color ?? "#ef4444", fontSize: 10, position: "right" }} />
        ))}
        {series.map((s, i) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.name ?? s.key} dot={false} isAnimationActive={false}
            stroke={s.color ?? COLORS[i % COLORS.length]} strokeDasharray={s.dash ? "5 4" : undefined} strokeWidth={1.6}
            connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function CardPlot({ cards, kind, height = 220 }: {
  cards: { t: string; surface: { position: number[]; load: number[] }; downhole: { position: number[]; load: number[] } }[];
  kind: "surface" | "downhole"; height?: number;
}) {
  const sets = cards.map((c) => {
    const d = c[kind];
    const pts = d.position.map((x, i) => ({ x, y: d.load[i] / 1000 }));
    return pts.length ? [...pts, pts[0]] : pts;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 5, right: 10, left: 0, bottom: 10 }}>
        <CartesianGrid stroke="#1e293b" />
        <XAxis type="number" dataKey="x" name="position" unit=" m" tick={{ fontSize: 10, fill: "#94a3b8" }} domain={["auto", "auto"]} />
        <YAxis type="number" dataKey="y" name="load" unit=" kN" tick={{ fontSize: 10, fill: "#94a3b8" }} width={52} domain={["auto", "auto"]} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
        {sets.map((s, i) => (
          <Scatter key={i} data={s} line={{ stroke: i === sets.length - 1 ? "#38bdf8" : "#475569", strokeWidth: i === sets.length - 1 ? 2 : 1 }}
            shape={() => <g />} isAnimationActive={false} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
