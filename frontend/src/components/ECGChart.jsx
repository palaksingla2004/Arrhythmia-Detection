import React, { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function ECGChart({ signal, explanationMap }) {
  const chartData = useMemo(() => {
    if (!signal.length) return [];
    const exp = explanationMap?.length === signal.length ? explanationMap : signal.map(() => 0);
    return signal.map((value, index) => {
      const importance = exp[index] ?? 0;
      return {
        index,
        value,
        hot: importance > 0.65 ? value : null,
        cool: importance <= 0.65 ? value : null,
      };
    });
  }, [signal, explanationMap]);

  return (
    <section className="card chart-card">
      <h3>ECG Waveform + Abnormal Highlights</h3>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="4 6" stroke="rgba(30, 70, 120, 0.2)" />
            <XAxis dataKey="index" stroke="#335a7f" />
            <YAxis stroke="#335a7f" />
            <Tooltip />
            <Line type="monotone" dataKey="cool" stroke="#157a6e" dot={false} strokeWidth={1.4} />
            <Line type="monotone" dataKey="hot" stroke="#d9472f" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

export default ECGChart;

