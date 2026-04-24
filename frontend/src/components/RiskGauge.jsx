import React from "react";

function RiskGauge({ riskScore }) {
  const pct = Math.max(0, Math.min(100, riskScore * 100));
  const hue = 140 - (pct * 1.4);

  return (
    <section className="card gauge-card">
      <h3>Risk Gauge</h3>
      <div className="gauge-wrap">
        <div
          className="gauge"
          style={{
            background: `conic-gradient(hsl(${hue}, 78%, 45%) ${pct * 3.6}deg, #d5dbe4 0deg)`,
          }}
        >
          <div className="gauge-inner">
            <span>{pct.toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </section>
  );
}

export default RiskGauge;

