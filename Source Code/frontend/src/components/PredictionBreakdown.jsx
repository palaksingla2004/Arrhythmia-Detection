import React from "react";

function PredictionBreakdown({ topClasses }) {
  return (
    <section className="card">
      <h3>Prediction Breakdown</h3>
      <div className="breakdown-list">
        {topClasses.map((entry) => (
          <div key={entry.label} className="breakdown-row">
            <div className="breakdown-header">
              <span>{entry.label}</span>
              <span>{(entry.probability * 100).toFixed(1)}%</span>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${Math.max(2, entry.probability * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default PredictionBreakdown;

