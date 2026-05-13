import React from "react";

function StatusCard({ arrhythmia, confidence, riskScore }) {
  const title = arrhythmia ? "Possible Arrhythmia" : "Normal Rhythm";
  const statusClass = arrhythmia ? "status-alert" : "status-normal";
  return (
    <section className={`card status-card ${statusClass}`}>
      <h2>{title}</h2>
      <p className="status-sub">
        Confidence {(confidence * 100).toFixed(1)}% · Risk {(riskScore * 100).toFixed(1)}%
      </p>
    </section>
  );
}

export default StatusCard;

