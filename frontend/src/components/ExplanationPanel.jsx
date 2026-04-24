import React from "react";

function ExplanationPanel({ explanationText, confidence, uncertainty, signalQuality }) {
  const lowConfidence = confidence < 0.6 || uncertainty > 0.2 || signalQuality !== "Good";
  return (
    <section className="card explanation-card">
      <h3>Explanation & Confidence</h3>
      <p>{explanationText}</p>
      <p className="mini-stats">
        Confidence {(confidence * 100).toFixed(1)}% · Uncertainty {(uncertainty * 100).toFixed(1)}% · Quality{" "}
        {signalQuality}
      </p>
      {lowConfidence ? (
        <div className="warning-box">
          Confidence is limited. Please consult a medical professional before making decisions.
        </div>
      ) : (
        <div className="ok-box">Model confidence is acceptable for screening support.</div>
      )}
    </section>
  );
}

export default ExplanationPanel;

