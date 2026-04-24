import React, { useMemo, useState } from "react";
import { generateReport, predictSignal, uploadSignalFile } from "./api/client";
import ECGChart from "./components/ECGChart";
import ExplanationPanel from "./components/ExplanationPanel";
import PredictionBreakdown from "./components/PredictionBreakdown";
import RiskGauge from "./components/RiskGauge";
import StatusCard from "./components/StatusCard";

function parseCsvLike(text) {
  return text
    .split(/[\n,;\t ]+/)
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v));
}

function makeDemoSignal(length = 900) {
  const arr = [];
  for (let i = 0; i < length; i += 1) {
    const t = i / 250;
    const beat = Math.sin(2 * Math.PI * 1.2 * t) * 0.1;
    const qrs = Math.exp(-((((t % 0.8) - 0.15) ** 2) / 0.0007)) * 1.1;
    const noise = (Math.random() - 0.5) * 0.03;
    arr.push(beat + qrs + noise);
  }
  return arr;
}

export default function App() {
  const [rawText, setRawText] = useState("");
  const [signal, setSignal] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [patientId, setPatientId] = useState("patient-001");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [reportInfo, setReportInfo] = useState(null);

  const canPredict = signal.length >= 100 && !busy;

  const previewStats = useMemo(() => {
    if (!signal.length) return null;
    const mean = signal.reduce((acc, v) => acc + v, 0) / signal.length;
    const variance = signal.reduce((acc, v) => acc + (v - mean) ** 2, 0) / signal.length;
    return {
      points: signal.length,
      mean,
      std: Math.sqrt(variance),
    };
  }, [signal]);

  const handleParseText = () => {
    const values = parseCsvLike(rawText);
    if (values.length < 100) {
      setError("Provide at least 100 numeric ECG points.");
      return;
    }
    setError("");
    setSignal(values);
  };

  const handleLoadDemo = () => {
    const demo = makeDemoSignal();
    setSignal(demo);
    setRawText(demo.slice(0, 200).join(","));
    setError("");
  };

  const handlePredict = async () => {
    if (!canPredict) return;
    setBusy(true);
    setError("");
    try {
      const result = await predictSignal(signal, 250);
      setPrediction(result);
      setReportInfo(null);
    } catch (e) {
      setError(e?.response?.data?.detail || "Prediction failed. Check backend server.");
    } finally {
      setBusy(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      await uploadSignalFile(file);
      const text = await file.text();
      setRawText(text.slice(0, 10000));
      const values = parseCsvLike(text);
      if (values.length >= 100) {
        setSignal(values);
      } else {
        setError("Uploaded file has fewer than 100 numeric values.");
      }
    } catch (e) {
      setError(e?.response?.data?.detail || "Upload failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleReport = async () => {
    if (!prediction) return;
    setBusy(true);
    setError("");
    try {
      const response = await generateReport(patientId, prediction, "Generated from dashboard.");
      setReportInfo(response);
    } catch (e) {
      setError(e?.response?.data?.detail || "Failed to generate report.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <h1>ECG Arrhythmia Intelligence Console</h1>
          <p>Clinical-grade AI screening dashboard with calibrated confidence and waveform explainability.</p>
        </div>
      </header>

      <section className="card input-card">
        <h3>Input ECG Signal</h3>
        <div className="input-row">
          <textarea
            value={rawText}
            onChange={(e) => setRawText(e.target.value)}
            placeholder="Paste ECG samples (CSV/space/newline separated)"
            rows={6}
          />
          <div className="input-actions">
            <label className="file-btn">
              Upload CSV/JSON
              <input type="file" accept=".csv,.json,.txt" onChange={handleFileUpload} />
            </label>
            <button onClick={handleParseText} disabled={busy}>
              Parse Signal
            </button>
            <button onClick={handleLoadDemo} disabled={busy}>
              Load Demo
            </button>
            <button className="primary" onClick={handlePredict} disabled={!canPredict}>
              {busy ? "Running..." : "Run Prediction"}
            </button>
          </div>
        </div>

        {previewStats && (
          <p className="input-meta">
            Parsed {previewStats.points} points · Mean {previewStats.mean.toFixed(3)} · SD{" "}
            {previewStats.std.toFixed(3)}
          </p>
        )}
        {error && <p className="error-text">{error}</p>}
      </section>

      {prediction && (
        <>
          <section className="grid-two">
            <StatusCard
              arrhythmia={prediction.arrhythmia}
              confidence={prediction.confidence}
              riskScore={prediction.risk_score}
            />
            <RiskGauge riskScore={prediction.risk_score} />
          </section>

          <section className="grid-two">
            <PredictionBreakdown topClasses={prediction.top_classes} />
            <ExplanationPanel
              explanationText={prediction.explanation_text}
              confidence={prediction.confidence}
              uncertainty={prediction.uncertainty}
              signalQuality={prediction.signal_quality}
            />
          </section>

          <ECGChart signal={signal} explanationMap={prediction.explanation_map} />

          <section className="card report-card">
            <h3>PDF Report</h3>
            <div className="report-row">
              <input value={patientId} onChange={(e) => setPatientId(e.target.value)} placeholder="Patient ID" />
              <button onClick={handleReport} disabled={busy}>
                Generate Report
              </button>
            </div>
            {reportInfo && (
              <p className="input-meta">
                Report created: <strong>{reportInfo.report_id}</strong> at <code>{reportInfo.report_path}</code>
              </p>
            )}
          </section>
        </>
      )}
    </main>
  );
}
