import React, { useState } from "react";
import { api } from "../services/api.js";
import { useApi, ErrorBox, pct, Progress } from "./Common.jsx";

export default function Prediction() {
  const { data: models } = useApi(api.models, []);
  const [file, setFile] = useState(null);
  const [model, setModel] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true); setError(null); setResult(null);
    try { setResult(await api.predict(file, model || undefined)); } catch (e) { setError(e.message); }
    setBusy(false);
  };
  return (
    <>
      <h2>Prediction</h2><p className="sub">Upload a product image. The backend runs the trained ResNet-18.</p>
      <ErrorBox error={error} />
      <div className="card">
        <input type="file" accept="image/*" onChange={(e) => { setFile(e.target.files[0]); setResult(null); }} />
        <select value={model} onChange={(e) => setModel(e.target.value)} style={{ margin: "0 10px", padding: 8 }}>
          <option value="">Best available model</option>
          {models?.models.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
        <button className="btn" disabled={!file || busy} onClick={run}>{busy ? "Predicting…" : "Predict"}</button>
        {models && models.models.length === 0 && <p className="notice">No trained model yet. Train one first.</p>}
      </div>
      <div className="row">
        {file && <div className="card"><img alt="upload" src={URL.createObjectURL(file)} style={{ maxWidth: "100%", maxHeight: 280 }} /></div>}
        {result && (
          <div className="card">
            <div className="sub" style={{ margin: 0 }}>Predicted defect</div>
            <div style={{ fontSize: 30, fontWeight: 700 }}>{result.predicted_class}</div>
            <p>Confidence: <b>{pct(result.confidence)}</b> · Model: <b>{result.model_name}</b></p>
            {Object.entries(result.probabilities).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 6 }}><div style={{ fontSize: 13 }}>{k} – {pct(v)}</div><Progress value={v} /></div>))}
            {result.dataset_source && <p style={{ fontSize: 12, color: "#6b7385" }}>Model trained on: {result.dataset_source}</p>}
          </div>)}
      </div>
    </>
  );
}
