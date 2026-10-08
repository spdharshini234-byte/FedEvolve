import React, { useState } from "react";
import { api } from "../services/api.js";
import { useApi, ErrorBox, pct, Empty } from "./Common.jsx";

const NAMES = [["federated_continual", "Federated Continual"], ["federated_no_replay", "No replay"], ["fedavg", "FedAvg"], ["centralized", "Centralized"], ["local_only", "Local Only"]];

export default function ConfusionMatrix() {
  const [exp, setExp] = useState("federated_continual");
  const { data, error } = useApi(() => api.confusion(exp), [exp]);
  const cm = data?.confusion_matrix;
  const max = cm ? Math.max(1, ...cm.matrix.flat()) : 1;
  return (
    <>
      <h2>Confusion Matrix</h2><p className="sub">Generated from the real test-set predictions of the saved model.</p>
      <div className="card">{NAMES.map(([k, l]) => <button key={k} className={`btn ${k === exp ? "" : "secondary"}`} onClick={() => setExp(k)}>{l}</button>)}</div>
      <ErrorBox error={error} />
      {cm ? (<div className="row">
        <div className="card"><table className="cm"><thead><tr><th>true ↓ / pred →</th>{cm.labels.map((l) => <th key={l}>{l}</th>)}</tr></thead><tbody>
          {cm.matrix.map((row, i) => <tr key={i}><th>{cm.labels[i]}</th>{row.map((v, j) =>
            <td key={j} style={{ background: `rgba(79,70,229,${v / max})`, color: v / max > 0.5 ? "#fff" : "#1c2333" }}>{v}</td>)}</tr>)}
        </tbody></table></div>
        <div className="card"><b>Per-class performance</b><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>n</th></tr></thead><tbody>
          {data.per_class.map((c) => <tr key={c.class}><td>{c.class}</td><td>{pct(c.precision)}</td><td>{pct(c.recall)}</td><td>{pct(c.f1)}</td><td>{c.support}</td></tr>)}
        </tbody></table></div></div>) : !error && <Empty>Loading…</Empty>}
    </>
  );
}
