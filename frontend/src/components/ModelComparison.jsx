import React from "react";
import { api } from "../services/api.js";
import { useApi, ErrorBox, BarGroup, pct, Empty } from "./Common.jsx";

export default function ModelComparison() {
  const { data, error } = useApi(api.comparison, [], 5000);
  const rows = data?.rows || [];
  return (
    <>
      <h2>Model Comparison</h2>
      <p className="sub">Centralized vs Local Only vs FedAvg vs Federated Continual (plus the no-replay baseline). Same test set for all.</p>
      <ErrorBox error={error} />
      {rows.length === 0 ? <Empty>No results yet. Run experiments first.</Empty> : (<>
        <div className="card"><BarGroup names={["Accuracy", "Macro-F1"]} rows={rows.map((r) => ({ label: r.display_name, values: [r.accuracy, r.macro_f1] }))} /></div>
        <div className="card" style={{ overflowX: "auto" }}>
          <table><thead><tr><th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>Old classes</th><th>New class</th><th>Forgetting</th><th>Comm. (MB)</th></tr></thead><tbody>
            {rows.map((r) => <tr key={r.experiment}><td>{r.display_name}</td><td>{pct(r.accuracy)}</td><td>{pct(r.macro_f1)}</td>
              <td>{pct(r.old_class_accuracy)}</td><td>{pct(r.new_class_accuracy)}</td><td>{pct(r.forgetting_score)}</td><td>{r.communication_mb}</td></tr>)}
          </tbody></table>
          <p style={{ fontSize: 12, color: "#6b7385" }}>Old/new = classes of earlier tasks / the last task. Local Only metrics pool the predictions of the 3 local models. Forgetting only applies to sequential (continual) runs.</p>
        </div>
      </>)}
    </>
  );
}
