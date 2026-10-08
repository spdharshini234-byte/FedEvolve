import React, { useState } from "react";
import { api } from "../services/api.js";
import { useApi, ErrorBox, LineChart, Progress, pct, Empty } from "./Common.jsx";

const EXPS = [["fedavg", "Train FedAvg"], ["federated_continual", "Train Federated Continual (+ no-replay baseline)"],
  ["centralized", "Train Centralized"], ["local_only", "Train Local Only"], ["full", "Run full experiment"]];

export default function TrainingRounds() {
  const { data: st, error, reload } = useApi(api.trainingStatus, [], 1500);
  const { data: rounds } = useApi(() => api.rounds("fedavg"), [], 3000);
  const [msg, setMsg] = useState(null);
  const running = st?.state === "running";

  const start = async (exp) => {
    setMsg(null);
    try { await api.startTraining(exp, true); reload(); } catch (e) { setMsg(e.message); }
  };
  const live = rounds?.live?.experiment === "fedavg" ? rounds.live.history : [];
  const hist = live.length ? live : rounds?.history || [];

  return (
    <>
      <h2>Federated Training</h2>
      <p className="sub">Start real training runs. The backend launches the Python pipeline and this page shows its live status.</p>
      <ErrorBox error={error || msg} />
      <div className="card">
        <button className="btn secondary" disabled={running} onClick={() => start("prepare_demo")}>Prepare demo data (synthetic)</button>
        {EXPS.map(([k, l]) => <button key={k} className="btn" disabled={running} onClick={() => start(k)}>{l}</button>)}
        <p style={{ fontSize: 12, color: "#6b7385", margin: 0 }}>On a laptop CPU each experiment takes a few minutes.</p>
      </div>
      {st && (
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <b>Status: <span className={`badge ${st.state === "completed" ? "good" : st.state === "failed" ? "bad" : "warn"}`}>{st.state}</span> {st.experiment}</b>
            <span>Round {st.current_round || 0} / {st.total_rounds || 0}</span>
          </div>
          <Progress value={st.total_rounds ? st.current_round / st.total_rounds : 0} />
          <p>{st.message}</p>
          <p style={{ margin: 0 }}><b>FedAvg status:</b> {st.fedavg_status}{st.current_task ? ` · ${st.current_task}` : ""}</p>
          <table style={{ marginTop: 10 }}>
            <thead><tr><th>Factory</th><th>Status</th><th>Local images</th><th>Last loss</th></tr></thead>
            <tbody>{Object.entries(st.clients || {}).map(([id, c]) => (
              <tr key={id}><td>{c.name}</td><td>{c.status}</td><td>{c.samples}</td><td>{c.loss != null ? c.loss.toFixed(4) : "–"}</td></tr>))}</tbody>
          </table>
          {st.log_tail?.length > 0 && <pre className="log">{st.log_tail.join("\n")}</pre>}
        </div>
      )}
      <div className="card">
        <b>FedAvg training history (validation accuracy of the global model)</b>
        {hist.length ? (<>
          <LineChart labels={hist.map((h) => h.round)} series={[{ name: "Accuracy", points: hist.map((h) => h.val_accuracy) }, { name: "Macro-F1", points: hist.map((h) => h.val_macro_f1) }]} />
          <table><thead><tr><th>Round</th><th>Val accuracy</th><th>Val macro-F1</th></tr></thead>
            <tbody>{hist.map((h) => <tr key={h.round}><td>{h.round}</td><td>{pct(h.val_accuracy)}</td><td>{pct(h.val_macro_f1)}</td></tr>)}</tbody></table>
        </>) : <Empty>No FedAvg run yet.</Empty>}
      </div>
    </>
  );
}
