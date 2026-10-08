import React from "react";
import { api } from "../services/api.js";
import { useApi, ErrorBox, LineChart, BarGroup, pct, Empty } from "./Common.jsx";

export default function ContinualLearning() {
  const { data, error } = useApi(api.continual, [], 4000);
  const rep = data?.runs?.federated_continual, base = data?.runs?.federated_no_replay;
  const ref = rep || base;
  const T = ref?.continual.tasks.length || 0;
  return (
    <>
      <h2>Continual Learning</h2>
      <p className="sub">New defect classes arrive one task at a time. Replay keeps a small local memory of old samples.</p>
      <ErrorBox error={error} />
      {!data?.available && <Empty>No continual run yet. Press “Train Federated Continual” on the Federated Training page.</Empty>}
      {ref && (<>
        <div className="grid">
          <div className="card stat"><div className="label">Existing (old) classes</div><div className="value" style={{ fontSize: 16 }}>{ref.old_new.old_classes.join(", ")}</div></div>
          <div className="card stat"><div className="label">New class (last task)</div><div className="value" style={{ fontSize: 16 }}>{ref.old_new.new_classes.join(", ")}</div></div>
          <div className="card stat"><div className="label">Forgetting – no replay</div><div className="value">{pct(base?.forgetting_score)}</div></div>
          <div className="card stat"><div className="label">Forgetting – with replay</div><div className="value">{pct(rep?.forgetting_score)}</div></div>
        </div>
        <div className="row">
          <div className="card"><b>Old-class accuracy after each task</b>
            <LineChart labels={ref.continual.tasks.map((_, i) => `T${i + 1}`)} series={[base, rep].filter(Boolean).map((r) => ({ name: r.display_name, points: r.continual.stages.map((s) => s.old_class_accuracy) }))} /></div>
          <div className="card"><b>Final old vs new class accuracy</b>
            <BarGroup names={["Old classes", "New class"]} rows={[base, rep].filter(Boolean).map((r) => ({ label: r.display_name, values: [r.old_new.old_class_accuracy, r.old_new.new_class_accuracy] }))} /></div>
        </div>
        {rep && (<div className="card"><b>Replay buffer (per factory, after each task)</b>
          <table><thead><tr><th>After</th><th>Factory</th><th>Samples remembered</th><th>Per class</th></tr></thead><tbody>
            {rep.continual.stages.flatMap((s) => s.replay_buffers.map((b) => (
              <tr key={s.task_name + b.client}><td>{s.task_name}</td><td>{b.client}</td><td>{b.size}</td><td>{Object.entries(b.per_class).map(([k, v]) => `${k}: ${v}`).join(", ") || "–"}</td></tr>)))}
          </tbody></table>
          <p style={{ fontSize: 12, color: "#6b7385" }}>The buffer stores references to local images; they never leave the factory.</p></div>)}
        <div className="card"><b>Accuracy matrix (rows: after learning task; columns: accuracy on task)</b>
          {[base, rep].filter(Boolean).map((r) => (
            <div key={r.display_name}><p style={{ margin: "10px 0 4px" }}>{r.display_name}</p>
              <table><thead><tr><th></th>{Array.from({ length: T }, (_, j) => <th key={j}>Task {j + 1}</th>)}</tr></thead><tbody>
                {r.continual.accuracy_matrix.map((row, t) => <tr key={t}><td>After task {t + 1}</td>{row.map((v, j) => <td key={j}>{pct(v)}</td>)}</tr>)}
              </tbody></table></div>))}
        </div>
      </>)}
    </>
  );
}
