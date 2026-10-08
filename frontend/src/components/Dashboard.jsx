import React from "react";
import { api } from "../services/api.js";
import { useApi, Stat, ErrorBox, pct, DemoNotice } from "./Common.jsx";

export default function Dashboard() {
  const { data: s, error } = useApi(api.status, [], 3000);
  return (
    <>
      <h2>Dashboard</h2><p className="sub">Live summary read from the Python backend.</p>
      <ErrorBox error={error} />
      {s && (<>
        <DemoNotice dataset={s.dataset} />
        <div className="grid">
          <Stat label="Simulated factories" value={s.num_factories} />
          <Stat label="Federated round" value={`${s.current_round} / ${s.total_rounds || s.configured_rounds}`} sub={`state: ${s.training_state}`} />
          <Stat label="Global model" value={s.global_model_status} sub={s.best_model?.display_name} />
          <Stat label="Accuracy" value={pct(s.best_model?.accuracy)} />
          <Stat label="Macro-F1" value={pct(s.best_model?.macro_f1)} />
          <Stat label="Defect classes" value={s.num_classes} sub={s.classes.join(", ")} />
        </div>
        {!s.data_ready && <div className="notice">No prepared data yet. Open <b>Federated Training</b> and press “Prepare demo data”, or follow the README.</div>}
        {s.data_ready && s.experiments_available.length === 0 && <div className="notice">Data is ready. Start an experiment on the <b>Federated Training</b> page.</div>}
        <div className="card"><b>Experiments with saved results:</b> {s.experiments_available.join(", ") || "none yet"}</div>
      </>)}
    </>
  );
}
