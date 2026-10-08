import React from "react";
import { api } from "../services/api.js";
import { useApi, ErrorBox, COLORS, Empty } from "./Common.jsx";

export default function ClientStatus() {
  const { data, error } = useApi(api.clients, [], 3000);
  return (
    <>
      <h2>Factory Clients</h2>
      <p className="sub">Each factory keeps its own images. Distributions are intentionally different (non-IID).</p>
      <ErrorBox error={error} />
      {data && data.clients.every((c) => c.ready === false) && <Empty>Clients not created yet. Run “Prepare demo data” on the Federated Training page.</Empty>}
      <div className="row">
        {data?.clients.filter((c) => c.ready !== false).map((c) => {
          const total = c.num_images || 1;
          return (
            <div className="card" key={c.client_id}>
              <h3 style={{ margin: 0 }}>{c.name}</h3>
              <p className="sub" style={{ margin: "4px 0 12px" }}>{c.num_images} local training images · <span className="badge">{c.training_status}</span></p>
              {Object.entries(c.class_distribution).map(([cls, n], i) => (
                <div key={cls} style={{ marginBottom: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}><span>{cls}</span><span>{n} ({Math.round((100 * n) / total)}%)</span></div>
                  <div className="bar"><div style={{ width: `${(100 * n) / total}%`, background: COLORS[i % COLORS.length] }} /></div>
                </div>
              ))}
              <p style={{ fontSize: 12, color: "#6b7385" }}>Raw images leave this factory: <b>no</b></p>
            </div>
          );
        })}
      </div>
    </>
  );
}
