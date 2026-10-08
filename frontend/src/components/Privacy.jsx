import React from "react";
import { api } from "../services/api.js";
import { useApi, ErrorBox, Stat, Empty } from "./Common.jsx";

export default function Privacy() {
  const { data, error } = useApi(api.privacy, [], 5000);
  return (
    <>
      <h2>Privacy / Communication</h2><p className="sub">What travels between the factories and the server.</p>
      <ErrorBox error={error} />
      {data && (<>
        <div className="grid">
          <Stat label="Raw images transferred" value={data.raw_images_transferred} sub="never sent to the server" />
          <Stat label="Model updates transferred" value={data.model_updates_transferred ? "yes" : "not yet"} sub="weights only" />
        </div>
        <div className="card"><b>Approximate communication cost (real, measured from the model size)</b>
          {Object.keys(data.runs).length === 0 ? <Empty>Run a federated experiment first.</Empty> : (
            <table><thead><tr><th>Run</th><th>Rounds</th><th>Upload (MB)</th><th>Download (MB)</th><th>Total (MB)</th><th>Raw images (bytes)</th></tr></thead><tbody>
              {Object.values(data.runs).map((r) => <tr key={r.display_name}><td>{r.display_name}</td><td>{r.rounds}</td>
                <td>{(r.upload_bytes / 1048576).toFixed(1)}</td><td>{(r.download_bytes / 1048576).toFixed(1)}</td><td>{r.total_mb}</td><td>{r.raw_image_bytes_transferred}</td></tr>)}
            </tbody></table>)}
        </div>
        <div className="card"><b>Limitations (please read)</b><ul>{data.limitations.map((l) => <li key={l}>{l}</li>)}</ul></div>
      </>)}
    </>
  );
}
