// All data shown in the web app comes from the FastAPI backend (real experiment results).
const BASE = import.meta.env.VITE_API_URL || "/api";

async function request(path, options) {
  let res;
  try {
    res = await fetch(BASE + path, options);
  } catch (e) {
    throw new Error("Cannot reach the backend. Start it with: uvicorn backend.app.main:app --port 8000");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  status: () => request("/status"),
  clients: () => request("/clients"),
  trainingStatus: () => request("/training/status"),
  startTraining: (experiment, demo = true) =>
    request("/training/start", { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify({ experiment, demo }) }),
  rounds: (experiment = "fedavg") => request(`/federated/rounds?experiment=${experiment}`),
  continual: () => request("/continual/status"),
  comparison: () => request("/comparison"),
  confusion: (name) => request(`/confusion-matrix/${name}`),
  privacy: () => request("/privacy"),
  models: () => request("/models"),
  predict: (file, model) => {
    const form = new FormData();
    form.append("file", file);
    return request("/predict" + (model ? `?model=${model}` : ""), { method: "POST", body: form });
  },
};
