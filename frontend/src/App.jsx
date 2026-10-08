import React, { useState } from "react";
import Dashboard from "./components/Dashboard.jsx";
import ClientStatus from "./components/ClientStatus.jsx";
import TrainingRounds from "./components/TrainingRounds.jsx";
import ContinualLearning from "./components/ContinualLearning.jsx";
import ModelComparison from "./components/ModelComparison.jsx";
import Prediction from "./components/Prediction.jsx";
import ConfusionMatrix from "./components/ConfusionMatrix.jsx";
import Privacy from "./components/Privacy.jsx";

const PAGES = [
  ["Dashboard", Dashboard], ["Factory Clients", ClientStatus], ["Federated Training", TrainingRounds],
  ["Continual Learning", ContinualLearning], ["Model Comparison", ModelComparison], ["Prediction", Prediction],
  ["Confusion Matrix", ConfusionMatrix], ["Privacy / Communication", Privacy],
];

export default function App() {
  const [page, setPage] = useState(0);
  const Page = PAGES[page][1];
  return (
    <div className="layout">
      <nav className="sidebar">
        <h1>Federated Continual<br />Defect Detection</h1>
        {PAGES.map(([name], i) => (
          <button key={name} className={i === page ? "active" : ""} onClick={() => setPage(i)}>{name}</button>
        ))}
        <p style={{ fontSize: 11, margin: "20px 8px", color: "#8b94ad" }}>Factories are simulated software clients.</p>
      </nav>
      <main className="content"><Page /></main>
    </div>
  );
}
