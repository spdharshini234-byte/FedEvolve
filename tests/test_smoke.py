"""Smoke test: the whole real pipeline ran (tiny settings) and produced real artifacts."""
import json


def test_pipeline_outputs(tiny_project):
    m = tiny_project / "results" / "metrics"
    for name in ["centralized", "local_only", "fedavg", "federated_no_replay", "federated_continual"]:
        r = json.loads((m / f"{name}.json").read_text())
        assert 0 <= r["overall"]["accuracy"] <= 1 and r["confusion_matrix"]["matrix"]
    assert (tiny_project / "models/global/federated_continual.pt").exists()
    assert (tiny_project / "results/experiment_results.csv").exists()


def test_federated_privacy_accounting(tiny_project):
    r = json.loads((tiny_project / "results/metrics/fedavg.json").read_text())
    assert r["communication"]["raw_image_bytes_transferred"] == 0
    assert r["communication"]["total_bytes"] > 0 and r["communication"]["rounds"] == 2


def test_continual_results(tiny_project):
    r = json.loads((tiny_project / "results/metrics/federated_continual.json").read_text())
    c = r["continual"]
    assert len(c["stages"]) == 3 and r["forgetting_score"] is not None
    assert c["stages"][-1]["replay_buffers"][0]["size"] > 0
    nr = json.loads((tiny_project / "results/metrics/federated_no_replay.json").read_text())
    assert all(b["size"] == 0 for b in nr["continual"]["stages"][-1]["replay_buffers"])
