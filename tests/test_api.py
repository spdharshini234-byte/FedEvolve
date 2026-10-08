import io
from fastapi.testclient import TestClient
from PIL import Image


def client(tiny_project):
    from backend.app.main import app
    return TestClient(app)


def test_health_and_status(tiny_project):
    c = client(tiny_project)
    assert c.get("/api/health").json() == {"status": "ok"}
    s = c.get("/api/status").json()
    assert s["num_factories"] == 3 and s["num_classes"] == 4 and s["best_model"]["accuracy"] >= 0


def test_clients_comparison_confusion(tiny_project):
    c = client(tiny_project)
    cl = c.get("/api/clients").json()["clients"]
    assert len(cl) == 3 and sum(x["num_images"] for x in cl) > 0
    assert len(c.get("/api/comparison").json()["rows"]) == 5
    assert c.get("/api/confusion-matrix/fedavg").json()["confusion_matrix"]["labels"]
    assert c.get("/api/confusion-matrix/does_not_exist").status_code == 404
    assert len(c.get("/api/federated/rounds?experiment=fedavg").json()["history"]) == 2
    assert c.get("/api/continual/status").json()["available"] is True
    assert c.get("/api/privacy").json()["raw_images_transferred"] == 0


def test_prediction_and_training_endpoints(tiny_project):
    c = client(tiny_project)
    buf = io.BytesIO()
    Image.new("RGB", (50, 50), (120, 120, 120)).save(buf, format="PNG")
    r = c.post("/api/predict", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and r.json()["predicted_class"] in ["scratch", "crack", "dent", "color_defect"]
    assert c.post("/api/predict", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
    assert c.get("/api/training/status").status_code == 200
    assert c.post("/api/training/start", json={"experiment": "bogus"}).status_code == 400
