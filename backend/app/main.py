"""FastAPI application.  Start from the project root:  uvicorn backend.app.main:app --reload --port 8000"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import analytics, experiments, federated, prediction

app = FastAPI(title="Federated Continual Defect Detection API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
for r in (analytics.router, federated.router, experiments.router, prediction.router):
    app.include_router(r, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
