from typing import Optional
from pydantic import BaseModel


class TrainRequest(BaseModel):
    # centralized | local_only | fedavg | federated_continual | full
    experiment: str = "fedavg"
    demo: bool = True   # auto-prepare the synthetic demo dataset if data is missing


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    model_name: str
    probabilities: dict
    dataset_source: Optional[str] = None
