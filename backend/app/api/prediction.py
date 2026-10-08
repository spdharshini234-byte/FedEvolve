from typing import Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from ..schemas import PredictionResponse
from ..services import prediction_service

router = APIRouter(tags=["prediction"])


@router.get("/models")
def models():
    return prediction_service.list_models()


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model: Optional[str] = None):
    data = await file.read()
    try:
        return prediction_service.predict(data, model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
