from fastapi import APIRouter, HTTPException
from ..services import analytics_service as an

router = APIRouter(tags=["experiments"])


@router.get("/experiments")
def experiments():
    return an.list_experiments()


@router.get("/experiments/{name}")
def experiment(name: str):
    try:
        return an.get_experiment(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@router.get("/comparison")
def comparison():
    return an.model_comparison()


@router.get("/confusion-matrix/{name}")
def confusion(name: str):
    try:
        return an.confusion(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
