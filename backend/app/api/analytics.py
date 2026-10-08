from fastapi import APIRouter
from ..services import analytics_service as an

router = APIRouter(tags=["analytics"])


@router.get("/status")
def status():
    return an.system_status()


@router.get("/continual/status")
def continual():
    return an.continual_status()


@router.get("/privacy")
def privacy():
    return an.privacy_summary()
