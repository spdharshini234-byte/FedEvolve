from fastapi import APIRouter, HTTPException
from ..schemas import TrainRequest
from ..services import federated_service as fs

router = APIRouter(tags=["federated"])


@router.get("/clients")
def clients():
    return fs.clients_info()


@router.post("/training/start")
def start(req: TrainRequest):
    try:
        return fs.start_training(req.experiment, req.demo)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@router.get("/training/status")
def status():
    return fs.training_status()


@router.get("/federated/rounds")
def rounds(experiment: str = "fedavg"):
    return fs.federated_rounds(experiment)
