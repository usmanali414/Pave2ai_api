from fastapi import APIRouter, HTTPException
from app.services.train.train import start_training


router = APIRouter()


@router.post("/train/{train_config_id}")
async def start_train(train_config_id: str):
    try:
        return await start_training(train_config_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        # Training already running
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
