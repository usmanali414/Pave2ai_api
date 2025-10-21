from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from app.services.train.train import start_training, resume_training, get_training_status, cancel_training, get_training_runs_for_config, get_all_training_runs

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


@router.post("/train/resume/{train_run_id}")
async def resume_train(train_run_id: str):
    try:
        return await resume_training(train_run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        # Cannot resume
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/status/{train_run_id}")
async def get_training_status_endpoint(train_run_id: str):
    """Get training status endpoint"""
    try:
        return await get_training_status(train_run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/cancel/{train_run_id}")
async def cancel_training_endpoint(train_run_id: str):
    """Cancel training endpoint"""
    try:
        return await cancel_training(train_run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train-runs")
async def get_all_training_runs_endpoint(
    project_id: Optional[str] = Query(None, description="Project ID to filter training runs"),
    status: Optional[str] = Query(None, description="Status to filter training runs")
):
    """Get all training runs with optional filtering"""
    try:
        return await get_all_training_runs(project_id, status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/runs/{train_config_id}")
async def get_training_runs_endpoint(train_config_id: str):
    """Get all training runs for a specific train config"""
    try:
        return await get_training_runs_for_config(train_config_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))