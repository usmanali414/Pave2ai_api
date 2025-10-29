from fastapi import APIRouter, HTTPException, Query
import asyncio
from app.services.inference.inference import trigger_inference

router = APIRouter()

@router.post("/infer")
async def infer_image(
    train_run_id: str = Query("68fee878d7b7be2bc7a3f501"),
    train_config_id: str = Query("68f7e4253d12fbbeeae2d9c2")):
    try:
        if not train_run_id and not isinstance(train_run_id, str) and not train_config_id and not isinstance(train_config_id, str):
            raise Exception("train_run_id and train_config_id are required and must be a string")

        await asyncio.create_task(trigger_inference(train_run_id, train_config_id))
        
        return {"status": "success", "message": "Inference triggered successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


