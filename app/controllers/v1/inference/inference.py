from fastapi import APIRouter, HTTPException, Query
import asyncio
from app.services.inference.inference import trigger_inference

router = APIRouter()

@router.post("/infer")
async def infer_image(train_run_id: str = Query(...)):
    try:
        if not train_run_id or not isinstance(train_run_id, str):
            raise HTTPException(status_code=400, detail="train_run_id is required and must be a string")

        await asyncio.create_task(trigger_inference(train_run_id))
        
        return {"status": "success", "message": "Inference triggered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


