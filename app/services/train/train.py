from datetime import datetime
from typing import Any, Dict

from bson import ObjectId
from app.database.conn import mongo_client
from config import database_config
from app.services.s3.s3_operations import S3Operations


async def start_training(train_config_id: str) -> Dict[str, Any]:
    """Fetch train_config and create a train_run with status 'training'.

    Prevents duplicate concurrent runs for the same config.
    """
    db = mongo_client.database
    train_configs = db[database_config["TRAIN_CONFIG_COLLECTION"]]
    train_runs = db[database_config["TRAIN_RUN_COLLECTION"]]

    # Validate and fetch train_config by ObjectId
    try:
        cfg_oid = ObjectId(train_config_id)
    except Exception:
        raise ValueError("invalid train_config_id")

    config = await train_configs.find_one({"_id": cfg_oid})
    if not config:
        raise ValueError("train_config not found")

    # Check if a training run is already active for this config
    existing_run = await train_runs.find_one({
        "train_config_id": train_config_id,
        "status": "training",
        "ended_at": None,
    })
    if existing_run:
        # Indicate training already running
        raise RuntimeError("training already running for this configuration")

    now = datetime.utcnow()
    run_doc = {
        "train_config_id": train_config_id,
        "status": "training",
        "created_at": now,
        "updated_at": now,
        "ended_at": None,
    }
    result = await train_runs.insert_one(run_doc)
    run_id = str(result.inserted_id)

    # Simulated training workflow: create dummy model weights, upload to input path,
    # then copy to output path.
    try:
        tenant_id: str = config.get("tenant_id")
        project_id: str = config.get("project_id")

        # Fetch bucket config for project to get training paths
        bucket_configs = db[database_config["BUCKET_CONFIG_COLLECTION"]]
        bucket_cfg = await bucket_configs.find_one({"project_id": project_id})
        if not bucket_cfg:
            raise RuntimeError("bucket_config not found for project")

        fs = (bucket_cfg.get("folder_structure") or {})
        input_base: str = fs.get("train_input_model")
        output_base: str = fs.get("train_output_model")
        if not input_base or not output_base:
            raise RuntimeError("training paths not configured in bucket_config")

        # Ensure trailing slash
        if not input_base.endswith("/"):
            input_base += "/"
        if not output_base.endswith("/"):
            output_base += "/"

        filename = "dummy_model_weights.h5"
        input_url = f"{input_base}{filename}"
        output_url = f"{output_base}{filename}"

        # If config.metadata.initial_weights is true, expect file already present in input path
        # Otherwise, generate/upload a dummy file like in example_usage.py
        should_use_existing = False
        try:
            meta = config.get("metadata") or {}
            should_use_existing = bool(meta.get("initial_weights"))
        except Exception:
            should_use_existing = False

        if not should_use_existing:
            file_path: str | None = None
            try:
                from tensorflow import keras  # type: ignore

                model = keras.Sequential([
                    keras.layers.Dense(8, activation='relu', input_shape=(4,)),
                    keras.layers.Dense(3, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='categorical_crossentropy')
                file_path = "dummy_model_weights.h5"
                model.save_weights(file_path)
                upload_res = S3Operations.upload_file(
                    file_data=file_path,
                    s3_url=input_url,
                    content_type="application/octet-stream",
                    metadata={"model": "keras", "type": "weights"}
                )
            except Exception:
                # Fallback: small placeholder bytes
                dummy_bytes = b"DUMMY_WEIGHTS_V1"
                upload_res = S3Operations.upload_file(
                    file_data=dummy_bytes,
                    s3_url=input_url,
                    content_type="application/octet-stream",
                    metadata={"model": "dummy", "type": "weights"}
                )
            finally:
                try:
                    if file_path:
                        import os
                        if os.path.exists(file_path):
                            os.remove(file_path)
                except Exception:
                    pass

            if not upload_res.get("success"):
                raise RuntimeError(f"upload to input failed: {upload_res.get('error')}")

        # Copy input file to output path to simulate training artifact generation
        copy_res = S3Operations.copy_file(source_s3_url=input_url, destination_s3_url=output_url, overwrite=True)
        if not copy_res.get("success"):
            raise RuntimeError(f"copy to output failed: {copy_res.get('error')}")

        # Mark run completed
        await train_runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "completed", "updated_at": datetime.utcnow(), "ended_at": datetime.utcnow()}}
        )
        return {"train_run_id": run_id, "status": "completed", "input_url": input_url, "output_url": output_url}

    except Exception as e:
        # Update run as failed
        await train_runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(e), "updated_at": datetime.utcnow(), "ended_at": datetime.utcnow()}}
        )
        raise


