import json
import sys
import os
from typing import Any, Dict
import pathlib, sys, os
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from config import API_BASE_URL
except ImportError:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"POST {url} failed {resp.status_code}: {resp.text}")
    return resp.json()


def seed_customer() -> Dict[str, Any]:
    suffix = uuid.uuid4().hex[:8]
    payload = {
        "customer_name": f"Acme Roads {suffix}",
        "email": f"ops+{suffix}@acmeroads.example",
        "description": "Pilot customer for AMCNN",
    }
    return post("/customer", payload)


def seed_model_config(customer_id: str) -> Dict[str, Any]:
    payload = {
        "customer_id": customer_id,
        "name": "amcnn",
        "version": 1,
        "status": "draft",
        "description": "Initial AMCNN config",
        "model_path": "/models/amcnn/v1",
        "model_name": "amcnn_v1.pt",
        "classes": [
            {"id": "0", "name": "background", "color_bgr": [0, 0, 0], "is_background": True},
            {"id": "1", "name": "crack", "color_bgr": [0, 0, 255]},
        ],
        "labels_dict": {"background": 0, "crack": 1},
    }
    return post("/model-config", payload)


def seed_bucket_config(customer_id: str) -> Dict[str, Any]:
    payload = {
        "customer_id": customer_id,
        "provider": "local",
        "bucket_name": "amcnn-datasets",
        "region": None,
        "endpoint_url": None,
        "access_key_id": "dummy_access_key",
        "secret_access_key": "dummy_secret",
        "dataset_path": "./dataset",
    }
    return post("/bucket-config", payload)


def main() -> None:
    print("Seeding sample data to API:", API_BASE_URL)

    customer = seed_customer()
    print("Customer:", json.dumps(customer, indent=2))

    model_cfg = seed_model_config(customer_id=customer["_id"]) if "_id" in customer else seed_model_config(customer_id=customer["id"]) 
    print("Model Config:", json.dumps(model_cfg, indent=2))

    bucket_cfg = seed_bucket_config(customer_id=customer["_id"]) if "_id" in customer else seed_bucket_config(customer_id=customer["id"]) 
    print("Bucket Config:", json.dumps(bucket_cfg, indent=2))

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)


