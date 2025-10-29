import os
from typing import Dict, Any
from app.deep_models.data_parser.RIEGL_PARSER.config import RIEGL_PARSER_CONFIG

def _normalize_name(s: str) -> str:
    return (s or "").strip().replace(" ", "_")

def _get_model_namespace_from_config(train_config: Dict[str, Any]) -> str:
    ls = RIEGL_PARSER_CONFIG["local_storage"]
    md = (train_config.get("metadata") or {})
    model_name = _normalize_name(md.get("model_name") or ls.get("default_model") or "AMCNN")
    model_version = _normalize_name(train_config.get("model_version") or ls.get("default_version") or "v1")
    base = ls["base_path"]  # e.g. "static"
    # static/AMCNN/v1
    return os.path.join(base, model_name, model_version)