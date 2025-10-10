from typing import Any, Dict

from app.database.conn import mongo_client
from app.utils.logger_utils import logger
from config import database_config


def _project_validator() -> Dict[str, Any]:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["name", "tenant_id", "user_id", "status", "created_at", "updated_at"],
            "properties": {
                "name": {"bsonType": "string"},
                "tenant_id": {"bsonType": "string"},
                "user_id": {"bsonType": "string"},
                "status": {"bsonType": "string"},
                "metadata": {"bsonType": ["object", "null"]},
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
            },
        }
    }


def _train_config_validator() -> Dict[str, Any]:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "name",
                "tenant_id",
                "project_id",
                "metadata",
                "model_version",
                "created_at",
                "updated_at",
            ],
            "properties": {
                "name": {"bsonType": "string"},
                "tenant_id": {"bsonType": "string"},
                "project_id": {"bsonType": "string"},
                "model_version": {"bsonType": "string"},
                # metadata contains data_parser, model_name, initial_weights
                "metadata": {
                    "bsonType": "object",
                    "properties": {
                        "data_parser": {"bsonType": ["string", "null"]},
                        "model_name": {"bsonType": ["string", "null"]},
                        "initial_weights": {"bsonType": ["bool", "null"]},
                    },
                },
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
                "deleted_at": {"bsonType": ["date", "null"]},
            },
        }
    }


def _bucket_config_validator() -> Dict[str, Any]:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "tenant_id",
                "project_id",
                "status",
                "created_at",
                "updated_at",
            ],
            "properties": {
                "tenant_id": {"bsonType": "string"},
                "project_id": {"bsonType": "string"},
                "folder_structure": {"bsonType": ["object", "null"]},
                "status": {"bsonType": "string"},
                "metadata": {"bsonType": ["object", "null"]},
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
            },
        }
    }


def _dataset_config_validator() -> Dict[str, Any]:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "project_id",
                "classes",
                "labels_dict",
                "created_at",
                "updated_at",
            ],
            "properties": {
                "project_id": {"bsonType": "string"},
                "classes": {"bsonType": "array"},
                "labels_dict": {"bsonType": "object"},
                "metadata": {"bsonType": ["object", "null"]},
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
                "deleted_at": {"bsonType": ["date", "null"]},
            },
        }
    }


def _model_config_validator() -> Dict[str, Any]:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "project_id",
                "name",
                "status",
                "model_path",
                "model_name",
                "created_at",
                "updated_at",
            ],
            "properties": {
                "project_id": {"bsonType": "string"},
                "name": {"bsonType": "string"},
                "status": {"bsonType": "string"},
                "description": {"bsonType": ["string", "null"]},
                "model_path": {"bsonType": "string"},
                "model_name": {"bsonType": "string"},
                "metadata": {"bsonType": ["object", "null"]},
                "created_at": {"bsonType": "date"},
                "updated_at": {"bsonType": "date"},
                "deleted_at": {"bsonType": ["date", "null"]},
            },
        }
    }


def _tenant_validator_lenient() -> Dict[str, Any]:
    # Keep this permissive to avoid blocking auth flows that create minimal tenant docs
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "properties": {
                "name": {"bsonType": ["string", "null"]},
                "company_name": {"bsonType": ["string", "null"]},
                "email": {"bsonType": ["string", "null"]},
                "created_at": {"bsonType": ["date", "null"]},
                "updated_at": {"bsonType": ["date", "null"]},
            },
        }
    }


def _user_validator_lenient() -> Dict[str, Any]:
    # Keep permissive; auth flow sets name/email/password/tenant_id/role without timestamps
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "properties": {
                "name": {"bsonType": ["string", "null"]},
                "email": {"bsonType": ["string", "null"]},
                "password": {"bsonType": ["string", "null"]},
                "tenant_id": {"bsonType": ["string", "null"]},
                "role": {"bsonType": ["string", "null"]},
                "created_at": {"bsonType": ["date", "null"]},
                "updated_at": {"bsonType": ["date", "null"]},
            },
        }
    }


def _admin_validator() -> Dict[str, Any]:
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["email", "password"],
            "properties": {
                "email": {"bsonType": "string"},
                "password": {"bsonType": "string"},
                "name": {"bsonType": ["string", "null"]},
                "status": {"bsonType": ["string", "null"]},
                "created_at": {"bsonType": ["date", "null"]},
                "updated_at": {"bsonType": ["date", "null"]},
            },
        }
    }


async def ensure_collections_and_indexes() -> None:
    """Create collections with validators and ensure indexes exist.

    This is idempotent and safe to call on every startup.
    """
    db = mongo_client.database

    collections: Dict[str, Dict[str, Any]] = {}
    # Add collections conditionally based on keys present in config
    if "PROJECT_COLLECTION" in database_config:
        collections[database_config["PROJECT_COLLECTION"]] = _project_validator()
    if "TRAIN_CONFIG_COLLECTION" in database_config:
        collections[database_config["TRAIN_CONFIG_COLLECTION"]] = _train_config_validator()
    if "BUCKET_CONFIG_COLLECTION" in database_config:
        collections[database_config["BUCKET_CONFIG_COLLECTION"]] = _bucket_config_validator()
    if "DATASET_CONFIG_COLLECTION" in database_config:
        collections[database_config["DATASET_CONFIG_COLLECTION"]] = _dataset_config_validator()
    if "MODEL_CONFIG_COLLECTION" in database_config:
        collections[database_config["MODEL_CONFIG_COLLECTION"]] = _model_config_validator()
    if "TENANT_COLLECTION" in database_config:
        collections[database_config["TENANT_COLLECTION"]] = _tenant_validator_lenient()
    if "USER_COLLECTION" in database_config:
        collections[database_config["USER_COLLECTION"]] = _user_validator_lenient()
    if "ADMIN_COLLECTION" in database_config:
        collections[database_config["ADMIN_COLLECTION"]] = _admin_validator()
    if "TRAIN_RUN_COLLECTION" in database_config:
        # define validator locally to keep file organized near ensure function
        collections[database_config["TRAIN_RUN_COLLECTION"]] = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": [
                    "train_config_id",
                    "status",
                    "created_at",
                    "updated_at",
                ],
                "properties": {
                    "train_config_id": {"bsonType": "string"},
                    # status examples: training, completed, failed, cancelled
                    "status": {"bsonType": "string"},
                    "created_at": {"bsonType": "date"},
                    "updated_at": {"bsonType": "date"},
                    "ended_at": {"bsonType": ["date", "null"]},
                    "step_status": {
                        "bsonType": ["object", "null"],
                        "properties": {
                            "loading_data": {"bsonType": ["string", "null"]},
                            "training": {"bsonType": ["string", "null"]},
                            "saving_model": {"bsonType": ["string", "null"]}
                        }
                    },
                    "results": {"bsonType": ["object", "null"]},
                    "error": {"bsonType": ["string", "null"]}
                },
            }
        }

    existing = await db.list_collection_names()

    for name, validator in collections.items():
        try:
            if name not in existing:
                await db.create_collection(name, validator=validator)
                logger.info(f"Created collection {name} with validator")
            else:
                try:
                    await db.command({
                        "collMod": name,
                        "validator": validator,
                        "validationLevel": "moderate",
                    })
                    logger.info(f"Updated validator for collection {name}")
                except Exception as e:
                    logger.warning(f"Could not update validator for {name}: {e}")
        except Exception as e:
            logger.error(f"Error ensuring collection {name}: {e}")

    # Indexes (create only for collections present)
    # Tenants unique email
    if "TENANT_COLLECTION" in database_config:
        try:
            await db[database_config["TENANT_COLLECTION"]].create_index("email", unique=True, name="uniq_tenant_email")
        except Exception as e:
            logger.warning(f"Create index tenants.email failed or exists: {e}")

    # Users unique email
    if "USER_COLLECTION" in database_config:
        try:
            await db[database_config["USER_COLLECTION"]].create_index("email", unique=True, name="uniq_user_email")
        except Exception as e:
            logger.warning(f"Create index users.email failed or exists: {e}")

    # Admins unique email
    if "ADMIN_COLLECTION" in database_config:
        try:
            await db[database_config["ADMIN_COLLECTION"]].create_index("email", unique=True, name="uniq_admin_email")
        except Exception as e:
            logger.warning(f"Create index admins.email failed or exists: {e}")

    if "DATASET_CONFIG_COLLECTION" in database_config:
        try:
            await db[database_config["DATASET_CONFIG_COLLECTION"]].create_index(
                "project_id", unique=False, name="idx_dataset_project_id"
            )
        except Exception as e:
            logger.warning(f"Create index dataset_configs (project_id) failed or exists: {e}")

    if "MODEL_CONFIG_COLLECTION" in database_config:
        try:
            await db[database_config["MODEL_CONFIG_COLLECTION"]].create_index(
                "project_id", unique=False, name="idx_model_project_id"
            )
        except Exception as e:
            logger.warning(f"Create index model_configs (project_id) failed or exists: {e}")

    if "BUCKET_CONFIG_COLLECTION" in database_config:
        try:
            await db[database_config["BUCKET_CONFIG_COLLECTION"]].create_index(
                [("tenant_id", 1), ("project_id", 1)], unique=False, name="idx_bucket_tenant_project"
            )
        except Exception as e:
            logger.warning(f"Create index bucket_configs (tenant_id,project_id) failed or exists: {e}")

    if "TRAIN_CONFIG_COLLECTION" in database_config:
        try:
            await db[database_config["TRAIN_CONFIG_COLLECTION"]].create_index(
                [("tenant_id", 1), ("project_id", 1)], unique=False, name="idx_train_tenant_project"
            )
        except Exception as e:
            logger.warning(f"Create index train_configs (tenant_id,project_id) failed or exists: {e}")

        # Enforce single train_config per project
        try:
            await db[database_config["TRAIN_CONFIG_COLLECTION"]].create_index(
                "project_id", unique=True, name="uniq_train_config_per_project"
            )
        except Exception as e:
            logger.warning(f"Create unique index train_configs.project_id failed or exists: {e}")

    if "TRAIN_RUN_COLLECTION" in database_config:
        try:
            await db[database_config["TRAIN_RUN_COLLECTION"]].create_index(
                [("train_config_id", 1), ("status", 1)], unique=False, name="idx_train_run_config_status"
            )
        except Exception as e:
            logger.warning(f"Create index train_runs (train_config_id,status) failed or exists: {e}")


