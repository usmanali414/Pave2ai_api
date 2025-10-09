from app.services.s3.s3_operations import s3_operations
from tensorflow import keras
import numpy as np
import tempfile
import os

def read_model_weights(model_weights_s3_url: str, local_path: str):
    """
    Download model weights from S3 and save locally.
    Returns the local file path.
    """
    result = s3_operations.download_file(
        s3_url=model_weights_s3_url,
        local_path=local_path
    )
    if not result.get("success"):
        raise Exception(f"❌ Failed to download model weights: {result.get('error')}")
    print(f"✅ Model weights downloaded from {model_weights_s3_url}")
    return local_path


def save_model_weights(local_path: str, model_weights_s3_url: str):
    """
    Upload model weights to S3.
    """
    result = s3_operations.upload_file(
        file_data=local_path,
        s3_url=model_weights_s3_url
    )
    if not result.get("success"):
        raise Exception(f"❌ Failed to upload model weights: {result.get('error')}")
    print(f"✅ Model weights uploaded to {model_weights_s3_url}")


def create_model():
    """Create a simple dummy model"""
    model = keras.Sequential([
        keras.layers.Dense(4, activation='relu', input_shape=(3,)),
        keras.layers.Dense(2, activation='softmax')
    ])
    return model


# ------------------ MAIN FLOW ------------------

# S3 Paths
input_s3_url = "s3://testusman123/68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/train/input_model/dummy_model_weights.h5"
output_s3_url = "s3://testusman123/68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/train/output_model/dummy_model_weights.h5"

# Create model
model = create_model()

# Temporary local path for weights
tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".h5").name
tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".h5").name

# Download (dummy or existing) input weights
try:
    read_model_weights(input_s3_url, tmp_input)
    model.load_weights(tmp_input)
    print("✅ Loaded input weights into model.")
except Exception as e:
    print(f"⚠️ Could not load input weights — using random initialization. ({e})")

# (Optional) train or modify model
# model.fit(...)

# Save updated weights locally
model.save_weights(tmp_output)
print(f"✅ Saved updated weights locally: {tmp_output}")

# Upload updated weights to output S3 path
save_model_weights(tmp_output, output_s3_url)

# (Optional) verify reload
new_model = create_model()
new_model.load_weights(tmp_output)
for w1, w2 in zip(model.get_weights(), new_model.get_weights()):
    assert np.allclose(w1, w2)
print("✅ Verification successful — weights match exactly.")

# Cleanup
os.remove(tmp_input)
os.remove(tmp_output)
