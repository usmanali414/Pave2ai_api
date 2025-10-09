"""
Example: Upload trained Keras model weights to S3 (train/input_model)
"""

from app.services.s3.s3_operations import s3_operations
from tensorflow import keras
import os


def create_and_save_dummy_model_weights():
    """Create a small Keras model and save its weights locally"""
    model = keras.Sequential([
        keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Save model weights locally
    local_weights_path = "dummy_model_weights.h5"
    model.save_weights(local_weights_path)
    return local_weights_path


def example_upload_model_weights():
    """Upload local Keras model weights to S3 input_model directory"""
    # Create dummy model weights
    weights_path = create_and_save_dummy_model_weights()
    print(f"✅ Model weights file created: {weights_path}")

    # Define S3 destination
    s3_destination = (
        "s3://testusman123/"
        "68db6b18bee4320874c73a71/project/"
        "68dca627a340f24038389ad5/train/input_model/dummy_model_weights.h5"
    )

    # Upload the weights file to S3
    result = s3_operations.upload_file(
        file_data=weights_path,
        s3_url=s3_destination,
        content_type="application/octet-stream",
        metadata={"model": "keras", "type": "weights"}
    )

    # Log upload result
    if result["success"]:
        print(f"✅ Model weights uploaded successfully:")
        print(f"   S3 URL: {result['s3_url']}")
        print(f"   Size: {result['size']} bytes")
    else:
        print(f"❌ Upload failed: {result['error']}")


if __name__ == "__main__":
    print("=== Uploading Keras Model Weights to S3 ===\n")
    example_upload_model_weights()
