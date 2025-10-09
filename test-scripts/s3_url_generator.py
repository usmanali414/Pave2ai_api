import os
import boto3
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read values
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION= os.getenv("AWS_REGION")

# # Create S3 client
# s3 = boto3.client(
#     "s3",
#     aws_access_key_id=ACCESS_KEY,
#     aws_secret_access_key=SECRET_KEY,
#     region_name=REGION
# )

# # Example: upload a file
# bucket_name = "testusmanali123"
# key = "tenant123/user456/project789/test.txt"

# s3.put_object(Bucket=bucket_name, Key=key, Body="Hello from .env setup!")
# print(f"Uploaded to s3://{bucket_name}/{key}")

# # import boto3

# # # Replace with your AWS credentials
# # ACCESS_KEY = "YOUR_ACCESS_KEY"
# # SECRET_KEY = "YOUR_SECRET_KEY"
# # REGION = "us-east-1"  # change if needed

def list_buckets():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,        # ✅ correct param
        aws_secret_access_key=SECRET_KEY,    # ✅ correct param
        region_name=REGION
    )

    response = s3.list_buckets()

    print("Available Buckets:")
    for bucket in response["Buckets"]:
        print(f"- {bucket['Name']} (Created: {bucket['CreationDate']})")

if __name__ == "__main__":
    list_buckets()

