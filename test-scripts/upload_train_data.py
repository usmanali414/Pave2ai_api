from app.services.s3.s3_operations import s3_operations
from pathlib import Path
import os

# Configuration: List of upload mappings
UPLOAD_CONFIGS = [
    {
        "name": "Images",
        "local_path": "E:\Pave2ai_api\static\dataset\orig_images",
        "s3_url": "s3://testusman123/accounts/68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/data/preprocessed/",
        "file_extension": "*.jpg"
    },
    {
        "name": "JSON Annotations",
        "local_path": "E:\Pave2ai_api\static\dataset\jsons",
        "s3_url": "s3://testusman123/accounts/68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/annotate/label/",
        "file_extension": "*.json"
    }
]

def upload_files_from_config():
    """Upload files based on the configuration list"""
    
    for config in UPLOAD_CONFIGS:
        print(f"Uploading {config['name']}...")
        
        local_dir = Path(config['local_path'])
        files_uploaded = 0
        files_failed = 0
        
        # Find and upload files matching the extension
        for file_path in local_dir.glob(config['file_extension']):
            s3_url = config['s3_url'] + file_path.name
            print(s3_url)
            
            print(f"  Uploading {file_path.name} to {s3_url}")
            result = s3_operations.upload_file(
                file_data=str(file_path),
                s3_url=s3_url
            )
            
            if result["success"]:
                print(f"  ✓ Successfully uploaded {file_path.name}")
                files_uploaded += 1
            else:
                print(f"  ✗ Failed to upload {file_path.name}: {result.get('error', 'Unknown error')}")
                files_failed += 1
        
        print(f"  Summary: {files_uploaded} successful, {files_failed} failed")
        print("\n" + "="*50 + "\n")

def upload_files_generic(local_path, s3_url, file_extension, description=""):
    """Generic function to upload files from local path to S3"""
    
    print(f"Uploading {description or 'files'}...")
    local_dir = Path(local_path)
    files_uploaded = 0
    files_failed = 0
    
    for file_path in local_dir.glob(file_extension):
        full_s3_url = s3_url + file_path.name
        print(full_s3_url)
        print(f"  Uploading {file_path.name} to {full_s3_url}")
        result = s3_operations.upload_file(
            file_data=str(file_path),
            s3_url=full_s3_url
        )
        
        if result["success"]:
            print(f"  ✓ Successfully uploaded {file_path.name}")
            files_uploaded += 1
        else:
            print(f"  ✗ Failed to upload {file_path.name}: {result.get('error', 'Unknown error')}")
            files_failed += 1
    
    print(f"  Summary: {files_uploaded} successful, {files_failed} failed")
    return files_uploaded, files_failed

if __name__ == "__main__":
    # Option 1: Use configuration-based approach
    upload_files_from_config()
    
    # Option 2: Use generic function directly (uncomment to use)
    # upload_files_generic(
    #     local_path="E:\\Pave2ai_api\\app\\static\\dataset\\orig_images",
    #     s3_url="s3://testusman123/accounts/68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/data/preprocessed/",
    #     file_extension="*.jpg",
    #     description="Images"
    # )
    # print("\n" + "="*50 + "\n")
    # upload_files_generic(
    #     local_path="E:\\Pave2ai_api\\app\\static\\dataset\\jsons",
    #     s3_url="s3://testusman123/accounts/68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/annotate/label/",
    #     file_extension="*.json",
    #     description="JSON Annotations"
    # )