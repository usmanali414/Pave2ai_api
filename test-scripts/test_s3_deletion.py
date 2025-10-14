#!/usr/bin/env python3
"""
Test script to verify S3 project deletion functionality.
"""
import sys
import os
import asyncio
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.s3.s3_helper import s3_helper
from app.services.s3.s3_operations import s3_operations
from config import S3_DATA_BUCKET
from app.utils.logger_utils import logger

def test_s3_deletion():
    """Test S3 project folder deletion."""
    print("Testing S3 project deletion functionality...")
    
    # Test data
    tenant_id = "68db6b18bee4320874c73a71"  # Example tenant ID
    project_id = "68dca627a340f24038389ad5"  # Example project ID
    
    print(f"Testing deletion for tenant: {tenant_id}")
    print(f"Testing deletion for project: {project_id}")
    
    # Check if S3 client is initialized
    print(f"S3 client initialized: {s3_helper.s3_client.is_initialized() if hasattr(s3_helper, 's3_client') else 'No s3_client attribute'}")
    
    # Test the delete_project_folders method
    try:
        print("Calling s3_helper.delete_project_folders()...")
        result = s3_helper.delete_project_folders(tenant_id, project_id)
        print(f"Delete result: {result}")
        
        if result:
            print("✅ S3 deletion successful!")
        else:
            print("❌ S3 deletion failed!")
            
    except Exception as e:
        print(f"❌ Error during S3 deletion: {str(e)}")
    
    # Also test using s3_operations directly
    try:
        print("\nTesting with s3_operations.delete_folder_contents()...")
        s3_path = f"s3://{S3_DATA_BUCKET}/accounts/{tenant_id}/project/{project_id}/"
        print(f"S3 path: {s3_path}")
        
        result = s3_operations.delete_folder_contents(s3_path)
        print(f"Delete result: {result}")
        
        if result.get("success"):
            print("✅ S3 operations deletion successful!")
        else:
            print(f"❌ S3 operations deletion failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error during S3 operations deletion: {str(e)}")

async def test_full_project_deletion():
    """Test the full project deletion service."""
    print("\n" + "="*50)
    print("Testing full project deletion service...")
    
    try:
        from app.services.project_deletion.project_deletion import delete_project_completely
        
        tenant_id = "68db6b18bee4320874c73a71"  # Example tenant ID
        project_id = "68dca627a340f24038389ad5"  # Example project ID
        print(f"Testing full deletion for tenant: {tenant_id}, project: {project_id}")
        
        result = await delete_project_completely(tenant_id, project_id)
        print(f"Full deletion result: {result}")
        
    except Exception as e:
        print(f"❌ Error during full project deletion: {str(e)}")

if __name__ == "__main__":
    print("S3 Project Deletion Test")
    print("=" * 50)
    
    # Test S3 deletion directly
    test_s3_deletion()
    
    # Test full project deletion service
    asyncio.run(test_full_project_deletion())
