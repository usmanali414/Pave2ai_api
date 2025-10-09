"""
Test script to verify S3 setup and folder creation
Run this after setting up AWS credentials in .env
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.s3.s3_client import s3_client
from app.services.s3.s3_helper import s3_helper
from config import S3_DATA_BUCKET


def test_s3_client_initialization():
    """Test if S3 client initializes properly"""
    print("\n=== Testing S3 Client Initialization ===")
    
    if s3_client.is_initialized():
        print("✓ S3 client initialized successfully")
        return True
    else:
        print("✗ S3 client not initialized. Check AWS credentials in .env")
        return False


def test_bucket_access():
    """Test if we can access the S3 bucket"""
    print("\n=== Testing Bucket Access ===")
    
    if not s3_client.is_initialized():
        print("✗ S3 client not initialized")
        return False
    
    try:
        # Try to list objects in bucket (just first 1)
        response = s3_client.client.list_objects_v2(
            Bucket=S3_DATA_BUCKET,
            MaxKeys=1
        )
        print(f"✓ Successfully accessed bucket: {S3_DATA_BUCKET}")
        return True
    except Exception as e:
        print(f"✗ Failed to access bucket: {str(e)}")
        return False


def test_folder_structure_creation():
    """Test creating folder structure for a test project"""
    print("\n=== Testing Folder Structure Creation ===")
    
    if not s3_client.is_initialized():
        print("✗ S3 client not initialized")
        return False
    
    # Use test IDs
    test_tenant_id = "test_tenant_123"
    test_project_id = "test_project_456"
    
    try:
        print(f"Creating folder structure for:")
        print(f"  Tenant ID: {test_tenant_id}")
        print(f"  Project ID: {test_project_id}")
        
        result = s3_helper.create_project_folder_structure(
            tenant_id=test_tenant_id,
            project_id=test_project_id
        )
        
        if result.get("success"):
            print(f"✓ Created {result.get('total_folders')} folders successfully")
            print(f"  Raw Data URL: {result.get('raw_data_url')}")
            return True
        else:
            print(f"✗ Folder creation incomplete: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to create folders: {str(e)}")
        return False


def test_folder_existence():
    """Test checking if a folder exists"""
    print("\n=== Testing Folder Existence Check ===")
    
    if not s3_client.is_initialized():
        print("✗ S3 client not initialized")
        return False
    
    test_tenant_id = "test_tenant_123"
    test_project_id = "test_project_456"
    
    try:
        exists = s3_helper.check_folder_exists(
            tenant_id=test_tenant_id,
            project_id=test_project_id,
            folder_path="data/raw/"
        )
        
        if exists:
            print(f"✓ Folder exists: data/raw/")
            return True
        else:
            print(f"✗ Folder does not exist (may be expected if not created yet)")
            return False
            
    except Exception as e:
        print(f"✗ Failed to check folder: {str(e)}")
        return False


def test_cleanup():
    """Clean up test folders"""
    print("\n=== Cleaning Up Test Folders ===")
    
    if not s3_client.is_initialized():
        print("✗ S3 client not initialized")
        return False
    
    test_tenant_id = "test_tenant_123"
    test_project_id = "test_project_456"
    
    try:
        success = s3_helper.delete_project_folders(
            tenant_id=test_tenant_id,
            project_id=test_project_id
        )
        
        if success:
            print("✓ Test folders cleaned up successfully")
            return True
        else:
            print("✗ Failed to clean up test folders")
            return False
            
    except Exception as e:
        print(f"✗ Failed to clean up: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("S3 SETUP VERIFICATION TEST")
    print("="*60)
    
    results = []
    
    # Test 1: Client initialization
    results.append(("Client Initialization", test_s3_client_initialization()))
    
    # Only continue if client initialized
    if not results[0][1]:
        print("\n❌ Cannot proceed without S3 client initialization")
        print("\nPlease ensure your .env file has:")
        print("  AWS_ACCESS_KEY_ID=your_key")
        print("  AWS_SECRET_ACCESS_KEY=your_secret")
        print("  AWS_REGION=us-east-1")
        print("  S3_DATA_BUCKET=mldatasets")
        return
    
    # Test 2: Bucket access
    results.append(("Bucket Access", test_bucket_access()))
    
    if results[1][1]:
        # Test 3: Create folders
        results.append(("Folder Creation", test_folder_structure_creation()))
        
        # Test 4: Check folder exists
        results.append(("Folder Existence", test_folder_existence()))
        
        # Test 5: Cleanup
        results.append(("Cleanup", test_cleanup()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n✅ All tests passed! S3 is properly configured.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()

