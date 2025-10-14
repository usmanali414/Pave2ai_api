"""
Test script for the new project deletion endpoint.
This script demonstrates how to use the complete project deletion functionality.
"""
import sys
import os
import requests
import json
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
BASE_URL = "http://localhost:8000"  # Adjust this to your FastAPI server URL
API_ENDPOINT = "/project/{tenant_id}/{project_id}"


def test_project_deletion(tenant_id: str, project_id: str, dry_run: bool = True) -> Dict[str, Any]:
    """
    Test the project deletion endpoint.
    
    Args:
        tenant_id: The tenant ID
        project_id: The project ID to delete
        dry_run: If True, just shows what would be deleted (not implemented in this test)
        
    Returns:
        Response from the deletion endpoint
    """
    url = f"{BASE_URL}{API_ENDPOINT.format(tenant_id=tenant_id, project_id=project_id)}"
    
    print(f"\n{'='*60}")
    print("🧪 TESTING PROJECT DELETION ENDPOINT")
    print(f"{'='*60}")
    print(f"Project ID: {project_id}")
    print(f"Endpoint: {url}")
    print(f"Method: DELETE")
    
    if dry_run:
        print(f"\n⚠️  DRY RUN MODE - This is just a test, no actual deletion will occur")
        print(f"To perform actual deletion, set dry_run=False")
        return {"status": "dry_run", "message": "No actual deletion performed"}
    
    try:
        # Make the DELETE request
        print(f"\n🚀 Sending DELETE request...")
        response = requests.delete(url, timeout=30)
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        
        # Parse response
        try:
            response_data = response.json()
            print(f"📊 Response Body:")
            print(json.dumps(response_data, indent=2))
        except json.JSONDecodeError:
            print(f"📊 Response Body (raw): {response.text}")
        
        # Analyze response
        if response.status_code == 200:
            print(f"\n✅ SUCCESS: Project deleted successfully")
        elif response.status_code == 207:
            print(f"\n⚠️  PARTIAL SUCCESS: Project deletion completed with warnings")
        elif response.status_code == 404:
            print(f"\n❌ NOT FOUND: Project {project_id} not found")
        elif response.status_code == 500:
            print(f"\n❌ SERVER ERROR: Internal server error occurred")
        else:
            print(f"\n❓ UNEXPECTED STATUS: {response.status_code}")
        
        return {
            "status_code": response.status_code,
            "response_data": response_data if 'response_data' in locals() else response.text,
            "success": response.status_code in [200, 207]
        }
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ REQUEST ERROR: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "success": False
        }
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "success": False
        }


def list_projects() -> None:
    """List available projects for testing."""
    try:
        url = f"{BASE_URL}/projects"
        print(f"\n📋 Fetching available projects from: {url}")
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            projects = response.json()
            print(f"✅ Found {len(projects)} projects:")
            for i, project in enumerate(projects, 1):
                print(f"  {i}. ID: {project.get('id', 'N/A')} - Name: {project.get('name', 'N/A')}")
        else:
            print(f"❌ Failed to fetch projects: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error fetching projects: {str(e)}")


def main():
    """Main function to run the test."""
    print("🧪 PROJECT DELETION ENDPOINT TEST")
    print("="*60)
    
    # Check if server is running
    try:
        health_response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"✅ Server is running at {BASE_URL}")
    except requests.exceptions.RequestException:
        print(f"❌ Server not reachable at {BASE_URL}")
        print(f"Please ensure your FastAPI server is running on {BASE_URL}")
        return 1
    
    # List available projects
    list_projects()
    
    # Get tenant_id and project ID from user
    print(f"\n{'─'*60}")
    tenant_id = input("Enter tenant ID: ").strip()
    if not tenant_id:
        print("❌ Tenant ID is required")
        return 1
    
    project_id = input("Enter project ID to test deletion (or 'exit' to quit): ").strip()
    
    if project_id.lower() == 'exit':
        print("👋 Goodbye!")
        return 0
    
    if not project_id:
        print("❌ No project ID provided")
        return 1
    
    # Confirm deletion
    print(f"\n⚠️  WARNING: You are about to DELETE project {project_id}")
    print(f"This will delete:")
    print(f"  - All S3 folders and files for this project")
    print(f"  - All bucket_config documents for this project")
    print(f"  - The project document itself")
    print(f"\nThis action CANNOT be undone!")
    
    confirm = input("\nType 'DELETE' to confirm (or anything else to cancel): ").strip()
    
    if confirm != 'DELETE':
        print("❌ Deletion cancelled")
        return 0
    
    # Perform deletion test
    result = test_project_deletion(tenant_id, project_id, dry_run=False)
    
    print(f"\n{'='*60}")
    if result.get('success'):
        print("🎉 TEST COMPLETED SUCCESSFULLY")
    else:
        print("❌ TEST FAILED")
    print(f"{'='*60}\n")
    
    return 0 if result.get('success') else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
