"""
Script to delete S3 folder structure for projects
WARNING: This is destructive and cannot be undone!
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.s3.s3_client import s3_client
from app.services.s3.s3_helper import s3_helper
from config import S3_DATA_BUCKET
import argparse


def delete_project_folders(tenant_id: str, project_id: str, confirm: bool = False) -> bool:
    """
    Delete all S3 folders for a specific project
    
    Args:
        tenant_id: Tenant ID
        project_id: Project ID
        confirm: If True, skip confirmation prompt
    
    Returns:
        True if successful, False otherwise
    """
    if not s3_client.is_initialized():
        print("❌ S3 client not initialized. Check AWS credentials in .env")
        return False
    
    base_path = f"accounts/{tenant_id}/project/{project_id}/"
    
    print(f"\n{'='*60}")
    print("⚠️  WARNING: DESTRUCTIVE OPERATION")
    print(f"{'='*60}")
    print(f"\nYou are about to DELETE all folders and files for:")
    print(f"  Tenant ID:  {tenant_id}")
    print(f"  Project ID: {project_id}")
    print(f"  Bucket:     {S3_DATA_BUCKET}")
    print(f"  Base Path:  {base_path}")
    
    try:
        bucket = s3_client.resource.Bucket(S3_DATA_BUCKET)
        
        # List all objects that will be deleted
        objects_to_delete = list(bucket.objects.filter(Prefix=base_path))
        
        if not objects_to_delete:
            print(f"\n✓ No objects found for this project. Nothing to delete.")
            return True
        
        print(f"\n📊 Found {len(objects_to_delete)} object(s) to delete:")
        for i, obj in enumerate(objects_to_delete[:10], 1):  # Show first 10
            print(f"  {i}. {obj.key}")
        
        if len(objects_to_delete) > 10:
            print(f"  ... and {len(objects_to_delete) - 10} more")
        
        # Confirmation
        if not confirm:
            print(f"\n{'─'*60}")
            response = input("\n⚠️  Type 'DELETE' to confirm deletion (or anything else to cancel): ")
            if response != 'DELETE':
                print("\n❌ Deletion cancelled.")
                return False
        
        # Delete all objects
        print(f"\n{'─'*60}")
        print("Deleting objects...")
        
        if objects_to_delete:
            bucket.delete_objects(
                Delete={
                    'Objects': [{'Key': obj.key} for obj in objects_to_delete]
                }
            )
            print(f"\n✓ Successfully deleted {len(objects_to_delete)} object(s)")
            print(f"✓ All folders removed for project {project_id}")
        
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error deleting project folders: {str(e)}")
        print(f"{'='*60}\n")
        return False


def delete_tenant_folders(tenant_id: str, confirm: bool = False) -> bool:
    """
    Delete all S3 folders for an entire tenant (all projects)
    
    Args:
        tenant_id: Tenant ID
        confirm: If True, skip confirmation prompt
    
    Returns:
        True if successful, False otherwise
    """
    if not s3_client.is_initialized():
        print("❌ S3 client not initialized. Check AWS credentials in .env")
        return False
    
    base_path = f"accounts/{tenant_id}/"
    
    print(f"\n{'='*60}")
    print("⚠️  WARNING: EXTREMELY DESTRUCTIVE OPERATION")
    print(f"{'='*60}")
    print(f"\nYou are about to DELETE ALL projects and data for:")
    print(f"  Tenant ID:  {tenant_id}")
    print(f"  Bucket:     {S3_DATA_BUCKET}")
    print(f"  Base Path:  {base_path}")
    
    try:
        bucket = s3_client.resource.Bucket(S3_DATA_BUCKET)
        
        # List all objects that will be deleted
        objects_to_delete = list(bucket.objects.filter(Prefix=base_path))
        
        if not objects_to_delete:
            print(f"\n✓ No objects found for this tenant. Nothing to delete.")
            return True
        
        # Count projects
        projects = set()
        for obj in objects_to_delete:
            parts = obj.key.split('/')
            if len(parts) >= 5 and parts[0] == 'accounts' and parts[2] == 'project':
                projects.add(parts[3])
        
        print(f"\n📊 Found {len(projects)} project(s) with {len(objects_to_delete)} object(s) to delete")
        print(f"\nProjects that will be deleted:")
        for i, proj_id in enumerate(list(projects)[:5], 1):
            print(f"  {i}. {proj_id}")
        if len(projects) > 5:
            print(f"  ... and {len(projects) - 5} more")
        
        # Confirmation
        if not confirm:
            print(f"\n{'─'*60}")
            print(f"⚠️  This will DELETE {len(projects)} project(s) and ALL their data!")
            response = input(f"\nType 'DELETE ALL' to confirm deletion (or anything else to cancel): ")
            if response != 'DELETE ALL':
                print("\n❌ Deletion cancelled.")
                return False
        
        # Delete all objects
        print(f"\n{'─'*60}")
        print("Deleting all objects...")
        
        if objects_to_delete:
            # Delete in batches of 1000 (S3 limit)
            batch_size = 1000
            for i in range(0, len(objects_to_delete), batch_size):
                batch = objects_to_delete[i:i+batch_size]
                bucket.delete_objects(
                    Delete={
                        'Objects': [{'Key': obj.key} for obj in batch]
                    }
                )
                print(f"  Deleted batch {i//batch_size + 1} ({len(batch)} objects)")
            
            print(f"\n✓ Successfully deleted {len(objects_to_delete)} object(s)")
            print(f"✓ All folders removed for tenant {tenant_id}")
        
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error deleting tenant folders: {str(e)}")
        print(f"{'='*60}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Delete S3 folder structure for projects or tenants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  WARNING: This script permanently deletes data from S3!

Examples:
  # Delete specific project (with confirmation)
  python delete_project_folders.py --tenant abc123 --project xyz789
  
  # Delete project without confirmation (use with caution!)
  python delete_project_folders.py --tenant abc123 --project xyz789 --yes
  
  # Delete all projects for a tenant
  python delete_project_folders.py --tenant abc123 --all-projects
  
  # Force delete tenant without confirmation (DANGER!)
  python delete_project_folders.py --tenant abc123 --all-projects --yes
        """
    )
    
    parser.add_argument('--tenant', '-t', required=True, help='Tenant ID')
    parser.add_argument('--project', '-p', help='Project ID to delete')
    parser.add_argument('--all-projects', action='store_true', help='Delete ALL projects for the tenant')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt (dangerous!)')
    
    args = parser.parse_args()
    
    # Check S3 initialization
    if not s3_client.is_initialized():
        print("\n❌ S3 client not initialized!")
        print("\nPlease ensure your .env file has:")
        print("  AWS_ACCESS_KEY_ID=your_key")
        print("  AWS_SECRET_ACCESS_KEY=your_secret")
        print("  AWS_REGION=us-east-1")
        print("  S3_DATA_BUCKET=your_bucket\n")
        return 1
    
    # Delete entire tenant
    if args.all_projects:
        success = delete_tenant_folders(
            tenant_id=args.tenant,
            confirm=args.yes
        )
        return 0 if success else 1
    
    # Delete specific project
    if args.project:
        success = delete_project_folders(
            tenant_id=args.tenant,
            project_id=args.project,
            confirm=args.yes
        )
        return 0 if success else 1
    
    # No valid arguments
    print("\nError: Please provide either --project or --all-projects")
    print("Use --help for more information\n")
    return 1


if __name__ == "__main__":
    exit(main())

