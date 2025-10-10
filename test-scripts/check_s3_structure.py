"""
Script to check S3 folder structure for projects
Verifies that all required folders exist in the S3 bucket
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.s3.s3_client import s3_client
from app.services.s3.s3_helper import s3_helper
from config import S3_DATA_BUCKET
import argparse
from typing import List, Dict


def check_folder_exists(bucket, folder_path: str) -> bool:
    """Check if a specific folder (0-byte object) exists in S3"""
    try:
        bucket.Object(folder_path).load()
        return True
    except:
        return False


def check_project_structure(tenant_id: str, project_id: str, verbose: bool = False) -> Dict:
    """
    Check if all required folders exist for a project
    
    Returns:
        Dict with status and missing folders
    """
    if not s3_client.is_initialized():
        print("❌ S3 client not initialized. Check AWS credentials in .env")
        return {"success": False, "error": "S3 not initialized"}
    
    base_path = f"accounts/{tenant_id}/project/{project_id}"
    
    # Define all required folders
    required_folders = [
        f"{base_path}/user/logs/",
        f"{base_path}/data/raw/",
        f"{base_path}/data/preprocessed/",
        f"{base_path}/annotate/tool/",
        f"{base_path}/annotate/label/",
        f"{base_path}/train/input_model/",
        f"{base_path}/train/output_model/",
        f"{base_path}/train/output_metadata/",
        f"{base_path}/inference/input_model/",
        f"{base_path}/inference/output_labels/",
        f"{base_path}/inference/output_metadata/",
    ]
    
    bucket = s3_client.resource.Bucket(S3_DATA_BUCKET)
    
    existing_folders = []
    missing_folders = []
    
    print(f"\n{'='*60}")
    print(f"Checking S3 structure for:")
    print(f"  Tenant ID:  {tenant_id}")
    print(f"  Project ID: {project_id}")
    print(f"  Bucket:     {S3_DATA_BUCKET}")
    print(f"{'='*60}\n")
    
    for folder_path in required_folders:
        exists = check_folder_exists(bucket, folder_path)
        
        if exists:
            existing_folders.append(folder_path)
            if verbose:
                print(f"✓ {folder_path}")
        else:
            missing_folders.append(folder_path)
            if verbose:
                print(f"✗ {folder_path}")
    
    # Summary
    total = len(required_folders)
    found = len(existing_folders)
    missing = len(missing_folders)
    
    print(f"\n{'─'*60}")
    print(f"Summary:")
    print(f"  Total folders:    {total}")
    print(f"  Found:            {found} ({'✓' if found == total else '✗'})")
    print(f"  Missing:          {missing}")
    
    if missing > 0:
        print(f"\n{'─'*60}")
        print("Missing folders:")
        for folder in missing_folders:
            print(f"  ✗ {folder}")
    
    print(f"{'='*60}\n")
    
    return {
        "success": missing == 0,
        "total": total,
        "found": found,
        "missing": missing,
        "existing_folders": existing_folders,
        "missing_folders": missing_folders
    }


def build_folder_tree(bucket) -> Dict:
    """
    Build a hierarchical tree structure of all folders in the bucket
    
    Returns:
        Nested dict representing the folder structure
    """
    tree = {}
    
    for obj in bucket.objects.all():
        key = obj.key
        parts = key.split('/')
        
        # Build nested dict
        current = tree
        for part in parts:
            if part:  # Skip empty parts
                if part not in current:
                    current[part] = {}
                current = current[part]
    
    return tree


def print_tree(tree: Dict, prefix: str = "", is_last: bool = True, level: int = 0, max_level: int = None):
    """
    Print a tree structure with nice formatting
    
    Args:
        tree: Nested dict representing folder structure
        prefix: Current line prefix for indentation
        is_last: Whether this is the last item in current level
        level: Current depth level
        max_level: Maximum depth to display (None = unlimited)
    """
    if max_level is not None and level >= max_level:
        return
    
    items = list(tree.items())
    for i, (name, subtree) in enumerate(items):
        is_last_item = (i == len(items) - 1)
        
        # Determine the symbols to use
        if level == 0:
            connector = ""
            new_prefix = ""
        else:
            connector = "└── " if is_last_item else "├── "
            new_prefix = prefix + ("    " if is_last_item else "│   ")
        
        # Print the current item
        if isinstance(subtree, dict) and subtree:
            # It's a folder (has children)
            print(f"{prefix}{connector}{name}/")
            print_tree(subtree, new_prefix, is_last_item, level + 1, max_level)
        else:
            # It's a file or empty folder
            print(f"{prefix}{connector}{name}")


def list_all_projects_in_bucket(verbose: bool = False) -> List[Dict]:
    """
    List all projects found in the S3 bucket and display the folder structure
    
    Returns:
        List of dicts with tenant_id and project_id
    """
    if not s3_client.is_initialized():
        print("❌ S3 client not initialized. Check AWS credentials in .env")
        return []
    
    print(f"\n{'='*60}")
    print(f"Scanning bucket: {S3_DATA_BUCKET}")
    print(f"{'='*60}\n")
    
    bucket = s3_client.resource.Bucket(S3_DATA_BUCKET)
    
    projects = []
    
    try:
        # Build folder tree
        print("Building folder structure...\n")
        tree = build_folder_tree(bucket)
        
        if not tree:
            print("❌ No folders found in bucket.\n")
            return []
        
        # Print the tree structure
        print(f"📁 {S3_DATA_BUCKET}/")
        print_tree(tree, "", True, 0, max_level=None if verbose else 4)
        
        print(f"\n{'─'*60}\n")
        
        # Extract project information
        for obj in bucket.objects.all():
            key = obj.key
            parts = key.split('/')
            
            # Looking for pattern: accounts/{tenant_id}/project/{project_id}/...
            if len(parts) >= 4 and parts[0] == 'accounts' and parts[2] == 'project':
                tenant_id = parts[1]
                project_id = parts[3]
                
                # Add to set to avoid duplicates
                project_key = (tenant_id, project_id)
                if project_key not in [(p['tenant_id'], p['project_id']) for p in projects]:
                    projects.append({
                        'tenant_id': tenant_id,
                        'project_id': project_id
                    })
        
        if len(projects) == 0:
            print("No projects found in bucket structure.")
        else:
            print(f"Found {len(projects)} project(s):\n")
            for i, proj in enumerate(projects, 1):
                print(f"{i}. Tenant: {proj['tenant_id']} | Project: {proj['project_id']}")
        
        print(f"\n{'='*60}\n")
        
        return projects
        
    except Exception as e:
        print(f"❌ Error scanning bucket: {str(e)}")
        return []


def create_missing_folders(tenant_id: str, project_id: str) -> bool:
    """
    Create missing folders for a project
    
    Returns:
        True if successful
    """
    print(f"\n{'='*60}")
    print("Creating missing folders...")
    print(f"{'='*60}\n")
    
    result = s3_helper.create_project_folder_structure(
        tenant_id=tenant_id,
        project_id=project_id
    )
    
    if result.get("success"):
        print(f"✓ Successfully created {result.get('total_folders')} folders")
        return True
    else:
        print(f"✗ Failed to create folders: {result.get('error', 'Unknown error')}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Check S3 folder structure for projects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all projects and show folder structure (depth limited)
  python check_s3_structure.py --list-all
  
  # List all with full depth
  python check_s3_structure.py --list-all --verbose
  
  # Check specific project
  python check_s3_structure.py --tenant abc123 --project xyz789
  
  # Show tree structure for specific project
  python check_s3_structure.py --tenant abc123 --project xyz789 --tree
  
  # Check with detailed folder list
  python check_s3_structure.py --tenant abc123 --project xyz789 --verbose
  
  # Create missing folders
  python check_s3_structure.py --tenant abc123 --project xyz789 --create
        """
    )
    
    parser.add_argument('--tenant', '-t', help='Tenant ID to check')
    parser.add_argument('--project', '-p', help='Project ID to check')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed folder list (no depth limit)')
    parser.add_argument('--list-all', '-l', action='store_true', help='List all projects and show folder structure')
    parser.add_argument('--create', '-c', action='store_true', help='Create missing folders')
    parser.add_argument('--tree', action='store_true', help='Show tree structure for specific project')
    
    args = parser.parse_args()
    
    # Check S3 initialization
    if not s3_client.is_initialized():
        print("\n❌ S3 client not initialized!")
        print("\nPlease ensure your .env file has:")
        print("  AWS_ACCESS_KEY_ID=your_key")
        print("  AWS_SECRET_ACCESS_KEY=your_secret")
        print("  AWS_REGION=us-east-1")
        print("  S3_DATA_BUCKET=mldatasets\n")
        return 1
    
    # List all projects
    if args.list_all:
        projects = list_all_projects_in_bucket(verbose=args.verbose)
        
        if projects and len(projects) > 0:
            print("\nTo check a specific project, use:")
            print(f"  python check_s3_structure.py --tenant <tenant_id> --project <project_id>")
        
        return 0
    
    # Check specific project
    if args.tenant and args.project:
        # Show tree structure if requested
        if args.tree:
            try:
                bucket = s3_client.resource.Bucket(S3_DATA_BUCKET)
                base_path = f"accounts/{args.tenant}/project/{args.project}/"
                
                print(f"\n{'='*60}")
                print(f"Folder structure for project:")
                print(f"  Tenant: {args.tenant}")
                print(f"  Project: {args.project}")
                print(f"{'='*60}\n")
                
                # Build tree for this specific project
                project_tree = {}
                for obj in bucket.objects.filter(Prefix=base_path):
                    key = obj.key
                    # Remove base path to get relative path
                    relative_path = key[len(base_path):]
                    if relative_path:
                        parts = relative_path.split('/')
                        current = project_tree
                        for part in parts:
                            if part:
                                if part not in current:
                                    current[part] = {}
                                current = current[part]
                
                if project_tree:
                    print(f"📁 accounts/")
                    print(f"└── {args.tenant}/")
                    print(f"    └── project/")
                    print(f"        └── {args.project}/")
                    # Print from 4 spaces indent (under project_id)
                    print_tree(project_tree, "        ", True, 0)
                    print()
                else:
                    print("❌ No folders found for this project.\n")
                
            except Exception as e:
                print(f"❌ Error building tree: {str(e)}\n")
        
        # Check folder structure
        result = check_project_structure(
            tenant_id=args.tenant,
            project_id=args.project,
            verbose=args.verbose
        )
        
        # Create folders if requested and there are missing ones
        if args.create and result.get('missing', 0) > 0:
            success = create_missing_folders(args.tenant, args.project)
            
            if success:
                print("\nRe-checking structure after creation...")
                result = check_project_structure(
                    tenant_id=args.tenant,
                    project_id=args.project,
                    verbose=args.verbose
                )
        
        # Return exit code based on success
        return 0 if result.get('success') else 1
    
    # No valid arguments
    print("\nError: Please provide either --list-all or both --tenant and --project")
    print("Use --help for more information\n")
    return 1


if __name__ == "__main__":
    exit(main())

