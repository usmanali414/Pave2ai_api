"""
Script to delete ONE SPECIFIC external project folder from S3 bucket.
This will delete ONLY: s3://testusman123/68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/

CRITICAL SAFETY: This will ONLY delete this EXACT folder path.
All other folders will remain untouched, including:
- accounts/ folder and all its contents
- Any other external folders outside accounts/
- Any other files or folders in the bucket
"""

from app.services.s3.s3_operations import s3_operations
from app.services.s3.s3_client import s3_client
from app.utils.logger_utils import logger

def show_bucket_structure():
    """Show current bucket structure to verify what will be preserved."""
    bucket_name = "testusman123"
    target_prefix = "68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/"
    
    print("🔍 CURRENT BUCKET STRUCTURE:")
    print("=" * 50)
    
    try:
        if not s3_client.is_initialized():
            print("❌ S3 client not initialized. Cannot show bucket structure.")
            return
        
        client = s3_client.client
        
        # List all objects to show bucket structure
        paginator = client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Delimiter='/')
        
        folders = set()
        target_found = False
        
        for page in pages:
            # Get folder prefixes
            if 'CommonPrefixes' in page:
                for prefix in page['CommonPrefixes']:
                    folder_path = prefix['Prefix']
                    folders.add(folder_path)
                    
                    # Check if this is our target folder
                    if folder_path == target_prefix:
                        target_found = True
                        print(f"🎯 TARGET TO DELETE: {folder_path} ⚠️")
                    elif folder_path.startswith('accounts/'):
                        print(f"✅ PRESERVED: {folder_path}")
                    else:
                        print(f"✅ PRESERVED: {folder_path}")
        
        print("=" * 50)
        if target_found:
            print(f"✅ Target folder found and will be deleted")
        else:
            print(f"⚠️  Target folder not found - nothing to delete")
            
    except Exception as e:
        print(f"❌ Error showing bucket structure: {str(e)}")

def delete_external_project_folder():
    """Delete the external project folder and all its contents."""
    
    # Define the S3 path to delete (external folder, not under accounts/)
    bucket_name = "testusman123"
    folder_prefix = "68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/"
    
    print(f"🗑️  Preparing to delete SPECIFIC folder: s3://{bucket_name}/{folder_prefix}")
    print("🔒 CRITICAL SAFETY: This will ONLY delete this EXACT folder path")
    print("🔒 SAFETY: All other folders will remain untouched")
    print("⚠️  This will delete ALL files in this specific folder and its subfolders!")
    
    # Show what will be preserved
    print(f"\n✅ WILL BE PRESERVED:")
    print(f"  - s3://{bucket_name}/accounts/ (entire accounts folder)")
    print(f"  - ANY other external folders outside accounts/")
    print(f"  - ANY other files or folders in the bucket")
    print(f"  - ONLY the exact path '{folder_prefix}' will be deleted")
    
    # Ask for confirmation
    confirmation = input(f"\nAre you sure you want to delete ONLY s3://{bucket_name}/{folder_prefix}? (yes/no): ").lower().strip()
    
    if confirmation != 'yes':
        print("❌ Deletion cancelled.")
        return False
    
    try:
        # List all files in the folder
        print(f"\n📋 Listing files in s3://{bucket_name}/{folder_prefix}")
        list_result = s3_operations.list_files(
            s3_url_prefix=f"s3://{bucket_name}/{folder_prefix}",
            max_keys=1000
        )
        
        if not list_result["success"]:
            print(f"❌ Failed to list files: {list_result['error']}")
            return False
        
        files = list_result["files"]
        if not files:
            print("✅ Folder is already empty or doesn't exist.")
            return True
        
        print(f"📁 Found {len(files)} files to delete:")
        for file_info in files:
            print(f"  - {file_info['key']}")
        
        # Delete each file
        print(f"\n🗑️  Deleting {len(files)} files...")
        deleted_count = 0
        failed_count = 0
        
        for file_info in files:
            s3_url = file_info["s3_url"]
            try:
                result = s3_operations.delete_file(s3_url)
                if result["success"]:
                    print(f"  ✅ Deleted: {file_info['key']}")
                    deleted_count += 1
                else:
                    print(f"  ❌ Failed to delete: {file_info['key']} - {result['error']}")
                    failed_count += 1
            except Exception as e:
                print(f"  ❌ Error deleting {file_info['key']}: {str(e)}")
                failed_count += 1
        
        # Summary
        print(f"\n📊 Deletion Summary:")
        print(f"  ✅ Successfully deleted: {deleted_count} files")
        print(f"  ❌ Failed to delete: {failed_count} files")
        
        if failed_count == 0:
            print(f"\n🎉 Successfully deleted all files from s3://{bucket_name}/{folder_prefix}")
            return True
        else:
            print(f"\n⚠️  Some files could not be deleted. Please check the errors above.")
            return False
            
    except Exception as e:
        logger.error(f"Error deleting external project folder: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return False

def delete_folder_recursive():
    """Alternative method using batch delete (more efficient for large folders)."""
    
    bucket_name = "testusman123"
    folder_prefix = "68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/"
    
    print(f"🗑️  Using batch delete for: s3://{bucket_name}/{folder_prefix}")
    print("🔒 CRITICAL SAFETY: Only deleting this EXACT folder path.")
    print("🔒 SAFETY: All other folders will remain untouched:")
    print("    - accounts/ folder and all contents")
    print("    - Any other external folders outside accounts/")
    print("    - Any other files or folders in the bucket")
    
    try:
        # Check if S3 client is initialized
        if not s3_client.is_initialized():
            print("❌ S3 client is not initialized. Cannot proceed with deletion.")
            return False
        
        # Get S3 client to use batch delete
        client = s3_client.client
        
        # List all objects with the prefix
        paginator = client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix)
        
        objects_to_delete = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Double-check that the key starts with our exact prefix
                    if obj['Key'].startswith(folder_prefix):
                        objects_to_delete.append({'Key': obj['Key']})
                    else:
                        print(f"⚠️  WARNING: Found object with unexpected prefix: {obj['Key']}")
        
        if not objects_to_delete:
            print("✅ Folder is already empty or doesn't exist.")
            return True
        
        print(f"📁 Found {len(objects_to_delete)} objects to delete")
        
        # Show what will be deleted for verification
        print("\n🔍 Objects that will be deleted:")
        for obj in objects_to_delete[:10]:  # Show first 10
            print(f"  - {obj['Key']}")
        if len(objects_to_delete) > 10:
            print(f"  ... and {len(objects_to_delete) - 10} more objects")
        
        # Final safety confirmation
        print(f"\n⚠️  FINAL SAFETY CHECK:")
        print(f"🔒 Will delete EXACTLY: s3://{bucket_name}/{folder_prefix}")
        print(f"🔒 Will NOT touch: s3://{bucket_name}/accounts/")
        print(f"🔒 Will NOT touch: ANY other external folders outside accounts/")
        print(f"🔒 Will NOT touch: ANY other files or folders")
        
        print(f"\n🔍 EXACT FOLDER TO DELETE:")
        print(f"   {folder_prefix}")
        print(f"\n🔍 EXAMPLES OF WHAT WILL BE PRESERVED:")
        print(f"   - accounts/68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/")
        print(f"   - any_other_external_folder/")
        print(f"   - any_other_file_or_folder")
        
        print(f"\n⚠️  FINAL CONFIRMATION REQUIRED:")
        print(f"To confirm deletion of: {folder_prefix}")
        print(f"You must type exactly: DELETE")
        
        final_confirm = input(f"\nType 'DELETE' to confirm deletion: ").strip()
        if final_confirm != "DELETE":
            print("❌ Deletion cancelled - you must type exactly 'DELETE' to confirm.")
            return False
        
        # Delete in batches of 1000 (S3 limit)
        batch_size = 1000
        total_deleted = 0
        
        for i in range(0, len(objects_to_delete), batch_size):
            batch = objects_to_delete[i:i + batch_size]
            
            delete_response = client.delete_objects(
                Bucket=bucket_name,
                Delete={
                    'Objects': batch,
                    'Quiet': False
                }
            )
            
            deleted_count = len(delete_response.get('Deleted', []))
            total_deleted += deleted_count
            
            print(f"  ✅ Deleted batch {i//batch_size + 1}: {deleted_count} objects")
            
            # Print any errors
            if 'Errors' in delete_response:
                for error in delete_response['Errors']:
                    print(f"  ❌ Error deleting {error['Key']}: {error['Message']}")
        
        print(f"\n🎉 Successfully deleted {total_deleted} objects from s3://{bucket_name}/{folder_prefix}")
        return True
        
    except Exception as e:
        logger.error(f"Error in batch delete: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("🗑️  S3 SPECIFIC External Project Folder Deletion Tool")
    print("=" * 70)
    print("This will delete ONLY: s3://testusman123/68db6b18bee4320874c73a71/project/68dca627a340f24038389ad5/")
    print("")
    print("🔒 SAFETY GUARANTEE:")
    print("  ✅ Will preserve: accounts/ folder and ALL its contents")
    print("  ✅ Will preserve: ANY other external folders outside accounts/")
    print("  ✅ Will preserve: ANY other files or folders in the bucket")
    print("  ❌ Will delete: ONLY this exact folder path")
    print("=" * 70)
    
    # Show current bucket structure first
    show_bucket_structure()
    
    # Choose deletion method
    method = input("\nChoose deletion method:\n1. Individual file deletion (safer, shows progress)\n2. Batch deletion (faster, for large folders)\nEnter choice (1 or 2): ").strip()
    
    if method == "1":
        success = delete_external_project_folder()
    elif method == "2":
        success = delete_folder_recursive()
    else:
        print("❌ Invalid choice. Exiting.")
        success = False
    
    if success:
        print("\n✅ Deletion completed successfully!")
    else:
        print("\n❌ Deletion failed or was cancelled.")
