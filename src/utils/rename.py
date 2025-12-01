import os
from pathlib import Path

def batch_rename_videos(folder_path, prefix="clear_inc", start_num=1, extension=".mp4"):
    """
    Batch rename video files
    
    Args:
        folder_path: directory path
        prefix: filename prefix
        start_num: starting number
        extension: file extension
    """
    folder = Path(folder_path)
    
    video_files = sorted(folder.glob(f"*{extension}"))
    
    if not video_files:
        print(f"No {extension} files found in {folder_path}")
        return
    
    print(f"Found {len(video_files)} video files")
    print("\nPreview of renaming:")
    print("-" * 60)
    
    renaming_plan = []
    for idx, video_file in enumerate(video_files, start=start_num):
        new_name = f"{prefix}_{idx:03d}{extension}"
        new_path = folder / new_name
        renaming_plan.append((video_file, new_path))
        print(f"{video_file.name:40s} -> {new_name}")
    
    print("-" * 60)
    confirm = input("\nProceed with renaming? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        for old_path, new_path in renaming_plan:
            old_path.rename(new_path)
            print(f"✓ Renamed: {new_path.name}")
        print(f"\n✅ Successfully renamed {len(renaming_plan)} files!")
    else:
        print("❌ Renaming cancelled.")

if __name__ == "__main__":
    folder_path = r"F:\Goodminton\data\raw_videos\clear\test\correct"
    batch_rename_videos(folder_path, prefix="clear_tst_c", start_num=1)