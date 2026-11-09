import os

# No files or directories are excluded
EXCLUDE_DIRS = set()
EXCLUDE_FILES = set()

def generate_tree(dir_path, prefix="", current_dir_name=""):
    # Get all entries sorted alphabetically without any filtering
    entries = sorted(os.listdir(dir_path))
    
    entries_count = len(entries)
    tree_str = ""
    
    for index, entry in enumerate(entries):
        path = os.path.join(dir_path, entry)
        is_last = (index == entries_count - 1)
        connector = "└── " if is_last else "├── "
        
        tree_str += f"{prefix}{connector}{entry}\n"
        
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            tree_str += generate_tree(path, prefix=prefix + extension, current_dir_name=entry)
            
    return tree_str

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    project_name = os.path.basename(root_dir)
    
    tree_output = f"{project_name}/\n" + generate_tree(root_dir, current_dir_name=project_name)
    
    output_filename = "folder_structure.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(tree_output)
        
    print(f"Folder structure successfully saved to {output_filename}")