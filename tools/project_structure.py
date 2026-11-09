
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path.cwd()

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".idea",
    ".vscode"
}

# File extensions to exclude
EXCLUDED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff",
    ".txt"
}

# Exclude YOLO annotation files
EXCLUDED_FILES = {
    "classes.txt"
}

def show_tree(directory, prefix=""):
    items = []

    for item in directory.iterdir():
        if item.is_dir():
            if item.name not in EXCLUDED_DIRS:
                items.append(item)

        elif item.is_file():
            if (
                item.suffix.lower() not in EXCLUDED_EXTENSIONS
                and item.name not in EXCLUDED_FILES
                and item.suffix.lower() != ".cache"
            ):
                items.append(item)

    items.sort(key=lambda x: (x.is_file(), x.name.lower()))

    for index, item in enumerate(items):
        is_last = index == len(items) - 1
        connector = "└── " if is_last else "├── "

        print(prefix + connector + item.name)

        if item.is_dir():
            extension = "    " if is_last else "│   "
            show_tree(item, prefix + extension)


print(ROOT.name + "/")
show_tree(ROOT)
