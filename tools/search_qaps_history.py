
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
    ".git", "__pycache__", ".venv", "venv",
    "data", "runs", "static", "evaluation_results"
}

EXTENSIONS = {".py", ".yaml", ".yml", ".md", ".json"}

KEYWORDS = [
    "threshold",
    "calibrat",
    "validation",
    "qaps",
    "gamma",
    "clahe",
    "median",
    "noise",
]

def search_files(directory):
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines()

            matches = [
                (number, line.strip())
                for number, line in enumerate(lines, start=1)
                if any(word in line.lower() for word in KEYWORDS)
            ]

            if matches:
                print(f"\n--- {path.relative_to(ROOT)} ---")

                for number, line in matches:
                    print(f"{number}: {line}")

        except OSError:
            continue

if __name__ == "__main__":
    search_files(ROOT)
