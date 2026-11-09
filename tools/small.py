from pathlib import Path

root = Path(__file__).resolve().parent.parent / "data"

for file in root.rglob("*"):
    if file.is_file() and (
        file.suffix.lower() in {".txt", ".yaml", ".yml", ".json"}
    ):
        print(file)