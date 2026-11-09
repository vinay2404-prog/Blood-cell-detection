import json
from pathlib import Path

source = Path("evaluation_results/image_details.json")
output = Path("evaluation_results/paired_compact.json")

with open(source, "r", encoding="utf-8") as f:
    data = json.load(f)

summary = {
    "total_images": len(data),
    "record_keys": list(data[0].keys()) if data else [],
    "records": []
}

for record in data:
    summary["records"].append({
        "image_id": record.get("image_id"),
        "ground_truth_objects": record.get("ground_truth_objects"),
        "baseline_predictions": record.get("baseline_predictions", []),
        "qaps_predictions": record.get("qaps_predictions", [])
    })

with open(output, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Created:", output)