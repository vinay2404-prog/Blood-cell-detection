"""Record a reproducibility manifest for a research run."""
from __future__ import annotations
import sys

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main():
    manifest = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
        "model_path": str(core.MODEL_PATH),
        "model_sha256": core.model_sha256(str(core.MODEL_PATH), core.MODEL_PATH.stat().st_mtime),
        "yolo_config": core.YOLO_CONFIG,
        "thresholds": core.THRESHOLDS,
        "algorithm_version": core.ALGORITHM_VERSION,
        "experiment_version": core.EXPERIMENT_VERSION,
        "python": sys.version,
    }
    out = ROOT / "evaluation_results" / "experiment_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()