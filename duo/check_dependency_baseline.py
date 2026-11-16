"""Reject new dependency conflicts while preserving the prior Intel profile."""

import json
from pathlib import Path
import subprocess

expected = {
    "The package `aiortc` requires `av>=14.0.0,<15.0.0`, but `15.1.0` is installed",
    "The package `pyannote-audio` requires `torchcodec>=0.7.0`, but it's not installed",
}
result = subprocess.run(
    ["/usr/local/bin/uv", "pip", "check", "--python", "/opt/speaches-venv/bin/python"],
    text=True,
    capture_output=True,
    check=False,
)
issues = {line for line in (result.stdout + result.stderr).splitlines() if line.startswith("The package `")}
if issues != expected or result.returncode != 1:
    raise RuntimeError(f"Dependency conflicts differ from the pinned previous profile: {sorted(issues)}")
Path("/app/duo/dependency-baseline.json").write_text(
    json.dumps({"inherited_conflicts": sorted(issues)}, indent=2) + "\n"
)
