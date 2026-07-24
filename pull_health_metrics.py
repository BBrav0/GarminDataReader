#!/usr/bin/env python3
"""Daily private health metrics pull: Garmin RHR plus Withings body metrics."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()


def ensure_venv() -> None:
    """Re-execute script with venv Python if not already using it."""
    if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        return

    for env_name in (".venv", "venv"):
        venv_python = SCRIPT_DIR / env_name / "bin" / "python3"
        if venv_python.exists():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)


def run_step(script_name: str) -> int:
    print(f"=== {script_name} ===")
    result = subprocess.run([sys.executable, str(SCRIPT_DIR / script_name)], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"WARN: {script_name} exited with status {result.returncode}")
    return result.returncode


def main() -> None:
    ensure_venv()
    results = {
        "pull_rhr.py": run_step("pull_rhr.py"),
        "pull_withings.py": run_step("pull_withings.py"),
    }
    failed = [name for name, code in results.items() if code != 0]
    if failed:
        raise SystemExit(f"Health metrics pull completed with failures: {', '.join(failed)}")
    print("Health metrics pull complete.")


if __name__ == "__main__":
    main()
