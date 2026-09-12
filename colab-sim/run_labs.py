"""Execute every lab notebook inside the colab-sim container and report
which ones ran clean, end to end, with no pre-installed data-science stack
to lean on — the same self-bootstrap a fresh Colab runtime forces.

Run from the repo root (this script is invoked FROM INSIDE the container by
docker-run.sh; it does not build or start Docker itself):

    python colab-sim/run_labs.py [notebook ...]

With no arguments, runs every day1/day2/day3 *.ipynb. Writes the executed
notebook to colab-sim/out/<name>.ipynb (with real captured outputs) and
prints a pass/fail line per notebook. Exits non-zero if any notebook errors.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "colab-sim" / "out"


def find_notebooks() -> list[Path]:
    nbs = []
    for day in ("day1", "day2", "day3"):
        nbs.extend(sorted((ROOT / day).glob("*.ipynb")))
    return nbs


def run_one(nb_path: Path) -> tuple[bool, float, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / nb_path.name
    start = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=1800",
            "--output",
            str(out_path),
            str(nb_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    ok = proc.returncode == 0
    log = proc.stdout + "\n" + proc.stderr
    return ok, elapsed, log


def main() -> int:
    targets = [Path(p) for p in sys.argv[1:]] or find_notebooks()
    if not targets:
        print("no notebooks found")
        return 1

    failures = []
    for nb in targets:
        print(f"--- executing {nb.relative_to(ROOT)} ---", flush=True)
        ok, elapsed, log = run_one(nb)
        status = "PASS" if ok else "FAIL"
        print(f"{status} ({elapsed:.1f}s) {nb.relative_to(ROOT)}")
        if not ok:
            failures.append(nb)
            print(log[-4000:])

    print("\n==== summary ====")
    for nb in targets:
        mark = "FAIL" if nb in failures else "ok"
        print(f"  [{mark}] {nb.relative_to(ROOT)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
