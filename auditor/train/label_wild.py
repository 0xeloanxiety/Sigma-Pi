"""auditor/train/label_wild.py

Runs Slither on a sample of SmartBugs-wild contracts to generate
weak reentrancy/unchecked-call labels. Output is a JSON file mapping
contract filename → list of flagged line numbers per class.

Run this on Colab (NOT locally — it takes hours).
Usage:
  python -m auditor.train.label_wild --sample 5000 --workers 4
"""

import argparse
import json
import subprocess
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── The Slither detectors we care about ──────────────────────────────────────
# Map Slither check name → our label index (matches TARGET_CLASSES order)
SLITHER_TO_LABEL_IDX = {
    "reentrancy-eth":      0,   # SWC-107
    "reentrancy-no-eth":   0,
    "reentrancy-benign":   0,
    "unchecked-lowlevel":  2,   # SWC-104
    "unchecked-send":      2,
}
# IMPORTANT: no index 1 (tx.origin) or 3 (access control) — those stay clean

WILD_DIR = Path("data/smartbugs-wild/contracts")
OUT_FILE  = Path("auditor/train/processed_data/wild_slither_labels.json")
TIMEOUT_SECONDS = 45   # kill Slither if it hangs


def run_slither_on_file(sol_path: Path) -> dict:
    """
    Run Slither on one .sol file. Returns a dict:
      { "contract": filename, "lines_by_idx": {0: [line, ...], 2: [line, ...]} }
    or None if Slither fails/times out.
    """
    try:
        result = subprocess.run(
            ["slither", str(sol_path), "--json", "-",
             "--disable-color", "--exclude-optimization",
             "--exclude-informational", "--exclude-low"],
            capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS
        )
        if not result.stdout.strip():
            return None

        data = json.loads(result.stdout)
        if not data.get("success"):
            return None

        lines_by_idx: dict[int, list[int]] = {}
        for detector in data.get("results", {}).get("detectors", []):
            check = detector.get("check", "")
            label_idx = SLITHER_TO_LABEL_IDX.get(check)
            if label_idx is None:
                continue   # not a class we care about

            for element in detector.get("elements", []):
                sm = element.get("source_mapping", {})
                for line in sm.get("lines", []):
                    lines_by_idx.setdefault(label_idx, []).append(line)

        return {
            "contract": sol_path.name,
            "lines_by_idx": {str(k): v for k, v in lines_by_idx.items()}
            # JSON keys must be strings
        }

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=5000,
                        help="Number of contracts to sample (default 5000)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel Slither workers (default 4)")
    args = parser.parse_args()

    all_files = list(WILD_DIR.glob("*.sol"))
    print(f"Found {len(all_files)} contracts in wild dataset.")

    random.seed(42)
    sample = random.sample(all_files, min(args.sample, len(all_files)))
    print(f"Sampling {len(sample)} contracts with {args.workers} workers...")

    results = []
    failed  = 0
    positives = {0: 0, 2: 0}   # count positives per class

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_slither_on_file, f): f for f in sample}
        for i, future in enumerate(as_completed(futures)):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(sample)} | "
                      f"ok={len(results)} failed={failed}")
            r = future.result()
            if r is None:
                failed += 1
                continue
            results.append(r)
            for idx_str in r["lines_by_idx"]:
                if int(idx_str) in positives:
                    positives[int(idx_str)] += 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f)

    print(f"\nDone! {len(results)} contracts labeled, {failed} failed/timed out.")
    print(f"Contracts with reentrancy (SWC-107) : {positives[0]}")
    print(f"Contracts with unchecked call (SWC-104): {positives[2]}")
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
