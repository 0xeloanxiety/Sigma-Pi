"""auditor/train/prepare_data.py

The main data preparation script.
Parses SolidiFI and SmartBugs-curated, extracts functions using tree-sitter,
applies the multi-hot labels, deduplicates, and emits train/eval JSONL files.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Import our previous modules
from auditor.parsing import extract_functions, FunctionChunk
from auditor.swc.catalog import (
    solidifi_folder_to_class,
    smartbugs_category_to_class,
    empty_label,
    TARGET_CLASSES,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# We assume this is run from the root of the repo (the ΣΠ folder)
DATA_DIR = Path("data")
SOLIDIFI_DIR = DATA_DIR / "SolidiFI-benchmark" / "buggy_contracts"
SMARTBUGS_JSON = DATA_DIR / "smartbugs-curated" / "vulnerabilities.json"
SMARTBUGS_ROOT = DATA_DIR / "smartbugs-curated"

OUT_DIR = Path("auditor/train/processed_data")


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_text(text: str) -> str:
    """Hash function text to detect duplicates."""
    # Strip whitespace to catch trivial duplicates
    clean_text = " ".join(text.split())
    return hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

def is_overlapping(func: FunctionChunk, start_line: int, end_line: int) -> bool:
    """Check if a buggy line range falls inside the function's line range."""
    # A bug overlaps if the bug's start is before the function's end, 
    # and the bug's end is after the function's start.
    return max(func.start_line, start_line) <= min(func.end_line, end_line)


# ── Step 1: Process SolidiFI (Training Data) ──────────────────────────────────

def process_solidifi() -> List[Dict[str, Any]]:
    print("Processing SolidiFI (Train set)...")
    dataset = []
    
    # Iterate over the 7 bug type folders
    if not SOLIDIFI_DIR.exists():
        print(f"Warning: {SOLIDIFI_DIR} not found.")
        return []
        
    for folder in SOLIDIFI_DIR.iterdir():
        if not folder.is_dir():
            continue
            
        target_class = solidifi_folder_to_class(folder.name)
        
        # Iterate over buggy_N.sol files
        for sol_file in folder.glob("buggy_*.sol"):
            # Find matching BugLog_N.csv
            bug_num = sol_file.stem.split("_")[1]
            csv_file = folder / f"BugLog_{bug_num}.csv"
            
            if not csv_file.exists():
                continue
                
            # Parse the CSV to get buggy line ranges
            # Format: loc, length, bug type, approach
            bug_ranges = []
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        start = int(row["loc"])
                        length = int(row["length"])
                        bug_ranges.append((start, start + length - 1))
                    except (ValueError, KeyError):
                        continue
            
            # Extract functions
            chunks = extract_functions(sol_file)
            
            for chunk in chunks:
                label = empty_label()
                
                # Check if this function contains any of the injected bugs
                has_bug = any(is_overlapping(chunk, r[0], r[1]) for r in bug_ranges)
                
                if has_bug and target_class is not None:
                    label[target_class.index] = 1
                
                dataset.append({
                    "contract": sol_file.name,
                    "function_name": chunk.name,
                    "text": chunk.text,
                    "label": label,
                    "source": "solidifi"
                })
                
    return dataset


# ── Step 2: Process SmartBugs (Eval Data) ─────────────────────────────────────

def process_smartbugs() -> List[Dict[str, Any]]:
    print("Processing SmartBugs-curated (Eval set)...")
    dataset = []
    
    if not SMARTBUGS_JSON.exists():
        print(f"Warning: {SMARTBUGS_JSON} not found.")
        return []
        
    with open(SMARTBUGS_JSON, "r", encoding="utf-8") as f:
        sb_data = json.load(f)
        
    for contract_info in sb_data:
        sol_path = SMARTBUGS_ROOT / contract_info["path"]
        if not sol_path.exists():
            continue
            
        # Get all vulnerable lines for this contract, grouped by category
        vuln_lines_by_category = {}
        for vuln in contract_info.get("vulnerabilities", []):
            cat = vuln["category"]
            if cat not in vuln_lines_by_category:
                vuln_lines_by_category[cat] = []
            # 'lines' is a list of line numbers
            vuln_lines_by_category[cat].extend(vuln["lines"])
            
        chunks = extract_functions(sol_path)
        
        for chunk in chunks:
            label = empty_label()
            
            # Check overlap for each category
            for cat, lines in vuln_lines_by_category.items():
                target_class = smartbugs_category_to_class(cat)
                if target_class is not None:
                    # If any vulnerable line falls inside this function chunk
                    if any(chunk.start_line <= line <= chunk.end_line for line in lines):
                        label[target_class.index] = 1
                        
            dataset.append({
                "contract": contract_info["name"],
                "function_name": chunk.name,
                "text": chunk.text,
                "label": label,
                "source": "smartbugs"
            })
            
    return dataset

# ── Step 3 (new): Process SmartBugs-wild (Slither-weak augmentation) ─────────
# CAVEAT: SWC-107 and SWC-104 labels here come from Slither (weak labels).
# Access control (index 3) is intentionally excluded — stays human-labeled only.
# These examples augment the train set only. eval.jsonl is never touched.

WILD_LABELS_JSON = OUT_DIR / "wild_slither_labels.json"
WILD_CONTRACTS_DIR = DATA_DIR / "smartbugs-wild" / "contracts"


def process_smartbugs_wild() -> List[Dict[str, Any]]:
    """
    Read pre-computed Slither labels (from label_wild.py), run tree-sitter on
    each contract, map flagged lines → function labels. Returns train examples.
    """
    print("Processing SmartBugs-wild (weak-label augmentation)...")

    if not WILD_LABELS_JSON.exists():
        print(f"  Warning: {WILD_LABELS_JSON} not found. Skipping wild augmentation.")
        print("  Run: python -m auditor.train.label_wild on Colab first.")
        return []

    with open(WILD_LABELS_JSON, "r") as f:
        label_data = json.load(f)

    # Build lookup: filename → {label_idx: [lines]}
    labels_by_contract: Dict[str, Dict[int, List[int]]] = {}
    for entry in label_data:
        name = entry["contract"]
        # Only keep contracts where Slither actually found something
        if entry.get("lines_by_idx"):
            labels_by_contract[name] = {
                int(k): v for k, v in entry["lines_by_idx"].items()
            }

    print(f"  Contracts with at least one finding: {len(labels_by_contract)}")

    dataset = []
    missing = 0

    for contract_name, lines_by_idx in labels_by_contract.items():
        sol_path = WILD_CONTRACTS_DIR / contract_name
        if not sol_path.exists():
            missing += 1
            continue

        chunks = extract_functions(sol_path)
        if not chunks:
            continue

        for chunk in chunks:
            label = empty_label()
            for label_idx, flagged_lines in lines_by_idx.items():
                # label_idx is 0 (SWC-107) or 2 (SWC-104) — never 1 or 3
                if any(chunk.start_line <= line <= chunk.end_line
                       for line in flagged_lines):
                    label[label_idx] = 1

            dataset.append({
                "contract": contract_name,
                "function_name": chunk.name,
                "text": chunk.text,
                "label": label,
                "source": "smartbugs_wild_weak",  # document weak-label origin
            })

    print(f"  Missing .sol files: {missing}")
    print(f"  Total functions extracted: {len(dataset)}")
    return dataset


# ── Step 3: Dedup and Save ────────────────────────────────────────────────────

def dedup_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_hashes: Set[str] = set()
    unique_data = []
    
    # Count how many positive labels we drop vs keep
    dropped_positives = 0
    kept_positives = 0
    
    for item in dataset:
        h = hash_text(item["text"])
        is_positive = sum(item["label"]) > 0
        
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_data.append(item)
            if is_positive:
                kept_positives += 1
        else:
            if is_positive:
                # If we've seen this exact code before, but now it has a bug label,
                # we have a choice. Usually we just drop the duplicate to prevent leakage.
                dropped_positives += 1
                
    print(f"  Total functions: {len(dataset)}")
    print(f"  Unique functions: {len(unique_data)} (Dropped {len(dataset) - len(unique_data)} duplicates)")
    print(f"  Positive labels kept: {kept_positives}, dropped as duplicate: {dropped_positives}")
    return unique_data

def save_jsonl(dataset: List[Dict[str, Any]], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train: SolidiFI (ground truth) + SmartBugs-wild (Slither weak labels)
    train_raw = process_solidifi()
    wild_raw = process_smartbugs_wild()
    if wild_raw:
        train_raw = train_raw + wild_raw
        print(f"\nCombined train set before dedup: {len(train_raw)} functions")

    print("Deduplicating train set...")
    train_clean = dedup_dataset(train_raw)

    # Eval: SmartBugs-curated (human labels only — NEVER augmented)
    eval_raw = process_smartbugs()
    print("Deduplicating SmartBugs (eval)...")
    eval_clean = dedup_dataset(eval_raw)

    train_path = OUT_DIR / "train.jsonl"
    eval_path  = OUT_DIR / "eval.jsonl"

    save_jsonl(train_clean, train_path)
    save_jsonl(eval_clean, eval_path)

    print("\nDone!")
    print(f"Saved {len(train_clean)} train examples to {train_path}")
    print(f"Saved {len(eval_clean)} eval examples to {eval_path}")

    print("\nTrain Set Class Distribution:")
    for i, target in enumerate(TARGET_CLASSES):
        count = sum(1 for item in train_clean if item["label"][i] == 1)
        print(f"  {target.title} ({target.swc_id}): {count} vulnerable functions")

    print("\nEval Set Class Distribution (SmartBugs-curated, human labels):")
    for i, target in enumerate(TARGET_CLASSES):
        count = sum(1 for item in eval_clean if item["label"][i] == 1)
        print(f"  {target.title} ({target.swc_id}): {count} vulnerable functions")

if __name__ == "__main__":
    main()
