"""auditor/swc/catalog.py

Single source of truth for the 4 target vulnerability classes
and how each dataset's labels map to them.

Nothing in here does any parsing or ML — it's pure data.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SWCClass:
    """One target vulnerability class."""
    swc_id: str    # e.g. "SWC-107"
    title: str     # human label, e.g. "Reentrancy"
    index: int     # position in the multi-hot label vector [0..3]


# ── The 4 target classes ──────────────────────────────────────────────────────
#
# The label vector is always in THIS order:
#   index 0 → SWC-107
#   index 1 → SWC-115
#   index 2 → SWC-104
#   index 3 → SWC-105/106
#
# So a function with reentrancy + unchecked call gets: [1, 0, 1, 0]

TARGET_CLASSES: list[SWCClass] = [
    SWCClass(swc_id="SWC-107",     title="Reentrancy",      index=0),
    SWCClass(swc_id="SWC-115",     title="tx.origin",        index=1),
    SWCClass(swc_id="SWC-104",     title="Unchecked Call",   index=2),
    SWCClass(swc_id="SWC-105/106", title="Access Control",   index=3),
]

# Lookup by SWC id — used when you have an id and need the index.
BY_SWC_ID: dict[str, SWCClass] = {c.swc_id: c for c in TARGET_CLASSES}


# ── SolidiFI folder name → SWC class ─────────────────────────────────────────
#
# SolidiFI has 7 bug type folders. Only 3 of them map to our target classes.
# The rest (TOD, Timestamp, Overflow) return None — those contracts still
# contribute CLEAN functions as negative examples, but the injected label
# itself is not a target class.

_SOLIDIFI_MAP: dict[str, Optional[str]] = {
    "Re-entrancy":           "SWC-107",
    "tx.origin":             "SWC-115",
    "Unchecked-Send":        "SWC-104",
    "Unhandled-Exceptions":  "SWC-104",   # same class as Unchecked-Send
    "TOD":                   None,
    "Timestamp-Dependency":  None,
    "Overflow-Underflow":    None,
}


def solidifi_folder_to_class(folder_name: str) -> Optional[SWCClass]:
    """
    Given a SolidiFI bug-type folder name, return the SWCClass it maps to,
    or None if it is not one of our 4 target classes.

    Example:
        solidifi_folder_to_class("Re-entrancy")   → SWCClass("SWC-107", ...)
        solidifi_folder_to_class("TOD")            → None
    """
    swc_id = _SOLIDIFI_MAP.get(folder_name)
    if swc_id is None:
        return None
    return BY_SWC_ID[swc_id]


# ── SmartBugs-curated category string → SWC class ────────────────────────────
#
# vulnerabilities.json uses these category strings.
# Only 3 of the 10 DASP categories overlap with our 4 target classes.
# Everything else (bad_randomness, front_running, etc.) → None.

_SMARTBUGS_MAP: dict[str, Optional[str]] = {
    "reentrancy":               "SWC-107",
    "unchecked_low_level_calls": "SWC-104",
    "access_control":           "SWC-105/106",
    # everything below is NOT a target class
    "arithmetic":               None,
    "bad_randomness":           None,
    "denial_of_service":        None,
    "front_running":            None,
    "other":                    None,
    "short_addresses":          None,
    "time_manipulation":        None,
}


def smartbugs_category_to_class(category: str) -> Optional[SWCClass]:
    """
    Given a SmartBugs-curated category string, return the SWCClass,
    or None if not a target class.

    Example:
        smartbugs_category_to_class("reentrancy")     → SWCClass("SWC-107", ...)
        smartbugs_category_to_class("bad_randomness") → None
    """
    swc_id = _SMARTBUGS_MAP.get(category)
    if swc_id is None:
        return None
    return BY_SWC_ID[swc_id]


# ── Helper: empty multi-hot label vector ──────────────────────────────────────

def empty_label() -> list[int]:
    """Return [0, 0, 0, 0] — the starting label for every function."""
    return [0] * len(TARGET_CLASSES)
