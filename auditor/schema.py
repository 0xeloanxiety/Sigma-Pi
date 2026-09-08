"""The Finding schema — the shared record every detector emits.

Slither, the classifier, and the LLM all produce different-shaped output.
Each one translates its result into a `Finding`, so the trust layer can compare
them by a common key: (swc_id, function).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# A pattern describing a valid SWC id: "SWC-107", or "SWC-105/106" (access control).
_SWC_RE = re.compile(r"^SWC-\d{3}(?:/\d{3})?$")


class Detector(str, Enum):
    """The fixed set of detectors. Using a name like Detector.SLITHER instead of
    the bare string "slither" means a typo becomes an error, not silent bad data."""

    SLITHER = "slither"
    CLASSIFIER = "classifier"
    LLM = "llm"


class Finding(BaseModel):
    """One normalized finding. Every detector emits this exact shape."""

    # extra="forbid": reject unknown fields (catches typos like `funtion=`).
    # frozen=True:    a Finding can't be changed after it's created (immutable).
    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- the 6 required core fields (the shared contract) ---
    swc_id: str = Field(..., description="e.g. 'SWC-107' or 'SWC-105/106'")
    title: str = Field(..., description="short human label, e.g. 'Reentrancy'")
    function: Optional[str] = Field(..., description="enclosing function; None = contract-level")
    line: Optional[int] = Field(..., ge=1, description="1-indexed source line, if known")
    source: Detector = Field(..., description="which detector emitted this")
    raw_confidence: Optional[float] = Field(..., ge=0.0, le=1.0, description="pre-calibration score in [0,1]")

    # --- optional extras (they have defaults, so you may leave them out) ---
    contract: Optional[str] = Field(default=None, description="enclosing contract name")
    end_line: Optional[int] = Field(default=None, ge=1)
    message: Optional[str] = Field(default=None, description="fuller reason (esp. the LLM)")

    @field_validator("swc_id")
    @classmethod
    def _check_swc_id(cls, value: str) -> str:
        """Reject anything that isn't a real SWC id (this is 'grounding')."""
        if not _SWC_RE.match(value):
            raise ValueError(f"{value!r} is not a valid SWC id")
        return value

    @property
    def key(self) -> tuple[str, Optional[str]]:
        """The trust layer's voting key: group findings by (swc_id, function)."""
        return (self.swc_id, self.function)
