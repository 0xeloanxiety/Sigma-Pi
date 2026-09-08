"""auditor/parsing.py

Reads a Solidity source file and returns all its functions,
each with its name, line range (1-indexed), and full source text.

This module is the single Solidity parsing surface.
It is called identically at training time (prepare_data.py)
and at inference time (the classifier detector).
DO NOT change the chunking logic after training — that would cause
distribution shift between training data and real inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tree_sitter_language_pack import get_parser


# ── Node types we treat as "functions" ───────────────────────────────────────
#
# tree-sitter-solidity uses these node type names.
# We include constructors and fallback because they can contain vulnerabilities
# (e.g. a wrong constructor name is an access-control bug in SWC-118).
# We include modifiers because access-control logic often lives inside them.

_FUNCTION_NODE_TYPES = {
    "function_definition",
    "constructor_definition",
    "fallback_receive_definition",
    "modifier_definition",
}

# Build the parser once at module load — it's expensive to recreate.
_PARSER = get_parser("solidity")


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FunctionChunk:
    """One extracted function (or constructor / modifier / fallback)."""
    name: str        # e.g. "withdraw", "<constructor>", "<fallback>"
    start_line: int  # 1-indexed, inclusive
    end_line: int    # 1-indexed, inclusive
    text: str        # full source text of this function


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_name(node, source_bytes: bytes) -> str:
    """
    Pull the function/modifier name out of a tree-sitter node.
    For constructors and fallbacks, which have no identifier child,
    return a fixed placeholder string.
    """
    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode(
                "utf-8", errors="replace"
            )
    # Node types without an identifier:
    if node.type == "constructor_definition":
        return "<constructor>"
    if node.type == "fallback_receive_definition":
        return "<fallback>"
    return "<unknown>"


def _walk(node, source_bytes: bytes, out: list[FunctionChunk]) -> None:
    """
    Recursively walk the parse tree.
    When a function-level node is found, record it and STOP recursing —
    Solidity does not nest functions so there is nothing deeper to find.
    """
    if node.type in _FUNCTION_NODE_TYPES:
        name = _extract_name(node, source_bytes)
        # tree-sitter line numbers are 0-indexed; convert to 1-indexed.
        start_line = node.start_point[0] + 1
        end_line   = node.end_point[0]   + 1
        text = source_bytes[node.start_byte:node.end_byte].decode(
            "utf-8", errors="replace"
        )
        out.append(FunctionChunk(
            name=name,
            start_line=start_line,
            end_line=end_line,
            text=text,
        ))
        return   # do NOT recurse further — nothing nested inside a function

    for child in node.children:
        _walk(child, source_bytes, out)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_functions(sol_path: "Path | str") -> list[FunctionChunk]:
    """
    Parse a .sol file and return one FunctionChunk per function found.

    Returns an empty list (not an exception) if the file cannot be read
    or if tree-sitter produces an empty tree. The caller decides what to
    do with unreadable files.

    Args:
        sol_path: path to the .sol file (str or Path)

    Returns:
        List of FunctionChunk, in source order.
    """
    sol_path = Path(sol_path)
    try:
        source_bytes = sol_path.read_bytes()
    except OSError:
        return []

    tree = _PARSER.parse(source_bytes)
    chunks: list[FunctionChunk] = []
    _walk(tree.root_node, source_bytes, chunks)
    return chunks


# ── Quick smoke test ──────────────────────────────────────────────────────────
# Run directly:  python auditor/parsing.py
# It should print one line per function found in buggy_1.sol.

if __name__ == "__main__":
    import sys
    from pathlib import Path

    test_file = Path(
        "data/SolidiFI-benchmark/buggy_contracts/Re-entrancy/buggy_1.sol"
    )
    chunks = extract_functions(test_file)

    if not chunks:
        print("ERROR: no functions found — check the path or tree-sitter install")
        sys.exit(1)

    print(f"Found {len(chunks)} functions in {test_file.name}:\n")
    for c in chunks:
        print(f"  lines {c.start_line:>4}-{c.end_line:<4}  {c.name}")
