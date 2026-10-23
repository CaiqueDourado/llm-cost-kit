"""CI guard: every file that calls an LLM must also record its cost.

An audit of the author's crawler found 48 of 63 production scripts calling an LLM without
recording any cost, and there was no way to know what the previous month had cost. The 48
were fixed in one day; this check keeps the 49th from being born without a meter.

It checks presence, not correctness: a file that calls ``*.models.generate_content``,
``*.chat.completions.create``, ``*.messages.create`` or ``*.responses.create`` must contain a
call to one of the recorder names. Pricing correctness is the price table's job.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Iterable

LLM_CALLS = ("models.generate_content", "chat.completions.create", "messages.create", "responses.create",
             "models.generate_content_stream")
RECORDERS = ("record", "ledger.record")
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", "tests", "test"}


def calls_llm(tree: ast.AST, patterns: Iterable[str] = LLM_CALLS) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            try:
                name = ast.unparse(node.func)
            except Exception:
                continue
            if any(name.endswith(p) for p in patterns):
                return True
    return False


def unmetered(root: str, recorders: Iterable[str] = RECORDERS, patterns: Iterable[str] = LLM_CALLS,
              skip: set[str] = SKIP_DIRS) -> list[str]:
    """Python files under ``root`` that call an LLM and never call a recorder."""
    rec = re.compile(r"\b(" + "|".join(re.escape(r.split(".")[-1]) for r in recorders) + r")\s*\(")
    missing = []
    for dp, dn, fs in os.walk(root):
        dn[:] = [d for d in dn if d not in skip]
        for f in sorted(fs):
            if not f.endswith(".py"):
                continue
            path = os.path.join(dp, f)
            try:
                src = open(path, encoding="utf-8", errors="ignore").read()
                tree = ast.parse(src)
            except (SyntaxError, ValueError):
                continue
            if calls_llm(tree, patterns) and not rec.search(src):
                missing.append(os.path.relpath(path, root))
    return missing
