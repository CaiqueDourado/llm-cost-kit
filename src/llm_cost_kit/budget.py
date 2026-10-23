"""Cut a prompt to fit without destroying material in silence.

Cutting the end of an oversized prompt kills whatever block happens to come last, and the
log only says "prompt truncated". Two better cuts:

- :func:`allocate`: the prompt has SEVERAL blocks of material (manufacturer page, reviews,
  transcripts, web pages). Each block gets a share of the budget by weight. Shares are
  handed out in rounds: a block that fits closes at its real size and gives the rest back to
  the blocks still open. Nobody loses more than the excess, and no budget is left unused.
- :func:`cut_at_boundary`: the prompt is ONE block of repeated items (one product after the
  other). It cuts between items, never in the middle of one, and reports which items fell out.

The tokenizer is passed in: anything with ``encode(str) -> list`` and ``decode(list) -> str``
(tiktoken, a Hugging Face tokenizer wrapper).
"""

from __future__ import annotations

import re
from typing import Callable, Mapping, Sequence


def allocate(blocks: Mapping[str, str], available: int, weights: Mapping[str, float], encoding,
             cutters: Mapping[str, Callable[[str, int], str]] | None = None) -> tuple[dict, dict]:
    """Fit ``blocks`` in ``available`` tokens, splitting the cut by ``weights``.

    Returns ``(blocks_after, report)``. ``report`` has one entry per block that was cut, with
    ``tokens_before``, ``tokens_after`` (measured after the cut, not assumed) and
    ``tokens_cut``. A block with no weight gets nothing. ``cutters`` lets a block cut itself,
    for example at item boundaries with :func:`cut_at_boundary`.

    Why rounds: with a single redistribution, the share a small block does not use stays
    parked in it. Measured on a real prompt (152,298 tokens available, 158,488 of
    material): the excess was 6,190 tokens, the old two-pass split cut 31,239 from the
    largest block and left 25,049 of budget unused. In rounds it cuts exactly 6,190.
    """
    tok = {k: encoding.encode(v or "") for k, v in blocks.items()}
    if sum(len(t) for t in tok.values()) <= available:
        return {k: v or "" for k, v in blocks.items()}, {}

    shares: dict[str, int] = {}
    open_, left = set(tok), available
    while open_:
        total_w = sum(weights.get(k, 0) for k in open_)
        if total_w <= 0:
            shares.update({k: 0 for k in open_})
            break
        proposal = {k: int(left * weights.get(k, 0) / total_w) for k in open_}
        closed = [k for k in open_ if len(tok[k]) <= proposal[k]]
        if not closed:
            shares.update(proposal)
            break
        for k in closed:
            shares[k] = len(tok[k])
            left -= len(tok[k])
            open_.discard(k)

    out, report = {}, {}
    for k, t in tok.items():
        if len(t) <= shares[k]:
            out[k] = blocks[k] or ""
            continue
        cut = (cutters or {}).get(k)
        out[k] = cut(blocks[k] or "", shares[k]) if cut else encoding.decode(t[: shares[k]])
        after = len(encoding.encode(out[k]))
        report[k] = {"tokens_before": len(t), "tokens_after": after, "tokens_cut": len(t) - after}
    return out, report


def cut_at_boundary(text: str, max_tokens: int, encoding, boundary: re.Pattern | str,
                    extra_boundaries: Sequence[str] = ()) -> tuple[str, str, dict]:
    """Cut ``text`` to ``max_tokens`` at the last item boundary that fits.

    ``boundary`` is a regex (with one group for the item label) matching the start of each
    item, e.g. ``r"^=== PRODUCT: (.+?) ===$"`` with ``re.M``. ``extra_boundaries`` are literal
    strings that also count as boundaries (section headers).

    Returns ``(kept, rest, report)``. ``report["labels_out"]`` lists the items that did not
    fit. ``report["no_boundary"]`` is True when not even the first item fits and the cut fell
    back to the middle of the text: the case to shout about in the log.

    The material to cut must be at the END of the prompt, after the instructions.
    """
    rx = re.compile(boundary, re.M) if isinstance(boundary, str) else boundary
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text, "", {}
    raw = encoding.decode(tokens[:max_tokens])
    limit = len(raw)
    starts = [m.start() for m in rx.finditer(text)]
    starts += [p for e in extra_boundaries if (p := text.find(e)) > 0]
    fit = [p for p in starts if 0 < p <= limit]
    # An item that ends exactly at the limit still fits: only whitespace before the next one.
    later = [p for p in starts if p > limit]
    if later and not text[limit:min(later)].strip():
        fit.append(min(later))
    if not fit:
        return raw, encoding.decode(tokens[max_tokens:]), {
            "no_boundary": True, "tokens_before": len(tokens), "tokens_after": max_tokens,
            "labels_out": [m.group(1) for m in rx.finditer(text)],
            "extras_out": [e for e in extra_boundaries if e in text]}
    pos = max(fit)
    kept, rest = text[:pos], text[pos:]
    return kept, rest, {
        "no_boundary": False, "tokens_before": len(tokens), "tokens_after": len(encoding.encode(kept)),
        "labels_out": [m.group(1) for m in rx.finditer(rest)],
        "extras_out": [e for e in extra_boundaries if e in rest]}


def boundary_cutter(boundary: re.Pattern | str, encoding) -> Callable[[str, int], str]:
    """A cutter for :func:`allocate` that cuts at item boundaries."""
    return lambda text, n: cut_at_boundary(text, n, encoding, boundary)[0]
