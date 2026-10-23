"""Count with your tokenizer, get billed by theirs.

Prompts are usually sized with tiktoken. Gemini bills with its own tokenizer, which counted
3.5% to 23.4% MORE on 87 real prompts of the author's crawler, depending on the content
(URLs, timestamps and ids tokenize worse). With a ceiling of 198,000 tiktoken tokens,
eight real prompts crossed 200,000 on Google's count: the price tier where input and cache
cost double.

There is no single factor: measure it per flow (:func:`measure_factor`), size the prompt with
:func:`safe_ceiling`, and check the big ones with the provider's free counter
(:func:`cliff_check`).
"""

from __future__ import annotations

from typing import Callable, Iterable

from .prices import PriceTable

#: Default target when the model has no high tier: how much material to send in one call.
DEFAULT_TARGET = 500_000


def measure_factor(pairs: Iterable[tuple[int, int]]) -> dict:
    """``pairs`` of (your_count, provider_count) for real prompts of one flow.

    Returns min, max and mean of provider/yours. Use the MAX for ceilings: erring low costs
    the high tier.
    """
    f = [p / y for y, p in pairs if y]
    if not f:
        raise ValueError("no pairs")
    return {"n": len(f), "min": min(f), "max": max(f), "mean": sum(f) / len(f)}


def safe_ceiling(factor: float, *, model: str | None = None, prices: PriceTable | None = None,
                 target: int | None = None, margin: float = 0.0) -> int:
    """How many of YOUR tokens fit under the provider's ``target``.

    ``target`` defaults to the model's high-tier threshold, or :data:`DEFAULT_TARGET` for a
    model without one. An unknown model gets 200,000: when in doubt, under-billing is worse.
    ``margin`` keeps room for content that tokenizes worse than what was measured.
    """
    if target is None:
        p = prices.get(model) if (prices and model) else None
        if p is None:
            target = 200_000
        elif p.high_tier:
            target = p.high_tier.above
        else:
            target = DEFAULT_TARGET
    return int(target / (factor * (1.0 + margin)))


def cliff_check(prompt, model: str, count: Callable[[object, str], int], prices: PriceTable, *,
                your_tokens: int | None = None, floor: int = 150_000,
                log: Callable[[str], None] = print) -> int | None:
    """Count ``prompt`` with the provider and warn when it crosses the model's high tier.

    ``count(prompt, model)`` is the provider's counter (for Gemini,
    ``lambda p, m: client.models.count_tokens(model=m, contents=p).total_tokens``; it is free
    but takes seconds). Below ``floor`` of your tokens it is skipped: with the worst factor
    measured in the author's crawler (1.234), 150,000 becomes 185,100, still under 200,000.
    Without ``your_tokens`` it always runs.

    Never raises and never blocks: returns the provider's count, or ``None``.
    """
    if your_tokens is not None and your_tokens < floor:
        return None
    try:
        total = int(count(prompt, model))
    except Exception as e:
        log(f"[llm-cost-kit] could not count with the provider: {type(e).__name__}")
        return None
    if your_tokens:
        log(f"[llm-cost-kit] your count {your_tokens} -> provider {total} (factor {total / your_tokens:.3f})")
    p = prices.get(model)
    if p and p.high_tier and total > p.high_tier.above:
        log(f"[llm-cost-kit] {model}: prompt of {total} tokens is above {p.high_tier.above}: "
            f"this call is billed at the high tier")
    return total
