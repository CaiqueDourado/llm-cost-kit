"""Price table with provenance, and the cost of one call.

Every price carries where it came from (``source``) and when someone last read it there
(``checked_at``). A price without a date is an estimate, and an estimate silently turns into a
decision; :meth:`PriceTable.unchecked` lists them so they can be fixed before anyone trusts a
report.

Units: USD per 1,000,000 tokens, as the providers publish them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

#: Cache price not checked on the provider's page. Different from ``None`` (the provider has
#: no cache discount): both are charged at the full input price, but only this one shows up in
#: :meth:`PriceTable.unchecked_cache`, so "caching saved nothing" is not mistaken for a fact.
UNCHECKED = "?"


@dataclass(frozen=True)
class HighTier:
    """Higher prices when the PROMPT is longer than ``above`` tokens (e.g. 200k on some models).

    The tier is decided by the whole prompt, cached part included. Input, output and cache do
    not necessarily scale by the same factor, so each has its own price.
    """
    above: int
    input: float
    output: float
    cache_read: float | str | None = None


@dataclass(frozen=True)
class OffPeak:
    """A cheaper time window. ``peak_utc`` are (start_hour, end_hour) ranges, end excluded."""
    peak_utc: tuple[tuple[int, int], ...]
    factor: float = 0.5
    weekends_off_peak: bool = True

    def is_off_peak(self, when: datetime | None = None) -> bool:
        t = when or datetime.now(timezone.utc)
        if t.tzinfo is not None:
            t = t.astimezone(timezone.utc)
        if self.weekends_off_peak and t.weekday() >= 5:
            return True
        return not any(a <= t.hour < b for a, b in self.peak_utc)


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    source: str
    checked_at: date | None = None
    cache_read: float | str | None = None
    per_image: float | None = None
    per_request: float = 0.0
    batch: bool = True              # has a batch mode at half price
    high_tier: HighTier | None = None
    off_peak: OffPeak | None = None
    retired: bool = False           # kept to price old rows; left out of unchecked()
    note: str = ""


BATCH_FACTOR = 0.5


def _date(v) -> date | None:
    if v in (None, ""):
        return None
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


def price_from_dict(d: Mapping) -> Price:
    ht = d.get("high_tier")
    op = d.get("off_peak")
    return Price(
        input=float(d["input"]), output=float(d["output"]), source=d.get("source", ""),
        checked_at=_date(d.get("checked_at")), cache_read=d.get("cache_read"),
        per_image=d.get("per_image"), per_request=float(d.get("per_request", 0) or 0),
        batch=bool(d.get("batch", True)),
        high_tier=HighTier(int(ht["above"]), float(ht["input"]), float(ht["output"]), ht.get("cache_read")) if ht else None,
        off_peak=OffPeak(tuple(tuple(x) for x in op["peak_utc"]), float(op.get("factor", 0.5)),
                         bool(op.get("weekends_off_peak", True))) if op else None,
        retired=bool(d.get("retired", False)), note=d.get("note", ""),
    )


class PriceTable:
    """Model name -> :class:`Price`.

    >>> table = PriceTable.load("prices.yaml")
    >>> table.cost("gemini-2.5-pro", input_tokens=250_000, output_tokens=2_000, thinking_tokens=6_000)
    """

    def __init__(self, prices: Mapping[str, Price] | None = None) -> None:
        self.prices: dict[str, Price] = dict(prices or {})

    # ---- loading -------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping) -> "PriceTable":
        models = data.get("models", data)
        return cls({name: p if isinstance(p, Price) else price_from_dict(p) for name, p in models.items()})

    @classmethod
    def load(cls, path: str | os.PathLike) -> "PriceTable":
        """Read a YAML (needs PyYAML) or JSON file with a ``models:`` mapping."""
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as e:  # pragma: no cover
                raise ImportError("pip install pyyaml to read YAML price files") from e
            return cls.from_dict(yaml.safe_load(text))
        return cls.from_dict(json.loads(text))

    # ---- queries -------------------------------------------------------------------------
    def get(self, model: str) -> Price | None:
        return self.prices.get(model)

    def __contains__(self, model: str) -> bool:
        return model in self.prices

    def unchecked(self) -> list[str]:
        """Models in use with no ``checked_at``: their cost is an estimate."""
        return sorted(m for m, p in self.prices.items() if not p.checked_at and not p.retired)

    def unchecked_cache(self) -> list[str]:
        """Models whose cache price was not checked: "cache saved nothing" may be a table error."""
        return sorted(m for m, p in self.prices.items() if p.cache_read == UNCHECKED)

    def stale(self, days: int = 90, today: date | None = None) -> list[str]:
        """Models whose price was last checked more than ``days`` ago."""
        today = today or date.today()
        return sorted(m for m, p in self.prices.items()
                      if p.checked_at and not p.retired and (today - p.checked_at).days > days)

    def tier(self, model: str, input_tokens: int) -> str | None:
        """Label of the price tier a prompt falls in, e.g. ``"<=200k"`` or ``">200k"``."""
        p = self.prices.get(model)
        if not p or not p.high_tier:
            return None
        k = p.high_tier.above // 1000
        return f">{k}k" if input_tokens > p.high_tier.above else f"<={k}k"

    def is_off_peak(self, model: str, when: datetime | None = None) -> bool:
        p = self.prices.get(model)
        return bool(p and p.off_peak and p.off_peak.is_off_peak(when))

    # ---- cost ----------------------------------------------------------------------------
    def cost(self, model: str, input_tokens: int = 0, output_tokens: int = 0, *, cached_input_tokens: int = 0,
             thinking_tokens: int = 0, images: int = 0, batch: bool = False,
             off_peak: bool | None = False, when: datetime | None = None) -> float | None:
        """USD of ONE call, or ``None`` when the model is not in the table.

        - ``input_tokens`` is the whole prompt; ``cached_input_tokens`` is the part of it served
          from cache, charged at the cache price.
        - ``thinking_tokens`` are charged as output. Pass them separately ONLY when the provider
          reports them outside ``output_tokens`` (Gemini does; OpenAI's reasoning tokens are
          already inside ``completion_tokens``). :func:`llm_cost_kit.usage.extract` gets this right.
        - The high tier is chosen by the whole prompt, cached part included.
        - ``off_peak=None`` decides by the clock (``when``, default now).

        Price per call, never on summed tokens: a day of calls summed into one prompt crosses
        the high tier that no single call crossed.
        """
        p = self.prices.get(model)
        if p is None:
            return None
        inp, out, cache = p.input, p.output, p.cache_read
        if p.high_tier and (input_tokens or 0) > p.high_tier.above:
            inp, out, cache = p.high_tier.input, p.high_tier.output, p.high_tier.cache_read
        cached = min(cached_input_tokens or 0, input_tokens or 0)
        cache_rate = cache if isinstance(cache, (int, float)) else inp  # None and UNCHECKED: full price
        usd = (((input_tokens or 0) - cached) * inp + cached * cache_rate
               + ((output_tokens or 0) + (thinking_tokens or 0)) * out) / 1_000_000
        usd += p.per_request
        if images and p.per_image:
            usd += images * p.per_image
        if batch and p.batch:
            usd *= BATCH_FACTOR
        if off_peak is None:
            off_peak = self.is_off_peak(model, when)
        if off_peak and p.off_peak:
            usd *= p.off_peak.factor
        return usd

    def check_sanity(self) -> list[str]:
        """Problems that are almost always typos: cache dearer than input, high tier cheaper..."""
        issues = []
        for m, p in self.prices.items():
            if not p.source:
                issues.append(f"{m}: no source")
            if isinstance(p.cache_read, (int, float)) and p.cache_read > p.input:
                issues.append(f"{m}: cache_read {p.cache_read} > input {p.input}")
            if p.high_tier and (p.high_tier.input < p.input or p.high_tier.output < p.output):
                issues.append(f"{m}: high tier cheaper than the normal one")
        return issues


def models_with(table: PriceTable, *, off_peak: bool | None = None, batch: bool | None = None) -> Iterable[str]:
    for m, p in table.prices.items():
        if off_peak is not None and bool(p.off_peak) != off_peak:
            continue
        if batch is not None and p.batch != batch:
            continue
        yield m
