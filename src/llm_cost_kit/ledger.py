"""One row per LLM call, written somewhere you can query later.

``Ledger.record`` never raises: a cost log that breaks the pipeline costs more than what it
measures. It does not fail in silence either. A failed write goes to stderr and to
``Ledger.errors``. In the author's crawler, a flow ran 14 times with zero cost rows and no
trace, because the failure was only printed in debug mode.
"""

from __future__ import annotations

import contextlib
import contextvars
import csv
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

from . import usage as U
from .prices import PriceTable

_target: contextvars.ContextVar[dict] = contextvars.ContextVar("llm_cost_target", default={})


@contextlib.contextmanager
def target(**fields):
    """Declare what the calls inside the block are for (product id, article, user...).

    Every row recorded inside gets these fields in ``meta`` without changing the signature
    of the function that calls the model::

        for product in products:
            with target(product_id=product["id"]):
                call_llm(prompt)        # a generic wrapper that only receives the prompt

    Context-local: safe with threads and asyncio. ``meta`` passed to ``record`` wins over it.
    """
    token = _target.set({**_target.get(), **{k: v for k, v in fields.items() if v is not None}})
    try:
        yield
    finally:
        _target.reset(token)


class Timer:
    ms: int | None = None


@contextlib.contextmanager
def timer():
    """``with timer() as t: resp = ...`` then ``t.ms``."""
    t, start = Timer(), time.perf_counter()
    try:
        yield t
    finally:
        t.ms = int((time.perf_counter() - start) * 1000)


# ------------------------------------------------------------------------------- sinks
FIELDS = ["at", "flow", "model", "provider", "input_tokens", "cached_input_tokens", "output_tokens",
          "thinking_tokens", "images", "batch", "off_peak", "tier", "cost_usd", "latency_ms",
          "price_checked_at", "meta"]


class JSONLSink:
    def __init__(self, path):
        self.path, self._lock = Path(path), threading.Lock()

    def write(self, row: dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class CSVSink:
    def __init__(self, path):
        self.path, self._lock = Path(path), threading.Lock()

    def write(self, row: dict) -> None:
        with self._lock:
            new = not self.path.exists()
            with self.path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow({k: json.dumps(row[k], ensure_ascii=False) if k == "meta" else row.get(k) for k in FIELDS})


class SQLiteSink:
    """A ``calls`` table; ``llm-cost-kit report file.sqlite`` summarises it."""

    def __init__(self, path):
        self.path, self._lock = str(path), threading.Lock()
        with sqlite3.connect(self.path) as db:
            db.execute("create table if not exists calls (" + ", ".join(
                f"{c} {'real' if c == 'cost_usd' else 'text' if c in ('at', 'flow', 'model', 'provider', 'tier', 'price_checked_at', 'meta') else 'integer'}"
                for c in FIELDS) + ")")

    def write(self, row: dict) -> None:
        vals = [json.dumps(row[c], ensure_ascii=False) if c == "meta" else row.get(c) for c in FIELDS]
        with self._lock, sqlite3.connect(self.path) as db:
            db.execute(f"insert into calls ({', '.join(FIELDS)}) values ({', '.join('?' * len(FIELDS))})", vals)


class CallbackSink:
    """Any function ``fn(row)``: a Postgres insert, a log line, a metrics client."""

    def __init__(self, fn: Callable[[dict], None]):
        self.fn = fn

    def write(self, row: dict) -> None:
        self.fn(row)


class MemorySink:
    def __init__(self):
        self.rows: list[dict] = []

    def write(self, row: dict) -> None:
        self.rows.append(row)


# ------------------------------------------------------------------------------- ledger
class Ledger:
    """Prices each call and writes a row.

    >>> ledger = Ledger(PriceTable.load("prices.yaml"), SQLiteSink("costs.sqlite"))
    >>> with timer() as t:
    ...     resp = client.models.generate_content(model=MODEL, contents=prompt)
    >>> ledger.record(resp, model=MODEL, flow="spec_sheet", latency_ms=t.ms, meta={"product_id": 12})
    """

    def __init__(self, prices: PriceTable, *sinks, warn_unknown: bool = True) -> None:
        self.prices = prices
        self.sinks = list(sinks) or [MemorySink()]
        self.errors: list[str] = []
        self.warn_unknown = warn_unknown
        self._warned: set[str] = set()

    def record(self, resp=None, *, model: str, flow: str, meta: Mapping | None = None,
               latency_ms: int | None = None, batch: bool = False, off_peak: bool | None = None,
               images: int = 0, input_tokens: int | None = None, output_tokens: int | None = None,
               cached_input_tokens: int | None = None, thinking_tokens: int | None = None) -> dict | None:
        """Write one row; returns it, or ``None`` when nothing could be measured. Never raises.

        Token counts come from ``resp`` (see :mod:`llm_cost_kit.usage`); explicit arguments
        override them. ``off_peak=None`` decides by the clock for models with an off-peak
        window. An unknown model is recorded with ``cost_usd = None``, never 0: zero makes an
        unpriced model look like the cheapest one.
        """
        try:
            u = U.extract(resp) if resp is not None else None
            u = u or U.Usage()
            ti = input_tokens if input_tokens is not None else u.input_tokens
            to = output_tokens if output_tokens is not None else u.output_tokens
            tc = cached_input_tokens if cached_input_tokens is not None else u.cached_input_tokens
            tt = thinking_tokens if thinking_tokens is not None else u.thinking_tokens
            if not any((ti, to, tc, tt, images)):
                return None
            if off_peak is None:
                off_peak = self.prices.is_off_peak(model)
            cost = self.prices.cost(model, ti, to, cached_input_tokens=tc, thinking_tokens=tt,
                                    images=images, batch=batch, off_peak=off_peak)
            p = self.prices.get(model)
            if p is None and self.warn_unknown and model not in self._warned:
                self._warned.add(model)
                print(f"[llm-cost-kit] {model} is not in the price table: cost recorded as null", file=sys.stderr)
            row = {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "flow": flow, "model": model, "provider": u.provider or None,
                "input_tokens": ti, "cached_input_tokens": tc, "output_tokens": to, "thinking_tokens": tt,
                "images": images, "batch": bool(batch), "off_peak": bool(off_peak),
                "tier": self.prices.tier(model, ti), "cost_usd": cost, "latency_ms": latency_ms,
                "price_checked_at": str(p.checked_at) if p and p.checked_at else None,
                "meta": {**_target.get(), **(meta or {})},
            }
            for s in self.sinks:
                s.write(row)
            return row
        except Exception as e:
            msg = f"[llm-cost-kit] failed to record {flow} {model}: {type(e).__name__}: {e}"
            self.errors.append(msg)
            print(msg, file=sys.stderr)
            return None


def summarize(rows: Iterable[Mapping], by: str = "flow") -> list[dict]:
    """Calls, tokens and cost grouped by ``flow``, ``model`` or a ``meta`` key (``meta.product_id``)."""
    out: dict = {}
    for r in rows:
        key = (r.get("meta") or {}).get(by[5:]) if by.startswith("meta.") else r.get(by)
        g = out.setdefault(key, {"key": key, "calls": 0, "input_tokens": 0, "output_tokens": 0,
                                 "thinking_tokens": 0, "cost_usd": 0.0, "unpriced_calls": 0})
        g["calls"] += 1
        for k in ("input_tokens", "output_tokens", "thinking_tokens"):
            g[k] += r.get(k) or 0
        if r.get("cost_usd") is None:
            g["unpriced_calls"] += 1
        else:
            g["cost_usd"] += r["cost_usd"]
    return sorted(out.values(), key=lambda g: -g["cost_usd"])


def read_rows(path) -> list[dict]:
    """Rows from a JSONL, CSV or SQLite ledger file."""
    path = str(path)
    if path.endswith((".sqlite", ".db", ".sqlite3")):
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute("select * from calls")]
    elif path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in ("input_tokens", "cached_input_tokens", "output_tokens", "thinking_tokens"):
                r[k] = int(r[k] or 0)
            r["cost_usd"] = float(r["cost_usd"]) if r["cost_usd"] not in ("", None) else None
    else:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for r in rows:
        if isinstance(r.get("meta"), str):
            r["meta"] = json.loads(r["meta"] or "{}")
    return rows
