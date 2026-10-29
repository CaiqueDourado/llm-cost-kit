"""One cost record for an LLM call, as JSON (docs/example-cost.json).

    python scripts/make_json.py

Prices the Gemini-shaped response of examples/quickstart.py (180,000 prompt tokens, 244 visible
output tokens, 6,099 thinking tokens) with examples/prices.yaml and writes:

- ``usage``: what ``extract_usage(resp)`` reads from the response;
- ``row``: the row ``Ledger.record(...)`` returns and writes to every sink;
- ``price``: the ``Price`` entry of the model in the table (``PriceTable.get(model)``), with its
  ``source`` and ``checked_at``. Prices are USD per 1,000,000 tokens.

The ledger stamps ``at`` with the current time; this script fixes the clock at FIXED_NOW so the
file does not change between runs. No network, no API key: the response is a stand-in object.
Run from the repository root (PyYAML is needed to read the YAML price table).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_cost_kit import Ledger, MemorySink, PriceTable, extract_usage, ledger, target  # noqa: E402

MODEL = "gemini-2.5-flash"
FIXED_NOW = datetime(2026, 10, 6, 14, 30, tzinfo=timezone.utc)
OUT = ROOT / "docs" / "example-cost.json"


class _FixedClock(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D102 - only what Ledger.record calls
        return FIXED_NOW if tz is None else FIXED_NOW.astimezone(tz)


def cost_record() -> dict:
    prices = PriceTable.load(ROOT / "examples" / "prices.yaml")
    resp = NS(usage_metadata=NS(prompt_token_count=180_000, candidates_token_count=244,
                                cached_content_token_count=0, thoughts_token_count=6_099))
    real_clock, ledger.datetime = ledger.datetime, _FixedClock
    try:
        with target(product_id=1):
            row = Ledger(prices, MemorySink()).record(resp, model=MODEL, flow="spec_sheet")
    finally:
        ledger.datetime = real_clock
    return {
        "usage": asdict(extract_usage(resp)),
        "row": row,
        "price": {"model": MODEL, **asdict(prices.get(MODEL))},
    }


def main() -> None:
    OUT.write_text(json.dumps(cost_record(), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
