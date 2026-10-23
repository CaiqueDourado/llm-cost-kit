"""Price and record LLM calls. Run: python examples/quickstart.py"""
from pathlib import Path
from types import SimpleNamespace as NS

from llm_cost_kit import Ledger, PriceTable, SQLiteSink, read_rows, summarize, target, timer

prices = PriceTable.load(Path(__file__).with_name("prices.yaml"))
ledger = Ledger(prices, SQLiteSink("costs.sqlite"))

# A Gemini-shaped response: 6,099 thinking tokens to write 244 visible ones.
resp = NS(usage_metadata=NS(prompt_token_count=180_000, candidates_token_count=244,
                            cached_content_token_count=0, thoughts_token_count=6_099))
for product_id in (1, 2):
    with target(product_id=product_id):
        with timer() as t:
            pass  # resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        ledger.record(resp, model="gemini-2.5-flash", flow="spec_sheet", latency_ms=t.ms)

for g in summarize(read_rows("costs.sqlite"), by="meta.product_id"):
    print(g["key"], g["calls"], f"US$ {g['cost_usd']:.4f}")
