import json
from types import SimpleNamespace as NS

import pytest

from llm_cost_kit import JSONLSink, Ledger, MemorySink, PriceTable, SQLiteSink, read_rows, summarize, target
from llm_cost_kit.usage import extract

T = PriceTable.from_dict({"g": {"input": 1, "output": 10, "cache_read": 0.1, "source": "x", "checked_at": "2026-10-06"},
                          "o": {"input": 1, "output": 10, "cache_read": 0.5, "source": "x"},
                          "a": {"input": 1, "output": 10, "cache_read": 0.1, "source": "x"}})

GEMINI = NS(usage_metadata=NS(prompt_token_count=1000, candidates_token_count=244, cached_content_token_count=400,
                              thoughts_token_count=6099))
OPENAI = NS(usage=NS(prompt_tokens=1000, completion_tokens=500, prompt_tokens_details=NS(cached_tokens=200),
                     completion_tokens_details=NS(reasoning_tokens=300)))
DEEPSEEK = NS(usage=NS(prompt_tokens=1000, completion_tokens=50, prompt_tokens_details=None,
                       completion_tokens_details=None, prompt_cache_hit_tokens=900))
ANTHROPIC = NS(type="message", usage=NS(input_tokens=100, output_tokens=50, cache_read_input_tokens=800,
                                         cache_creation_input_tokens=100))


def test_gemini_thinking_is_separate_and_billed():
    u = extract(GEMINI)
    assert (u.input_tokens, u.output_tokens, u.thinking_tokens, u.cached_input_tokens) == (1000, 244, 6099, 400)
    assert u.billed_output == 6343


def test_openai_reasoning_is_not_counted_twice():
    u = extract(OPENAI)
    assert u.output_tokens == 200 and u.thinking_tokens == 300 and u.billed_output == 500


def test_deepseek_cache_field():
    assert extract(DEEPSEEK).cached_input_tokens == 900


def test_anthropic_input_includes_cache_reads_and_writes():
    u = extract(ANTHROPIC)
    assert (u.input_tokens, u.cached_input_tokens) == (1000, 800)


def test_unknown_object():
    assert extract(object()) is None


def test_ledger_records_target_and_never_raises(tmp_path):
    mem = MemorySink()
    led = Ledger(T, mem)
    with target(product_id=7):
        row = led.record(GEMINI, model="g", flow="spec", meta={"step": 1})
    assert row["meta"] == {"product_id": 7, "step": 1}
    assert row["cost_usd"] == pytest.approx((600 * 1 + 400 * 0.1 + 6343 * 10) / 1e6)
    assert row["price_checked_at"] == "2026-10-06"

    class Broken:
        def write(self, row):
            raise IOError("disk full")
    bad = Ledger(T, Broken())
    assert bad.record(GEMINI, model="g", flow="spec") is None and "disk full" in bad.errors[0]


def test_unknown_model_row_has_null_cost():
    mem = MemorySink()
    Ledger(T, mem, warn_unknown=False).record(GEMINI, model="nope", flow="x")
    assert mem.rows[0]["cost_usd"] is None


def test_nothing_measured_writes_nothing():
    mem = MemorySink()
    assert Ledger(T, mem).record(None, model="g", flow="x") is None and mem.rows == []


def test_sinks_round_trip_and_summary(tmp_path):
    db, jl = tmp_path / "c.sqlite", tmp_path / "c.jsonl"
    led = Ledger(T, SQLiteSink(db), JSONLSink(jl))
    for pid in (1, 1, 2):
        with target(product_id=pid):
            led.record(OPENAI, model="o", flow="compare")
    led.record(GEMINI, model="g", flow="spec")
    for path in (db, jl):
        rows = read_rows(path)
        assert len(rows) == 4
        by_flow = {g["key"]: g for g in summarize(rows)}
        assert by_flow["compare"]["calls"] == 3
        by_product = {g["key"]: g["calls"] for g in summarize(rows, by="meta.product_id")}
        assert by_product[1] == 2 and by_product[2] == 1
