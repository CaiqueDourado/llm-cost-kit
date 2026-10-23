from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from llm_cost_kit import UNCHECKED, PriceTable

T = PriceTable.load(Path(__file__).resolve().parents[1] / "examples" / "prices.yaml")


def test_unknown_model_costs_None_not_zero():
    assert T.cost("no-such-model", 1000, 1000) is None


def test_simple_cost():
    assert T.cost("gemini-2.5-flash", 1_000_000, 1_000_000) == pytest.approx(0.30 + 2.50)


def test_cache_comes_out_of_input_instead_of_adding():
    full = T.cost("gemini-2.5-flash", 1_000_000, 0)
    cached = T.cost("gemini-2.5-flash", 1_000_000, 0, cached_input_tokens=1_000_000)
    assert cached == pytest.approx(0.03) and cached < full


def test_unchecked_cache_is_charged_at_full_price_and_listed():
    assert T.cost("mistral-small-2603", 1_000_000, 0, cached_input_tokens=1_000_000) == pytest.approx(0.15)
    assert T.unchecked_cache() == ["mistral-small-2603"]


def test_thinking_is_billed_as_output():
    assert T.cost("gemini-2.5-flash", 0, 244, thinking_tokens=6099) == pytest.approx((244 + 6099) * 2.50 / 1e6)


def test_high_tier_by_whole_prompt_and_not_same_factor():
    low = T.cost("gemini-2.5-pro", 200_000, 1_000_000)
    high = T.cost("gemini-2.5-pro", 200_001, 1_000_000)
    assert high > low
    # output goes to 1.5x, input doubles
    assert T.cost("gemini-2.5-pro", 0, 1_000_000) == pytest.approx(10.0)
    assert T.cost("gemini-2.5-pro", 250_000, 0, cached_input_tokens=250_000) == pytest.approx(0.25 * 0.25)
    assert T.tier("gemini-2.5-pro", 250_000) == ">200k" and T.tier("gemini-2.5-flash", 250_000) is None


def test_tier_starts_where_the_provider_says():
    assert T.tier("qwen3.7-flash", 40_000) == ">32k"


def test_batch_halves_only_who_has_batch():
    assert T.cost("gemini-2.5-flash", 1_000_000, 0, batch=True) == pytest.approx(0.15)
    assert T.cost("deepseek-flash", 1_000_000, 0, batch=True) == pytest.approx(0.30)


def test_off_peak_by_clock():
    peak = datetime(2026, 10, 14, 2, 0, tzinfo=timezone.utc)      # Wednesday 02:00 UTC
    cheap = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
    weekend = datetime(2026, 10, 17, 2, 0, tzinfo=timezone.utc)
    assert T.cost("deepseek-flash", 1_000_000, 0, off_peak=None, when=peak) == pytest.approx(0.30)
    assert T.cost("deepseek-flash", 1_000_000, 0, off_peak=None, when=cheap) == pytest.approx(0.15)
    assert T.is_off_peak("deepseek-flash", weekend)
    assert not T.is_off_peak("gemini-2.5-flash", cheap)


def test_per_request_fee():
    assert T.cost("sonar", 100, 100) == pytest.approx(0.008 + 200 / 1e6)


def test_unchecked_and_stale_lists():
    t = PriceTable.from_dict({"models": {"a": {"input": 1, "output": 2, "source": "x"},
                                         "b": {"input": 1, "output": 2, "source": "x", "checked_at": "2026-01-01"},
                                         "c": {"input": 1, "output": 2, "source": "x", "retired": True}}})
    assert t.unchecked() == ["a"]
    assert t.stale(90, today=date(2026, 10, 22)) == ["b"]


def test_sanity_catches_typos():
    t = PriceTable.from_dict({"m": {"input": 0.075, "output": 2.5, "cache_read": 0.3, "source": "x"}})
    assert t.check_sanity() == ["m: cache_read 0.3 > input 0.075"]
    assert T.check_sanity() == []


def test_local_model_costs_zero_not_None():
    assert T.cost("qwen3:8b", 5000, 500) == 0
