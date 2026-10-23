import re

import pytest

from llm_cost_kit import PriceTable, allocate, boundary_cutter, cut_at_boundary, error_kind, should_retry
from llm_cost_kit.guard import unmetered
from llm_cost_kit.tokens import cliff_check, measure_factor, safe_ceiling


class Words:
    """A toy tokenizer: one token per word."""
    def encode(self, s):
        return s.split()

    def decode(self, t):
        return " ".join(t)


W = Words()


def blk(n, w="x"):
    return " ".join([w] * n)


def test_allocate_cuts_only_the_excess_in_rounds():
    blocks = {"maker": blk(12_013), "reviews": blk(59_123), "sites": blk(87_352)}
    out, rep = allocate(blocks, 152_298, {"maker": 0.5, "reviews": 0.3, "sites": 0.2}, W)
    assert list(rep) == ["sites"]
    assert rep["sites"]["tokens_cut"] == 158_488 - 152_298
    assert sum(len(v.split()) for v in out.values()) == 152_298


def test_allocate_no_cut_when_it_fits():
    out, rep = allocate({"a": "um dois", "b": None}, 10, {"a": 1}, W)
    assert out == {"a": "um dois", "b": ""} and rep == {}


def test_block_without_weight_gets_nothing():
    out, rep = allocate({"a": blk(10), "b": blk(10)}, 15, {"a": 1}, W)
    assert out["b"] == "" and len(out["a"].split()) == 10


TEXT = "instrucoes fixas\n=== PRODUCT: A ===\n" + blk(10, "a") + "\n=== PRODUCT: B ===\n" + blk(10, "b") + \
       "\n=== PRODUCT: C ===\n" + blk(10, "c")


def test_cut_at_boundary_keeps_whole_items_and_names_the_rest():
    kept, rest, rep = cut_at_boundary(TEXT, 30, W, r"^=== PRODUCT: (.+?) ===$")
    assert kept.rstrip().endswith(blk(10, "b").split()[-1]) and "PRODUCT: C" not in kept
    assert rep["labels_out"] == ["C"] and rep["no_boundary"] is False


def test_cut_without_boundary_is_flagged():
    _, _, rep = cut_at_boundary(blk(50), 10, W, r"^=== PRODUCT: (.+?) ===$")
    assert rep["no_boundary"] is True


def test_boundary_cutter_inside_allocate():
    out, rep = allocate({"items": TEXT, "other": blk(5)}, 30, {"items": 0.9, "other": 0.1},
                        W, cutters={"items": boundary_cutter(r"^=== PRODUCT: (.+?) ===$", W)})
    assert "PRODUCT: C" not in out["items"]


@pytest.mark.parametrize("msg,retry,kind", [
    ("Error code: 429 - {'error': {'code': 'insufficient_quota'}}", False, "account"),
    ("429 billing_not_active", False, "account"),
    ("Error code: 402 - Insufficient Balance", False, "account"),
    ("Error code: 403 - This model is not available in your subscription", False, "account"),
    ("Error code: 429 - Rate limit exceeded", True, "transient"),
    ("503 UNAVAILABLE", True, "transient"),
    ("Connection refused", False, "service"),
    ("KeyError: 'choices'", False, "other"),
])
def test_error_classification(msg, retry, kind):
    assert should_retry(msg) is retry and error_kind(msg) == kind


P = PriceTable.from_dict({"pro": {"input": 1, "output": 1, "source": "x",
                                  "high_tier": {"above": 200000, "input": 2, "output": 1.5}},
                          "flash": {"input": 1, "output": 1, "source": "x"}})


def test_safe_ceiling_by_model():
    assert safe_ceiling(1.234, model="pro", prices=P) == 162_074
    assert safe_ceiling(1.234, model="flash", prices=P) == int(500_000 / 1.234)
    assert safe_ceiling(1.234, model="unknown", prices=P) == 162_074
    assert safe_ceiling(1.2, model="pro", prices=P, margin=0.1) < safe_ceiling(1.2, model="pro", prices=P)


def test_measure_factor():
    f = measure_factor([(100_000, 103_500), (100_000, 123_400)])
    assert f["min"] == pytest.approx(1.035) and f["max"] == pytest.approx(1.234)


def test_cliff_check_warns_skips_and_never_raises():
    logs = []
    assert cliff_check("p", "pro", lambda p, m: 210_000, P, your_tokens=170_000, log=logs.append) == 210_000
    assert any("high tier" in l for l in logs)
    assert cliff_check("p", "pro", lambda p, m: 1 / 0, P, log=logs.append) is None
    assert cliff_check("p", "pro", lambda p, m: 999, P, your_tokens=1000) is None


def test_guard_finds_unmetered_files(tmp_path):
    (tmp_path / "ok.py").write_text("r = client.models.generate_content(model=M, contents=p)\nledger.record(r, model=M, flow='x')\n")
    (tmp_path / "bad.py").write_text("r = client.chat.completions.create(model=M, messages=[])\n")
    (tmp_path / "none.py").write_text("print('hi')\n")
    assert unmetered(str(tmp_path)) == ["bad.py"]
