"""llm-cost-kit: what each LLM call really cost, with prices you can trace to a source."""

from . import budget, errors, guard, tokens, usage
from .budget import allocate, boundary_cutter, cut_at_boundary
from .errors import kind as error_kind, should_retry
from .ledger import (CallbackSink, CSVSink, JSONLSink, Ledger, MemorySink, SQLiteSink, read_rows, summarize,
                     target, timer)
from .prices import UNCHECKED, HighTier, OffPeak, Price, PriceTable
from .usage import Usage, extract as extract_usage

__version__ = "0.1.0"
__all__ = ["PriceTable", "Price", "HighTier", "OffPeak", "UNCHECKED", "Ledger", "target", "timer",
           "JSONLSink", "CSVSink", "SQLiteSink", "CallbackSink", "MemorySink", "summarize", "read_rows",
           "Usage", "extract_usage", "should_retry", "error_kind", "allocate", "cut_at_boundary",
           "boundary_cutter", "budget", "errors", "guard", "tokens", "usage"]
