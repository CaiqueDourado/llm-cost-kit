"""Command line.

    llm-cost-kit prices prices.yaml [--stale 90]        # unchecked, stale and suspicious prices
    llm-cost-kit cost prices.yaml MODEL --in 250000 --out 2000 [--cached N] [--thinking N] [--batch]
    llm-cost-kit report costs.sqlite [--by flow|model|meta.product_id]
    llm-cost-kit check PATH [--recorder NAME]            # files that call an LLM without recording cost
"""

from __future__ import annotations

import argparse
import sys

from .guard import unmetered
from .ledger import read_rows, summarize
from .prices import PriceTable


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="llm-cost-kit")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prices"); s.add_argument("file"); s.add_argument("--stale", type=int, default=90)
    s = sub.add_parser("cost"); s.add_argument("file"); s.add_argument("model")
    s.add_argument("--in", dest="inp", type=int, default=0); s.add_argument("--out", type=int, default=0)
    s.add_argument("--cached", type=int, default=0); s.add_argument("--thinking", type=int, default=0)
    s.add_argument("--images", type=int, default=0); s.add_argument("--batch", action="store_true")
    s.add_argument("--off-peak", action="store_true")
    s = sub.add_parser("report"); s.add_argument("file"); s.add_argument("--by", default="flow")
    s = sub.add_parser("check"); s.add_argument("path", nargs="?", default=".")
    s.add_argument("--recorder", action="append", help="name of your recording function (repeatable); default: record")
    a = p.parse_args(argv)

    if a.cmd == "prices":
        t = PriceTable.load(a.file)
        print(f"{len(t.prices)} models")
        for title, items in (("never checked", t.unchecked()), ("cache price not checked", t.unchecked_cache()),
                             (f"checked more than {a.stale} days ago", t.stale(a.stale)), ("suspicious", t.check_sanity())):
            if items:
                print(f"\n{title}:")
                for i in items:
                    print(f"  {i}")
        return 1 if t.unchecked() or t.check_sanity() else 0

    if a.cmd == "cost":
        t = PriceTable.load(a.file)
        usd = t.cost(a.model, a.inp, a.out, cached_input_tokens=a.cached, thinking_tokens=a.thinking,
                     images=a.images, batch=a.batch, off_peak=a.off_peak)
        if usd is None:
            print(f"{a.model} is not in the table"); return 1
        tier = t.tier(a.model, a.inp)
        print(f"US$ {usd:.6f}" + (f"  (tier {tier})" if tier else ""))
        return 0

    if a.cmd == "report":
        groups = summarize(read_rows(a.file), by=a.by)
        print(f"{'key':<32} {'calls':>7} {'input':>12} {'output':>10} {'thinking':>10} {'US$':>10}")
        for g in groups:
            flag = f"  ({g['unpriced_calls']} unpriced)" if g["unpriced_calls"] else ""
            print(f"{str(g['key'])[:32]:<32} {g['calls']:>7} {g['input_tokens']:>12} {g['output_tokens']:>10} "
                  f"{g['thinking_tokens']:>10} {g['cost_usd']:>10.4f}{flag}")
        print(f"{'total':<32} {sum(g['calls'] for g in groups):>7} {'':>34} {sum(g['cost_usd'] for g in groups):>10.4f}")
        return 0

    missing = unmetered(a.path, recorders=a.recorder or ("record",))
    for m in missing:
        print(f"no cost record: {m}")
    print(f"{len(missing)} file(s) call an LLM without recording cost")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
