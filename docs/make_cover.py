"""Cover: a receipt of one LLM call, with the lines most cost trackers get wrong."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent / "img"
INK, MUTED, LINE, BG, BLUE, ORANGE, AQUA = "#111111", "#555555", "#d9dde3", "#fdfdfc", "#2a78d6", "#eb6834", "#1baf7a"
for fam in ("Inter", "DejaVu Sans"):
    try:
        matplotlib.font_manager.findfont(fam, fallback_to_default=False); plt.rcParams["font.family"] = fam; break
    except ValueError:
        pass

T = {
    "": dict(tag="What each LLM call\nreally cost", sub="prices with source and date,\na ledger that never breaks your pipeline",
             head="one call · gemini-2.5-pro", rows=[("prompt", "250,000 tok", "above 200k: high tier", True),
                                                     ("cached", "120,000 tok", "cache price", False),
                                                     ("output", "2,000 tok", "", False),
                                                     ("thinking", "6,000 tok", "billed as output", True),
                                                     ("batch", "", "× 0.5", False)],
             total="total", checked="price checked 2026-10-06 · source: ai.google.dev",
             traps_t="what it gets right", traps=["thinking billed as output", "OpenAI reasoning not counted twice",
                                                  "price tier by the whole prompt", "per call, never summed",
                                                  "unknown model = null, not 0", "quota 429 ≠ rate 429"],
             foot="also: prompt budget by blocks · tokenizer factor · CI guard for unmetered calls"),
    ".pt-BR": dict(tag="Quanto cada chamada\nde LLM custou de verdade", sub="preços com fonte e data,\num registro que nunca derruba o pipeline",
                   head="uma chamada · gemini-2.5-pro", rows=[("prompt", "250.000 tok", "acima de 200k: faixa alta", True),
                                                              ("cache", "120.000 tok", "preço de cache", False),
                                                              ("saída", "2.000 tok", "", False),
                                                              ("thinking", "6.000 tok", "cobrado como saída", True),
                                                              ("lote", "", "× 0,5", False)],
                   total="total", checked="preço conferido em 06/10/2026 · fonte: ai.google.dev",
                   traps_t="o que ela acerta", traps=["thinking cobrado como saída", "reasoning da OpenAI sem contar 2x",
                                                      "faixa decidida pelo prompt inteiro", "por chamada, nunca somado",
                                                      "modelo desconhecido = nulo, não 0", "429 de cota ≠ 429 de limite"],
                   foot="também: orçamento de prompt por blocos · fator do tokenizador · guarda de CI"),
}
# cost of the receipt example, from the example table
from llm_cost_kit import PriceTable
pt = PriceTable.load(Path(__file__).resolve().parents[1] / "examples" / "prices.yaml")
usd = pt.cost("gemini-2.5-pro", 250_000, 2_000, cached_input_tokens=120_000, thinking_tokens=6_000, batch=True)

OUT.mkdir(parents=True, exist_ok=True)
for suf, t in T.items():
    fig = plt.figure(figsize=(11, 5.6), dpi=110); fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 110); ax.set_ylim(0, 56); ax.axis("off")
    # mark: a coin-like tag
    ax.add_patch(FancyBboxPatch((5, 41), 9, 9, boxstyle="round,pad=0,rounding_size=2", fc=BLUE, ec="none"))
    ax.text(9.5, 45.5, "$", color="white", fontsize=22, weight="bold", ha="center", va="center")
    ax.text(17, 45.5, "llm-cost-kit", fontsize=24, weight="bold", color=INK, va="center")
    ax.text(5, 32.5, t["tag"], fontsize=15, color=INK, va="center", linespacing=1.25)
    ax.text(5, 24.5, t["sub"], fontsize=10.5, color=MUTED, va="center", linespacing=1.35)
    # receipt
    rx, ry, rw, rh = 41, 8, 34, 42
    ax.add_patch(FancyBboxPatch((rx, ry), rw, rh, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=LINE, lw=1.2))
    ax.text(rx + 2, ry + rh - 3.5, t["head"], fontsize=10.5, weight="bold", color=INK)
    for i, (k, v, note, warn) in enumerate(t["rows"]):
        y = ry + rh - 9 - i * 5.6
        ax.text(rx + 2, y, k, fontsize=9.5, color=INK)
        ax.text(rx + 17, y, v, fontsize=9.5, color=INK, ha="right", family="monospace")
        if note:
            ax.text(rx + 19, y, note, fontsize=8.2, color=ORANGE if warn else MUTED, va="center")
    ax.plot([rx + 2, rx + rw - 2], [ry + 9.2, ry + 9.2], color=LINE, lw=1, ls=(0, (2, 2)))
    ax.text(rx + 2, ry + 5.8, t["total"], fontsize=11, weight="bold", color=INK)
    ax.text(rx + rw - 2, ry + 5.8, f"US$ {usd:.4f}", fontsize=11, weight="bold", color=INK, ha="right", family="monospace")
    ax.text(rx + 2, ry + 2.2, t["checked"], fontsize=7.6, color=MUTED)
    # traps
    tx = 80
    ax.text(tx, 46, t["traps_t"], fontsize=11, weight="bold", color=INK)
    for j, s in enumerate(t["traps"]):
        ax.text(tx, 40.5 - j * 5, "✓  " + s, fontsize=9.2, color=INK)
    ax.text(5, 3.2, t["foot"], fontsize=8.6, color=MUTED)
    ax.text(105, 3.2, "github.com/CaiqueDourado", fontsize=8.6, color=MUTED, ha="right")
    fig.savefig(OUT / f"cover{suf}.png", facecolor=BG); plt.close(fig)
print(f"ok {usd:.6f}")
