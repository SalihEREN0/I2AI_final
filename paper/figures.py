"""Makale şekilleri: A4 sonuçlarından (a4data.json) sütun genişliğinde PDF şekiller üretir.

Şablonun kuralı: şekil etiketleri 8 pt Times New Roman. Sütun genişliği IEEEtran A4 konferans
düzeninde 3,5 in; şekiller o boyda çizilir ki LaTeX ölçeklemesin ve yazı 8 pt kalsın.
Çalıştırma: python figures.py  (çıktı: fig/ablation.pdf, fig/learning.pdf)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
# Proje klasörü: teslim paketinde ve depoda `project/`, geliştirme düzeninde `abalone-proje/`.
PROJ = next((p for p in (HERE.parent / "project", HERE.parent / "abalone-proje") if p.is_dir()), None)
if PROJ is None:
    raise SystemExit("proje klasörü bulunamadı: " + str(HERE.parent / "project"))
DATA = PROJ / "assignments" / "a4-final-agent" / "results" / "a4data.json"   # sunumdaki a4data.json ile aynı dosya
FINAL = PROJ / "experiments" / "final"     # hızlı motorla tekrarlanan maçlar
OUT = HERE / "fig"
COL = 3.5                                   # IEEEtran sütun genişliği (in)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e1"
# Kategorik sıra (dataviz varsayılan paleti, ilk altı yuva; validate_palette.js light: geçti).
# Üçü beyazda 3:1'in altında: kimliği renk tek başına taşımasın diye çizgi deseni ve lejant da var.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DASH = ["-", (0, (5, 1.5)), (0, (1.2, 1.2)), (0, (5, 1.5, 1.2, 1.5)), (0, (2.5, 1.5)), (0, (8, 2))]
LABEL = {"centre": "centrality"}            # koddaki özellik adı → metindeki ve Tablo V'teki ad

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": INK2, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})
# Kenar boşlukları elle: "tight" kırpma genişliği değiştirir, LaTeX'te ölçeklenen şeklin yazısı 8 pt kalmaz.


def _final(key):
    """Hızlı motorla yapılan maçın özeti (experiments/final), henüz yoksa None."""
    p = FINAL / key / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    return {"score": s["a_score"], "ci": s["a_score_ci95"], "w": s["a_wins"], "d": s["draws"], "l": s["a_losses"],
            "games": s["games"]}


def ablation(d):
    # Her satır: tam ajan, bir parçası kapatılmış kendisine karşı; puan = (G + B/2) / N, %95 GA.
    # İçi boş gri: A4 motoru (50 oyun). Dolu siyah: hızlı motor (sıralama, TT ve killer 200 oyun, diğerleri 50).
    rows = [("e2-ablate-learning", "Learned weights"), ("e2-ablate-quiescence", "Quiescence search"),
            ("e2-ablate-ordering", "Move ordering"), ("e2-ablate-tt", "Transposition table"),
            ("e2-ablate-killers", "Killer moves")]
    new = {k: _final(k + "-200") or _final(k) for k, _ in rows}
    both = all(new.values())
    fig, ax = plt.subplots(figsize=(COL, 1.62 if both else 1.36))
    fig.subplots_adjust(left=0.265, right=0.995, bottom=0.24 if both else 0.27, top=0.83 if both else 0.91)
    for i, (key, name) in enumerate(rows):
        y = len(rows) - 1 - i
        series = [(d["m"][key], y + (0.17 if both else 0), "white", INK2)]
        if both:
            series.append((new[key], y - 0.17, INK, INK))
        for m, yy, face, edge in series:
            lo, hi = m["ci"]
            ax.plot([lo, hi], [yy, yy], color=edge, lw=0.9, solid_capstyle="butt")
            ax.plot(m["score"], yy, "o", ms=3.6, mfc=face if both else INK, mec=edge if both else "white",
                    mew=0.8 if both else 0.6, zorder=3)
        m = new[key] if both else d["m"][key]
        ax.text(1.03, y, f'{m["score"]:.2f} ({m["w"]}–{m["d"]}–{m["l"]})', va="center", ha="left", color=INK2)
    if both:
        ax.plot([], [], "o", ms=3.6, mfc="white", mec=INK2, mew=0.8, label="presented engine")
        ax.plot([], [], "o", ms=3.6, mfc=INK, mec=INK, label="new engine")
        ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, handletextpad=0.3,
                  columnspacing=1.2, borderaxespad=0.2)
    ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.text(0.5, len(rows) - 0.45, "equal", ha="center", va="bottom", color=INK2)
    ax.set_yticks(range(len(rows)), [n for _, n in reversed(rows)])
    ax.tick_params(axis="y", length=0)
    # Eksen 1,0'da biter; sağdaki boşluk puan ve G–B–M yazıları için (200 oyunda "0.59 (112–12–76)" kadar uzun).
    xmax = 1.4
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.spines["bottom"].set_bounds(0.35, 1.0)
    ax.set_xlim(0.35, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("Score of the full agent against the agent without the component", x=(0.675 - 0.35) / (xmax - 0.35))
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    fig.savefig(OUT / "ablation.pdf")


def learning(d):
    t = d["tdleaf"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL, 2.4), gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.72})
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.33, top=0.93)
    # (a) Her kayıtlı sürüm, elle verilen ağırlıklara karşı 50 oyun.
    xs = [c["games"] for c in d["curve"]]
    ys = [c["score"] for c in d["curve"]]
    for x, c in zip(xs, d["curve"]):
        a.plot([x, x], c["ci"], color=INK, lw=0.8)
    a.plot(xs, ys, "-o", color=INK, lw=0.9, ms=3.2, mec="white", mew=0.5, zorder=3)
    a.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)))
    a.text(1250, 0.52, "equal", ha="right", va="bottom", color=INK2)
    a.set_ylim(0.45, 1.02)
    a.set_xlim(0, 1260)
    a.set_ylabel("Score")                     # rakibi başlık söylüyor; uzun etiket şeklin üst kenarına değiyordu
    a.set_title("(a) Checkpoints against the hand-set weights (50 games each)", fontsize=8, loc="left", pad=3)
    a.grid(axis="y", color=GRID, lw=0.4)
    a.set_axisbelow(True)
    # (b) Ağırlıkların öz-oyun boyunca yolu (her nokta 40 oyunda bir).
    gx = [p["games"] for p in t["path"]]
    for i, f in enumerate(t["features"]):
        b.plot([0] + gx, [t["init"][f]] + [p["w"][f] for p in t["path"]], color=SERIES[i], ls=DASH[i], lw=1.1,
               label=LABEL.get(f, f))
    b.axhline(0, color=INK2, lw=0.4)
    b.set_yticks([-10, 0, 10, 20])
    b.set_xlim(0, 1260)
    b.set_xlabel("Self-play games")
    b.set_ylabel("Weight")
    b.set_title("(b) TDLeaf($\\lambda$) weights during self-play", fontsize=8, loc="left", pad=3)
    b.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.4), handlelength=2.6, columnspacing=1.2)
    b.grid(axis="y", color=GRID, lw=0.4)
    b.set_axisbelow(True)
    fig.savefig(OUT / "learning.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    data = json.loads(DATA.read_text(encoding="utf-8"))
    ablation(data)
    learning(data)
    print("fig/ablation.pdf, fig/learning.pdf")
