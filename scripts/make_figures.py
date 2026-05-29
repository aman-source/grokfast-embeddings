"""
Regenerate all three publication figures from results/ JSONs.
Outputs PNG (300 dpi) + SVG to figures/.

Run from repo root:
  python scripts/make_figures.py
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_S2 = ROOT / "results" / "stage2"
OUT        = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   13,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.dpi":        150,
})

C_BASE       = "#888888"
C_BASE_LIGHT = "#BBBBBB"
C_EMB        = "#2563EB"
C_EMB_LIGHT  = "#93C5FD"


def load(name: str) -> dict:
    return json.load(open(RESULTS_S2 / f"{name}.json"))


def tok_b(j: dict) -> list:
    return [t / 1e9 for t in j["tokens_seen"]]


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg",           bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .svg")


# ── load data ────────────────────────────────────────────────────────────────
b0 = load("baseline_seed0")
b1 = load("baseline_seed1")
e0 = load("grokfast_emb_seed0")
e1 = load("grokfast_emb_seed1")

# ── fig1: loss curves ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(tok_b(b0), b0["val_loss"], color=C_BASE_LIGHT, lw=1.8, ls="--")
ax.plot(tok_b(b1), b1["val_loss"], color=C_BASE,       lw=1.8, ls="--")
ax.plot(tok_b(e0), e0["val_loss"], color=C_EMB_LIGHT,  lw=1.8)
ax.plot(tok_b(e1), e1["val_loss"], color=C_EMB,        lw=1.8)
for j, c in [(b0, C_BASE_LIGHT), (b1, C_BASE), (e0, C_EMB_LIGHT), (e1, C_EMB)]:
    ax.annotate(f"{j['val_loss'][-1]:.4f}",
                xy=(tok_b(j)[-1], j["val_loss"][-1]),
                xytext=(8, 0), textcoords="offset points",
                fontsize=10, color=c, va="center")
handles = [
    mpatches.Patch(facecolor=C_BASE, label="baseline (seed 0 & 1)"),
    mpatches.Patch(facecolor=C_EMB,  label="grokfast_emb (seed 0 & 1)"),
]
ax.legend(handles=handles, frameon=False, fontsize=12)
ax.set_xlabel("Tokens seen (B)", fontsize=14)
ax.set_ylabel("Validation loss", fontsize=14)
ax.set_title("Embedding-only Grokfast vs AdamW baseline\n200M params / 2B tokens / 2 seeds",
             fontsize=14, pad=12)
ax.set_xlim(left=0)
ax.set_ylim(bottom=2.8)
fig.tight_layout()
save(fig, "fig1_loss_curves")

# ── fig2: bucket perplexity (money shot) ────────────────────────────────────
keys      = ["bucket_perp_top1k", "bucket_perp_1k10k", "bucket_perp_rare"]
buckets   = ["Top-1K\n(common)", "1K-10K\n(mid)", "Rare 10K+\n(tail)"]
base_vals = [np.mean([b0[k][-1], b1[k][-1]]) for k in keys]
emb_vals  = [np.mean([e0[k][-1], e1[k][-1]]) for k in keys]
pct_drop  = [(e - b) / b * 100 for b, e in zip(base_vals, emb_vals)]

x = np.arange(len(buckets))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5.5))
bars_b = ax.bar(x - w/2, base_vals, width=w, color=C_BASE, label="baseline (mean)",     zorder=3)
bars_e = ax.bar(x + w/2, emb_vals,  width=w, color=C_EMB,  label="grokfast_emb (mean)", zorder=3)
for bar in bars_b:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10, color="#555")
for bar in bars_e:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10, color=C_EMB)
for i, (bv, pct) in enumerate(zip(base_vals, pct_drop)):
    ax.annotate(f"{pct:+.0f}%", xy=(i, bv * 1.10),
                ha="center", va="bottom", fontsize=14, fontweight="bold", color="#DC2626")
ax.set_xticks(x)
ax.set_xticklabels(buckets, fontsize=13)
ax.set_ylabel("Perplexity (lower = better)", fontsize=14)
ax.set_title("Embedding-only Grokfast: gain concentrates on rare tokens", fontsize=14, pad=12)
ax.legend(frameon=False, fontsize=12)
ax.set_ylim(bottom=0, top=max(base_vals) * 1.25)
fig.tight_layout()
save(fig, "fig2_bucket_perplexity")

# ── fig3: seed consistency ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
xs     = [0, 1, 3, 4]
vals   = [b0["val_loss"][-1], b1["val_loss"][-1], e0["val_loss"][-1], e1["val_loss"][-1]]
colors = [C_BASE, C_BASE, C_EMB, C_EMB]
labels = ["baseline\nseed 0", "baseline\nseed 1", "emb\nseed 0", "emb\nseed 1"]
for x_, v, c in zip(xs, vals, colors):
    ax.scatter(x_, v, color=c, s=200, zorder=5)
base_mean = np.mean(vals[:2]);  emb_mean = np.mean(vals[2:])
base_std  = np.std(vals[:2]);   emb_std  = np.std(vals[2:])
gap       = base_mean - emb_mean
ratio     = gap / max(base_std, emb_std)
ax.hlines(base_mean, -0.35, 1.35, colors=C_BASE, lw=1.5, ls="--", zorder=3)
ax.hlines(emb_mean,   2.65, 4.35, colors=C_EMB,  lw=1.5, ls="--", zorder=3)
ax.annotate("", xy=(2.0, emb_mean), xytext=(2.0, base_mean),
            arrowprops=dict(arrowstyle="<->", color="#DC2626", lw=2.2))
ax.text(2.25, (base_mean + emb_mean) / 2,
        f"gap = {gap:.4f}\n({ratio:.0f}x seed std)", color="#DC2626", fontsize=11, va="center")
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Final validation loss", fontsize=14)
ax.set_title(f"Effect dwarfs seed noise  (gap ~{ratio:.0f}x std)", fontsize=14, pad=12)
ax.set_ylim(3.14, 3.25)
handles = [mpatches.Patch(facecolor=C_BASE, label="baseline"),
           mpatches.Patch(facecolor=C_EMB,  label="grokfast_emb")]
ax.legend(handles=handles, frameon=False, fontsize=12, loc="upper right")
fig.tight_layout()
save(fig, "fig3_seed_consistency")

print(f"\nAll figures written to {OUT}/")
