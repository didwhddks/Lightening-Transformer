#!/usr/bin/env python3
"""
Publication-quality power breakdown donut chart.

Targets IEEE/ACM single-column figures (~3.5 in wide).
Outputs both PNG (300 DPI) and PDF (vector).

Expected CSV columns:
  dota         - component name
  power (mW)   - absolute power (preferred; used to recompute percentages)
  percentage (%) - optional fallback

A row with dota == "total" is treated as the total annotation value.

Usage:
  python plot.py <input.csv> <output_prefix>
  e.g.  python plot.py dota_power.csv figs/power_breakdown
        → figs/power_breakdown.png  +  figs/power_breakdown.pdf
"""

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

# ── Publication-style rcParams ───────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "pdf.fonttype":       42,   # embed fonts (required by IEEE)
    "ps.fonttype":        42,
    "axes.linewidth":     0.6,
    "lines.linewidth":    0.8,
})

# ── Tunable knobs ─────────────────────────────────────────────────────────────
MERGE_ADDER_MEM   = True      # merge adder+mem → Others
DROP_EPS          = 1e-9      # drop components with ~0% power

FIGSIZE           = (3.5, 3.0)   # single-column IEEE (inches)
DONUT_WIDTH       = 0.38         # thickness of donut ring (0=full donut, 0.5=thick)
STARTANGLE        = 90
COUNTERCLOCK      = False

# Colorblind-friendly palette (IBM Design / Paul Tol "bright")
PALETTE = [
    "#4477AA",   # blue      – DAC
    "#EE6677",   # red       – MZM
    "#228833",   # green     – Photodetector
    "#CCBB44",   # yellow    – ADC
    "#66CCEE",   # cyan      – TIA
    "#AA3377",   # purple    – MD
    "#BBBBBB",   # grey      – laser / Others
    "#332288",   # indigo
    "#FF8C00",   # orange
]

LABEL_RADIUS      = 1.22     # label anchor radius (1.0 = pie edge)
MIN_DY            = 0.13     # minimum vertical gap between same-side labels
LINEWIDTH_WEDGE   = 0.5      # wedge border
LINEWIDTH_LEADER  = 0.6      # leader line
LEADER_COLOR      = "#444444"

FONT_LABEL        = 7.0
FONT_TOTAL_SIZE   = 10.0
SMALL_PCT         = 2.0      # below this, shrink label font
BOLD_PCT          = 20.0     # above this, bold label
# ─────────────────────────────────────────────────────────────────────────────


def pretty_name(s: str) -> str:
    ACRONYMS = {"DAC", "ADC", "TIA", "MZM", "MRR", "MD"}
    s = str(s).strip()
    if s.upper() in ACRONYMS:
        return s.upper()
    if s.lower() == "others":
        return "Others"
    return s[:1].upper() + s[1:]


def wrap_label(name: str, max_len: int = 11) -> str:
    if "\n" in name or len(name) <= max_len:
        return name
    mid = len(name) // 2
    return name[:mid] + "\n" + name[mid:]


def repel_labels(items: list, min_dy: float, iterations: int = 30) -> list:
    """Bidirectional iterative label repulsion to avoid overlaps."""
    if not items:
        return items
    items = sorted(items, key=lambda d: d["y"], reverse=True)
    Y_MAX, Y_MIN = 1.30, -1.30

    for _ in range(iterations):
        # top → bottom pass: push downward
        for i in range(1, len(items)):
            gap = items[i - 1]["y"] - items[i]["y"]
            if gap < min_dy:
                items[i]["y"] = items[i - 1]["y"] - min_dy

        # bottom → top pass: push upward
        for i in range(len(items) - 2, -1, -1):
            gap = items[i]["y"] - items[i + 1]["y"]
            if gap < min_dy:
                items[i]["y"] = items[i + 1]["y"] + min_dy

    # clamp to visible range
    for d in items:
        d["y"] = max(min(d["y"], Y_MAX), Y_MIN)
    return items


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df["_key"] = df["dota"].astype(str).str.lower().str.strip()

    total_row = df[df["_key"] == "total"]
    total_w   = None
    if len(total_row) == 1 and "power (mW)" in df.columns:
        total_w = float(total_row["power (mW)"].iloc[0]) / 1000.0   # mW → W

    data = df[df["_key"] != "total"].copy()

    if MERGE_ADDER_MEM:
        data.loc[data["_key"].isin(["adder", "mem"]), "_key"] = "others"

    agg_cols = [c for c in ["power (mW)", "percentage (%)"] if c in data.columns]
    if not agg_cols:
        raise ValueError("CSV must contain 'power (mW)' or 'percentage (%)'.")

    grouped = data.groupby("_key", as_index=False)[agg_cols].sum()

    if "power (mW)" in grouped.columns:
        tot = grouped["power (mW)"].sum()
        grouped["pct"] = grouped["power (mW)"] / tot * 100.0
    else:
        tot = grouped["percentage (%)"].sum()
        grouped["pct"] = grouped["percentage (%)"] / tot * 100.0

    grouped = grouped[grouped["pct"].abs() > DROP_EPS].copy()
    grouped = grouped.sort_values("pct", ascending=False).reset_index(drop=True)
    grouped["display_name"] = grouped["_key"].apply(pretty_name).apply(wrap_label)

    return grouped, total_w


def draw(csv_path: str, out_prefix: str):
    grouped, total_w = load_data(csv_path)

    values = grouped["pct"].tolist()
    names  = grouped["display_name"].tolist()
    pcts   = grouped["pct"].tolist()
    n      = len(values)

    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # ── Draw donut ────────────────────────────────────────────────────────────
    wedges, _ = ax.pie(
        values,
        labels=None,
        startangle=STARTANGLE,
        counterclock=COUNTERCLOCK,
        colors=colors,
        wedgeprops=dict(
            edgecolor="white",
            linewidth=LINEWIDTH_WEDGE,
            width=DONUT_WIDTH,    # donut hole
        ),
        radius=1.0,
    )

    # ── Total power in center ──────────────────────────────────────────────
    if total_w is not None:
        label_str = f"{total_w*1000:.0f} mW" if total_w < 1.0 else f"{total_w:.2f} W"
        ax.text(
            0, 0, label_str,
            ha="center", va="center",
            fontsize=FONT_TOTAL_SIZE,
            fontweight="bold",
            color="#222222",
        )

    # ── Outside labels with leader lines ──────────────────────────────────
    items = []
    for w, name, pct in zip(wedges, names, pcts):
        ang   = 0.5 * (w.theta1 + w.theta2)
        rad   = np.deg2rad(ang)
        x_lbl = np.cos(rad) * LABEL_RADIUS
        y_lbl = np.sin(rad) * LABEL_RADIUS
        side  = "right" if x_lbl >= 0 else "left"
        items.append(dict(w=w, name=name, pct=pct,
                          ang=ang, x=x_lbl, y=y_lbl, side=side))

    left_items  = repel_labels([d for d in items if d["side"] == "left"],  MIN_DY)
    right_items = repel_labels([d for d in items if d["side"] == "right"], MIN_DY)

    for d in left_items + right_items:
        ang = np.deg2rad(d["ang"])
        # leader line: from outer wedge edge → label
        x0 = np.cos(ang) * 1.02
        y0 = np.sin(ang) * 1.02
        x1, y1 = d["x"], d["y"]
        ha = "left" if d["side"] == "right" else "right"

        fs     = FONT_LABEL * 0.85 if d["pct"] < SMALL_PCT else FONT_LABEL
        weight = "bold" if d["pct"] >= BOLD_PCT else "normal"
        label  = f"{d['name']}\n({d['pct']:.1f}%)"

        ax.annotate(
            label,
            xy=(x0, y0),
            xytext=(x1, y1),
            ha=ha,
            va="center",
            fontsize=fs,
            fontweight=weight,
            color="#222222",
            arrowprops=dict(
                arrowstyle="-",
                color=LEADER_COLOR,
                lw=LINEWIDTH_LEADER,
            ),
        )

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0.3)

    path = out_prefix + ".png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot.py <input.csv> <output_prefix>")
        print("  e.g. python plot.py dota_power.csv figs/power_breakdown")
        sys.exit(1)
    draw(sys.argv[1], sys.argv[2])
