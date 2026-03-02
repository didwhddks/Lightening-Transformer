#!/usr/bin/env python3
"""
Pie chart (publication-style) with outside labels + leader lines and overlap avoidance.

Features:
- Optionally merge 'adder' + 'mem' into 'Others' (enabled by default).
- Drop ~0% items.
- Manual label placement with left/right "repel" to avoid overlaps.
- Total power text placed outside the pie (top-right), similar to the example figure.

Expected CSV columns:
- dota
- power (mW)      (preferred; used to recompute percentages)
- percentage (%)  (optional fallback)

Optionally a row with dota == "total" for the total value.

Usage:
  python plot.py /path/to/dota_power.csv /path/to/dota_power.png
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt


# ----- TUNABLE STYLE KNOBS -----
MERGE_ADDER_MEM_TO_OTHERS = True   # set False if you want to keep adder/mem separate
DROP_ZERO_EPS = 1e-12

FIGSIZE = (7, 5)
DPI = 300

STARTANGLE = 90
COUNTERCLOCK = False

LABEL_R = 1.28         # radius for label text
MIN_DY = 0.085         # min vertical spacing between labels on the same side
LINEWIDTH_SLICE = 0.6  # wedge border width
LINEWIDTH_ARROW = 0.8  # leader line width

FONT_LABEL = 13
FONT_LABEL_SMALL = 11
FONT_TOTAL = 22

BOLD_PCT_THRESHOLD = 20.0  # labels >= this % become bold
SMALL_PCT_THRESHOLD = 1.0  # labels < this % use smaller font
# --------------------------------


def pretty_name(s: str) -> str:
    acronyms = {"DAC", "ADC", "TIA", "MZM", "MRR"}
    s_clean = str(s).strip()
    if s_clean.upper() in acronyms:
        return s_clean.upper()
    if s_clean.lower() == "others":
        return "Others"
    return s_clean[:1].upper() + s_clean[1:]


def wrap_label(name: str, max_len: int = 10) -> str:
    """
    Wrap long single-word labels into two lines (similar to 'Photodet\\nector').
    """
    if "\n" in name:
        return name
    if len(name) <= max_len:
        return name
    mid = len(name) // 2
    return name[:mid] + "\n" + name[mid:]


def main(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)

    # Total text (if exists)
    total_row = df[df["dota"].astype(str).str.lower().str.strip() == "total"]
    total_w = None
    if len(total_row) == 1 and "power (mW)" in df.columns:
        total_w = float(total_row["power (mW)"].iloc[0]) / 1000.0

    # Exclude total row
    data = df[df["dota"].astype(str).str.lower().str.strip() != "total"].copy()
    data["dota_norm"] = data["dota"].astype(str).str.lower().str.strip()

    # Merge adder + mem into others (optional)
    if MERGE_ADDER_MEM_TO_OTHERS:
        data.loc[data["dota_norm"].isin(["adder", "mem"]), "dota_norm"] = "others"

    # Aggregate numeric columns
    agg_cols = []
    if "power (mW)" in data.columns:
        agg_cols.append("power (mW)")
    if "percentage (%)" in data.columns:
        agg_cols.append("percentage (%)")
    if not agg_cols:
        raise ValueError("CSV must contain either 'power (mW)' or 'percentage (%)'.")

    grouped = data.groupby("dota_norm", as_index=False)[agg_cols].sum()

    # Compute percentages
    if "power (mW)" in grouped.columns:
        total_power_mw = grouped["power (mW)"].sum()
        if abs(total_power_mw) < DROP_ZERO_EPS:
            raise ValueError("Total power is 0; nothing to plot.")
        grouped["pct"] = grouped["power (mW)"] / total_power_mw * 100.0
    else:
        s = grouped["percentage (%)"].sum()
        if abs(s) < DROP_ZERO_EPS:
            raise ValueError("Sum of percentages is 0; nothing to plot.")
        grouped["pct"] = grouped["percentage (%)"] / s * 100.0

    # Drop ~0%
    grouped = grouped[grouped["pct"].abs() > DROP_ZERO_EPS].copy()
    if grouped.empty:
        raise ValueError("All categories have 0% contribution after filtering; nothing to plot.")

    # Sort by pct descending to stabilize label style and make large slices first
    grouped = grouped.sort_values("pct", ascending=False).reset_index(drop=True)

    # Prepare display labels: "NAME\n(12.3%)"
    disp_names = grouped["dota_norm"].apply(pretty_name).apply(wrap_label)
    grouped["label"] = disp_names + "\n(" + grouped["pct"].map(lambda x: f"{x:.1f}%") + ")"

    values = grouped["pct"].tolist()
    labels = grouped["label"].tolist()
    pcts = grouped["pct"].tolist()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Draw pie WITHOUT labels; we'll place them manually
    wedges, _ = ax.pie(
        values,
        labels=None,
        startangle=STARTANGLE,
        counterclock=COUNTERCLOCK,
        wedgeprops=dict(edgecolor="white", linewidth=LINEWIDTH_SLICE),
    )

    # Manual label placement with left/right repel
    import numpy as np

    items = []
    for w, lab, pct in zip(wedges, labels, pcts):
        ang = 0.5 * (w.theta1 + w.theta2)
        a = np.deg2rad(ang)
        x = np.cos(a) * LABEL_R
        y = np.sin(a) * LABEL_R
        side = "right" if x >= 0 else "left"
        items.append({"w": w, "lab": lab, "pct": pct, "x": x, "y": y, "side": side, "ang": ang})

    def repel(side_items):
        # top->bottom by y, enforce spacing
        side_items.sort(key=lambda d: d["y"], reverse=True)
        for i in range(1, len(side_items)):
            y_prev = side_items[i - 1]["y"]
            y_cur = side_items[i]["y"]
            if y_prev - y_cur < MIN_DY:
                side_items[i]["y"] = y_prev - MIN_DY

        # clamp to keep labels in frame
        y_max, y_min = 1.20, -1.20
        for d in side_items:
            d["y"] = max(min(d["y"], y_max), y_min)
        return side_items

    left = repel([d for d in items if d["side"] == "left"])
    right = repel([d for d in items if d["side"] == "right"])

    for d in left + right:
        w = d["w"]
        lab = d["lab"]
        pct = d["pct"]

        a = np.deg2rad(d["ang"])
        # arrow starts from wedge outer edge
        x0 = np.cos(a) * 1.0
        y0 = np.sin(a) * 1.0

        ha = "left" if d["side"] == "right" else "right"
        fs = FONT_LABEL_SMALL if pct < SMALL_PCT_THRESHOLD else FONT_LABEL
        weight = "bold" if pct >= BOLD_PCT_THRESHOLD else "normal"

        ax.annotate(
            lab,
            xy=(x0, y0),
            xytext=(d["x"], d["y"]),
            ha=ha,
            va="center",
            fontsize=fs,
            fontweight=weight,
            arrowprops=dict(arrowstyle="-", lw=LINEWIDTH_ARROW),
        )

    # Total text outside, top-right (axes fraction)
    if total_w is not None:
        ax.text(
            0.98,
            0.92,
            f"{total_w:.2f}W",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=FONT_TOTAL,
            fontweight="bold",
        )

    ax.set(aspect="equal")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot.py <input.csv> <output.png>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])