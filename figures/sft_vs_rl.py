#!/usr/bin/env python
"""Draws Figure 5 ("Biological representations contribute similarly little to SFT and RL checkpoint
performance"): intact against shuffled foundation model representations for SFT and RL checkpoints.

    python figures/sft_vs_rl.py [--out_dir <dir>]

Panel a is BioReason disease prediction accuracy with Z_Evo2 intact against shuffled under four text
conditions, SFT and RL checkpoints, on 1,449 queries. Panel b is accuracy on the 165 genome-dependent
queries with Z_Evo2 intact against shuffled for the 42 checkpoints of the post-training sweep, on
0.25-0.75 axes. Panel c is BioReason-Pro weighted F_max with ESM3 intact against shuffled under four
text conditions, SFT and RL checkpoints. Panels a and c are drawn by sft_vs_rl_panels.py and panel b
by posttraining_sweep_panels.py, from their data loaders. Each legend is a column of titled blocks to
the right of its panel. Writes fig_sft_vs_rl (pdf, png, svg) and fig_sft_vs_rl_numbers.csv to
outputs/figures, or to --out_dir.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, ax_in, panel_letter, emit, check_overflow,  # noqa: E402
                          DNA, PROTEIN, LEGEND_FS)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M                  # noqa: E402
import sft_vs_rl_panels as A           # noqa: E402  panels a and c
import posttraining_sweep_panels as P  # noqa: E402  panel b

OUT_DEFAULT = M.OUT_DEFAULT
STEM = "fig_sft_vs_rl"
S = 1.0                                # drawn at print size: 7 pt ticks/labels, 6 pt legends

# ---- layout, inches ----------------------------------------------------------------------------
# Each panel's legend is a column of titled blocks to the right of that panel. The column widths are
# measured before the canvas is sized, so a label change moves the panels rather than colliding with
# them.
AX = 1.52                              # square panels
L_AX = 0.48                            # left gutter: y tick labels + y label
LEG_PAD = 0.09                         # axes right edge -> its legend column
SEP = 0.44                             # legend column -> the next panel's y tick labels
R_MARGIN = 0.10
TOP, TIT, BOT = 0.08, 0.28, 0.38       # BOT holds the x tick labels and the x label
GAP_BLK = 0.05                         # between stacked blocks of one column
LET_DX = 0.30                          # panel letter, left of its axes
H = TOP + TIT + AX + BOT
BLK_KW = dict(frameon=False, fontsize=LEGEND_FS, handletextpad=0.3, labelspacing=0.2,
              borderaxespad=0.0, borderpad=0.2, handlelength=0.9)   # panel b's blocks

# Legend entries of the panel modules with a line break inserted, to narrow the legend columns.
DNA_RUNG_WRAPPED = ["all text", "pathway fields,\nsymbols masked", "query naming\nthe gene",
                    "query,\nsymbol masked"]
PROT_RUNG_WRAPPED = ["GO-GPT &\nInterPro", "GO-GPT", "InterPro", "None"]
TITLE_A = "BioReason\ntext sources"
TITLE_C = "BioReason-Pro\ntext sources"
TITLE_B = {"SFT LoRA rank": "SFT LoRA\nrank", "RL epochs": "RL\nepochs", "Model size": "Model\nsize"}
# Panel b's axes. The 42 checkpoints occupy 0.309-0.691 on both axes, so both axes run 0.25-0.75
# with ticks every 0.1, and the panel stays square with equal accuracy on the 45 degree line. Panels
# a and c keep 0-1: their text conditions without the gene symbol or without text reach 0.04 and
# 0.20.
B_LIM, B_TICKS = (0.25, 0.75), [0.3, 0.4, 0.5, 0.6, 0.7]


def check_b_fits(ax, cells):
    """Raises if any checkpoint marker, including its edge, falls outside panel b's axes."""
    lo, hi = B_LIM
    pts = [(c["dep_wt"], c["dep_scr"]) for c in cells]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    per_unit = (hi - lo) / (ax.get_window_extent().width / ax.figure.dpi)   # axis units per inch
    rad = max(P.rl_size(c["rl"]) for c in cells) / 72 / 2 * per_unit        # largest marker radius
    edges = [min(xs) - rad, min(ys) - rad, max(xs) + rad, max(ys) + rad]
    assert lo <= min(edges) and max(edges) <= hi, (B_LIM, edges)
    print(f"[b] {len(pts)} checkpoints on {B_LIM}: x {min(xs):.4f}-{max(xs):.4f}, "
          f"y {min(ys):.4f}-{max(ys):.4f}; with the largest marker ({rad * 2:.4f} axis units wide) "
          f"the outermost edge is {min(edges):.4f}/{max(edges):.4f}, nothing clipped")


def legend_columns():
    """One list of (title, handles, legend kwargs) per panel, in panel order, built by the handle
    functions of sft_vs_rl_panels.py and posttraining_sweep_panels.py. Called once per figure, since a
    legend handle belongs to the figure it was drawn in."""
    return {"a": [(TITLE_A, A.rung_handles(DNA, DNA_RUNG_WRAPPED, s=S), dict(A.LEG_KW))],
            "b": [(TITLE_B[name], hs, dict(BLK_KW)) for name, hs in P.scatter_legend_blocks(s=S)],
            "c": [(TITLE_C, A.rung_handles(PROTEIN, PROT_RUNG_WRAPPED, s=S), dict(A.LEG_KW)),
                  ("Checkpoint", A.ckpt_handles(s=S), dict(A.LEG_KW))]}


def add_block(fig, title, handles, kw, x, y_top, size):
    """One legend block, placed by inches from the left and top edges."""
    W_, H_ = size
    return fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x / W_, 1 - y_top / H_),
                      ncol=1, title=title, title_fontsize=LEGEND_FS, alignment="left", **kw)


def measure_columns():
    """Width and per-block height of each panel's legend column, in inches, measured on a throwaway
    canvas so the real canvas can be sized around them."""
    fig = plt.figure(figsize=(12, 4))
    out = {}
    for key, blocks in legend_columns().items():
        sizes = []
        for title, handles, kw in blocks:
            lg = add_block(fig, title, handles, kw, 0.0, 0.0, (12, 4))
            fig.canvas.draw()
            b = lg.get_window_extent(fig.canvas.get_renderer())
            sizes.append((b.width / fig.dpi, b.height / fig.dpi))
        out[key] = sizes
    plt.close(fig)
    return out


def layout(cols):
    """Panel x positions and legend x positions, in inches from the left edge.

    The pitch between panel origins is the same for all three groups, set by the widest panel and
    legend group plus the gutter the next panel's y labels need. The gap left after each legend
    column therefore varies with that column's width; `draw` prints the three gaps."""
    w = [max(x for x, _ in cols[k]) for k in "abc"]
    pitch = max(AX + LEG_PAD + wi for wi in w) + SEP
    px = [L_AX + i * pitch for i in range(3)]
    lx = [x + AX + LEG_PAD for x in px]
    return px, lx, w, pitch


def build(dna, prot, cells, cols, px, lx, W):
    """The whole figure at a given canvas width; positions are absolute inches, so W only sets the
    right margin. Returns the figure and panel b's axes."""
    fig = plt.figure(figsize=(W, H))
    row_top = TOP + TIT
    ax_a = ax_in(fig, px[0], row_top, AX, AX)
    A.draw_dna_panel(ax_a, dna, s=S)
    panel_letter(fig, px[0] - LET_DX, TOP, "a")

    ax_b = ax_in(fig, px[1], row_top, AX, AX)
    P.draw_scatter_panel(ax_b, cells, s=S, lim=B_LIM, ticks=B_TICKS)
    panel_letter(fig, px[1] - LET_DX, TOP, "b")

    ax_c = ax_in(fig, px[2], row_top, AX, AX)
    A.draw_protein_panel(ax_c, prot, s=S)
    panel_letter(fig, px[2] - LET_DX, TOP, "c")

    # ---- each panel's legend: a column of titled blocks beside that panel, the stack centred on
    # the panel's axes
    for i, (key, blocks) in enumerate(legend_columns().items()):
        hs = [h for _, h in cols[key]]
        y_top = row_top + AX / 2 - (sum(hs) + GAP_BLK * (len(hs) - 1)) / 2
        for (title, handles, kw), h in zip(blocks, hs):
            add_block(fig, title, handles, kw, lx[i], y_top, (W, H))
            y_top += h + GAP_BLK
    fig.canvas.draw()
    return fig, ax_b


def draw(dna, prot, cells, out_dir):
    cols = measure_columns()
    px, lx, w, pitch = layout(cols)
    for key, wi in zip("abc", w):
        h = sum(h for _, h in cols[key]) + GAP_BLK * (len(cols[key]) - 1)
        print(f"[legend] column {key}: {wi:.3f} in wide, {h:.3f} in tall stacked "
              f"({len(cols[key])} block(s)) against a {AX:.2f} in panel")
    # pass 1: measure the ink, so the right margin can be made equal to the left one
    fig, _ = build(dna, prot, cells, cols, px, lx, lx[2] + w[2] + 1.0)
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    plt.close(fig)
    W = bb.x1 + bb.x0
    fig, ax_b = build(dna, prot, cells, cols, px, lx, W)
    check_b_fits(ax_b, cells)
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    gaps = [pitch - (AX + LEG_PAD + wi) for wi in w]
    print(f"[layout] pitch {pitch:.3f} in between panel origins; gap after each legend column "
          f"{gaps[0]:.3f} / {gaps[1]:.3f} / {gaps[2]:.3f} in (the last is the right margin plus "
          f"{SEP:.2f} in of nothing)")
    print(f"[layout] {W:.2f} x {H:.2f} in; ink {bb.x0:.3f}-{bb.x1:.3f} in, so the left margin is "
          f"{bb.x0:.3f} in and the right {W - bb.x1:.3f} in; {bb.y0:.3f} in above the bottom edge")
    print(f"[layout] at width=\\linewidth (5.5 in): 7 pt type prints at {7 * 5.5 / W:.2f} pt, 6 pt "
          f"legend type at {6 * 5.5 / W:.2f} pt, a panel at {AX * 5.5 / W:.2f} in")
    fixed = W - 3 * AX
    print(f"[layout] W = {fixed:.3f} + 3 x panel edge, so 5.00 pt printed type needs W <= 7.70 in, "
          f"i.e. panels of {(7.70 - fixed) / 3:.3f} in (printed {(7.70 - fixed) / 3 * 5.5 / 7.70:.2f} in)")
    n_over = len(check_overflow(fig, STEM))
    print(f"[layout] {STEM}: {n_over} overflow items")
    return emit(fig, STEM, out_dir)


# panel of this figure <- (figure, panel) recorded by the module that draws the panel
PANEL_OF = {("sft_vs_rl", "a"): "a", ("posttraining_published", "a"): "b", ("protein", "a"): "c"}
KEYS = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
        "ci_low", "ci_high", "sd_over_folds", "source", "key"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--dep", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dna = A.data_dna()                 # records ("sft_vs_rl", "a")
    prot = A.data_protein()            # records ("protein", "a")
    cells = P.load(a.partial, a.dep)   # records ("posttraining_published", "a" and "b-d source")
    init_print_style()
    draw(dna, prot, cells, a.out_dir)

    rows = []
    for letter in "abc":
        src = [k for k, v in PANEL_OF.items() if v == letter][0]
        rows += [{**{k: r.get(k, "") for k in KEYS}, "figure": "sft_vs_rl", "panel": letter}
                 for r in M.ROWS if (r["figure"], r["panel"]) == src]
    p = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KEYS)
        w.writeheader()
        w.writerows(rows)
    n = {l: sum(r["panel"] == l for r in rows) for l in "abc"}
    print(f"wrote {p} ({len(rows)} rows: a {n['a']}, b {n['b']}, c {n['c']})")


if __name__ == "__main__":
    main()
