#!/usr/bin/env python
"""Draws the RQ2 appendix figure "Post-training changes BioReason performance": accuracy with Z_Evo2
intact along the three axes of the post-training sweep.

    python figures/posttraining_sweep.py [--out_dir <dir>] [--dep <genome_dependent_accuracy.json>]

Panel a is SFT only against SFT LoRA rank, panel b RL for 4 epochs on the SFT rank 256 checkpoint
against RL LoRA rank, and panel c RL rank 16 on that checkpoint against RL epochs (0 is the SFT
checkpoint). Each panel shows accuracy per query on the 165 genome-dependent queries and on all
1,449 queries, for Qwen3-1.7B and Qwen3-4B. The panels and their legend are drawn by
posttraining_sweep_panels.py at print size. Every drawn marker is read back from the figure and
checked against the numbers CSV. Writes fig_posttraining_published (pdf, png, svg) and
fig_posttraining_published_numbers.csv to outputs/figures, or to --out_dir.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, ax_in, align_xlabels, panel_letter, emit,  # noqa: E402
                          check_text_collisions, check_overflow, DNA)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M                    # noqa: E402
import posttraining_sweep_panels as P    # noqa: E402

OUT_DEFAULT = M.OUT_DEFAULT
STEM = "fig_posttraining_published"
S = 1.0                                  # drawn at print size: 7 pt ticks/labels, 6 pt legends

# ---- layout, inches: the panel geometry of posttraining_sweep_panels.py divided by its 7/5.5 canvas factor
TOP, TIT, AX, BOT = 0.10, 0.30, 1.40, 0.53
P_W, GAP = 0.90, 0.22
L_AX, R_MARGIN = 0.48, 0.12              # y tick labels + y label; last x tick label
W = L_AX + 3 * P_W + 2 * GAP + R_MARGIN
H = TOP + TIT + AX + BOT
PX = [L_AX, L_AX + P_W + GAP, L_AX + 2 * (P_W + GAP)]


def harvest(axes):
    """Every marker the three panels draw, read back from the Line2D artists as
    (panel letter, series, marker, x, y). The single-point marker calls in
    posttraining_sweep_panels.series are the only one-point lines in these axes. The series is the
    marker fill (white for all 1,449 queries, the DNA hue for the 165 genome-dependent queries) and
    the marker shape is the model size (o for 1.7B, s for 4B), which separates the two lines of a
    panel at the same x."""
    white, dna = to_rgba("white"), to_rgba(DNA.model)
    out = []
    for letter, ax in zip("abc", axes):
        for ln in ax.lines:
            xs, ys = ln.get_xdata(), ln.get_ydata()
            if len(xs) != 1:
                continue
            fc = to_rgba(ln.get_markerfacecolor())
            series = "all" if fc == white else "dep165" if fc == dna else None
            assert series is not None, (letter, fc)
            out.append((letter, series, ln.get_marker(), float(xs[0]), float(ys[0])))
    return out


def check_drawn_against_csv(drawn, cells, csv_path):
    """Asserts, marker by marker, that the figure draws the values in the numbers CSV.

    `drawn` is every marker read back from the artists. The expected set is rebuilt from
    P.sweep_panel_specs, so each marker is matched by (panel, series, marker shape, x) to one
    checkpoint, and its y is compared to that checkpoint's value: the value on all 1,449 queries
    against the CSV and against k/1,449, and the value on the 165 genome-dependent queries against
    genome_dependent_accuracy.json as loaded into `cells`."""
    rows = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            cell = r["model"].split()[2]                       # "LoRA sweep <cell>[ (= published ...)]"
            rows[cell] = (float(r["value"]), int(r["n"]), int(r["k"]))
    assert len(rows) == 42, len(rows)
    by_cell = {c["cell"]: c for c in cells}
    for cell, (v, n, k) in rows.items():
        assert n == 1449 and abs(k / n - v) < 1e-12, (cell, v, n, k)   # per query, not per genome
        assert v == by_cell[cell]["all_wt"], (cell, v, by_cell[cell]["all_wt"])
    expect = {}
    for letter, (_, _, pts_by_tag, _, _) in zip("abc", P.sweep_panel_specs(cells)):
        for tag, pts in pts_by_tag.items():
            mk = P.SIZE_MARK[tag][0]
            for c in pts:
                expect[(letter, "all", mk, float(c["_x"]))] = (c["cell"], rows[c["cell"]][0])
                expect[(letter, "dep165", mk, float(c["_x"]))] = (c["cell"], by_cell[c["cell"]]["dep_wt"])
    assert len(drawn) == len(expect) == 36, (len(drawn), len(expect))   # 6 checkpoints x 2 series x 3 panels
    for letter, series, mk, x, y in drawn:
        assert (letter, series, mk, x) in expect, (letter, series, mk, x)
        cell, want = expect[(letter, series, mk, x)]
        assert y == want, (letter, series, mk, x, cell, y, want)
    n_all = sum(s == "all" for _, s, _, _, _ in drawn)
    print(f"[qa] {len(drawn)} markers drawn ({n_all} on all 1,449 queries, {len(drawn) - n_all} on the "
          f"165 genome-dependent queries); each matched to one checkpoint by (panel, series, marker, x) "
          f"and equal to its value exactly; k/1449 reproduces all {len(rows)} CSV values")


def draw(cells, out_dir):
    fig = plt.figure(figsize=(W, H))
    row_top = TOP + TIT
    axes = []
    for i, (letter, spec) in enumerate(zip("abc", P.sweep_panel_specs(cells))):
        ax = ax_in(fig, PX[i], row_top, P_W, AX)
        P.draw_sweep_panel(ax, spec, first=(i == 0), s=S)
        panel_letter(fig, PX[i] - (0.40 if i == 0 else 0.17) * S, TOP, letter, fontsize=8 * S)
        axes.append(ax)
    align_xlabels(fig, axes, pad_in=0.05 * S)
    P.place_sweep_legend_row(fig, axes, (PX[0] + PX[2] + P_W) / 2, s=S)
    fig.canvas.draw()
    low = fig.get_tightbbox(fig.canvas.get_renderer()).y0
    print(f"[layout] lowest ink {low:.3f} in above the bottom edge (BOT = {BOT:.2f} in)")
    drawn = harvest(axes)
    print(f"[qa] {STEM}: {check_text_collisions(fig, STEM)} text collisions, "
          f"{len(check_overflow(fig, STEM))} overflow items")
    return emit(fig, STEM, out_dir), drawn


KEYS = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k",
        "ci_low", "ci_high", "sd_over_folds", "source", "key"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--dep", default=None, help="genome_dependent_accuracy.json to read (default: the sweep run dir)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cells = P.load(a.partial, a.dep)
    init_print_style()
    _, drawn = draw(cells, a.out_dir)

    # posttraining_sweep_panels.load records these rows as "b-d source": one row per checkpoint, the
    # accuracy on all 1,449 queries, from which sweep_panel_specs selects the points of each panel
    # (6 checkpoints per panel, two of them in two panels). Here they are labelled "a-c source".
    rows = [{**{k: r.get(k, "") for k in KEYS}, "figure": "posttraining_published",
             "panel": "a-c source"}
            for r in M.ROWS if (r["figure"], r["panel"]) == ("posttraining_published", "b-d source")]
    p = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KEYS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {p} ({len(rows)} rows, all 'a-c source')")
    check_drawn_against_csv(drawn, cells, p)
    P.check_published_per_query(cells)


if __name__ == "__main__":
    main()
