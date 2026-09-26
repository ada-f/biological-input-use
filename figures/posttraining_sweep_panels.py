#!/usr/bin/env python
"""Loaders and panels for the post-training sweep (RQ2): BioReason accuracy for the 42 checkpoints
of the post-training sweep, including the released SFT and RL checkpoints.

sft_vs_rl.py draws draw_scatter_panel() as Figure 5b (accuracy on the 165 genome-dependent queries
with Z_Evo2 intact against shuffled, one marker per checkpoint). posttraining_sweep.py draws the
three sweep_panel_specs() panels as the RQ2 appendix figure: accuracy with Z_Evo2 intact on the 165
genome-dependent queries and on all 1,449 queries, both per query, as SFT LoRA rank, RL LoRA rank
and RL epochs vary. load() reads genome_dependent_accuracy.json (or --dep) and
per_query_accuracy.json in dna/bioreason/posttraining_sweep under INPUT_USE_RESULTS_DIR, and checks
the accuracy of the two released checkpoints against their own records in
dna/bioreason/perturbations.

Run on its own, the module draws all four panels in one figure,
fig_posttraining_published_panels, which is not in the paper, with its numbers CSV in
outputs/figures or --out_dir.
"""
from __future__ import annotations

import argparse, csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_style import (init_print_style, ax_in, align_xlabels, panel_letter, emit,  # noqa: E402
                          DNA, GREY, TICK_FS, LEGEND_FS, _mix)
import seaborn as sns  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq1_data as M  # noqa: E402

OUT_DEFAULT = M.OUT_DEFAULT
STEM = "fig_posttraining_published_panels"
S = 7.0 / 5.5
RUN = M.RD.DNA_POSTTRAINING_SWEEP
SIZE_MARK = {"qwen3_1p7b": ("o", "1.7B", "-"), "qwen3_4b": ("s", "4B", "--")}
SFT_RANKS, RL_RANKS, RL_EPOCHS = [16, 64, 256], [4, 16, 64], [0, 3, 4]
PUB_INIT, PUB_RL_RANK = 256, 16          # the published RL's lineage
# ---- the two released checkpoints, accuracy per query on all 1,449 queries -----------------------
# PLOTTED: this sweep's own generations for the two released checkpoints, from the sweep's
# records_<cell>.jsonl, so all 42 points come from one generation pipeline.
# RELEASED: the released checkpoints' own records in the perturbation run, the accuracies quoted in
# Sec. 3.2 (0.823 SFT, 0.847 RL). The weights are identical, but greedy generations diverge
# between the two runs, so the accuracies differ. Both are asserted in check_published_per_query().
PUB_PERQUERY_PLOTTED = {"L_sft_1p7b_r64": (1186, 0.8185), "L_rl_1p7b_sft256_r16_e4": (1246, 0.8599)}
PUB_PERQUERY_RELEASED = {"L_sft_1p7b_r64": (1193, 0.8233), "L_rl_1p7b_sft256_r16_e4": (1227, 0.8468)}
RELEASED_PQ = {"L_sft_1p7b_r64": M.RD.dna_per_query(M.KEGG, "sft"),
               "L_rl_1p7b_sft256_r16_e4": M.RD.dna_per_query(M.KEGG, "rl")}
MEW = 0.5                        # outline width (x S)
PUB_MEW = MEW                    # released checkpoints are drawn like every other point


def sft_tint(r):
    t = math.log2(r / 16) / math.log2(256 / 16)
    return _mix(DNA.model, "#FFFFFF", 0.78 * (1 - t))


def rl_size(e):
    return 3.6 + 0.9 * e


def load(partial, dep_path=None):
    dep = M.load(dep_path or M.RD.DNA_SWEEP_GENOME_DEPENDENT)
    # The all-1,449-query series is a mean over queries. genome_dependent_accuracy.json holds that
    # series only as a mean over genomes, so it comes from per_query_accuracy.json, which derives it
    # from the same records; the assertions below check its genome average against
    # genome_dependent_accuracy.json per cell.
    pq = M.load(M.RD.DNA_SWEEP_PER_QUERY)["cells"]
    assert set(pq) == set(dep["cells"]), (len(pq), len(dep["cells"]))
    cells = []
    for cell, e in dep["cells"].items():
        m = e["cell_meta"]
        sft_rank = m["lora_rank"] if m["family"] == "lora_sft_rank" else m["sft_init_rank"]
        q = pq[cell]["wt"]
        assert abs(q["accuracy_per_genome"] - round(e["wt"]["acc_all_rows_genome_mean"], 4)) < 5e-5, cell
        assert q["n_queries"] == 1449 and q["n_genomes"] == 708, cell
        assert abs(q["accuracy_per_query_full"] - q["k_all"] / 1449) < 1e-12, cell
        assert abs(q["acc_dep165_rows"] - round(e["wt"]["acc_dep165_rows"], 4)) < 5e-5, cell
        c = {"cell": cell, "tag": m["model_tag"], "family": m["family"], "sft_rank": sft_rank,
             "rl_rank": m["lora_rank"] if m["family"] == "lora_rl_rank" else None, "rl": int(m["rl_epochs"]),
             "published": m.get("published_as"), "dep_wt": e["wt"]["acc_dep165_rows"],
             "dep_scr": e["scramble"]["acc_dep165_rows"], "all_wt": q["accuracy_per_query_full"],
             "all_ci": q["ci_cluster"], "all_k": q["k_all"], "all_gmean": e["wt"]["acc_all_rows_genome_mean"],
             "k_wt": e["wt"]["k_dep"], "k_scr": e["scramble"]["k_dep"],
             "vs_released": e.get("vs_released_records")}
        cells.append(c)
        ck = f"SFT r{sft_rank}" + (f" / GRPO r{c['rl_rank']} {c['rl']} ep" if c["rl_rank"] else "")
        model = f"LoRA sweep {cell}" + (f" (= published {c['published']})" if c["published"] else "")
        for arm, key, k in (("wt", "dep_wt", "k_wt"), ("scramble", "dep_scr", "k_scr")):
            M.record("posttraining_published", "a", model=model, checkpoint=ck, group="165 genome-dependent rows",
                     series=f"genome {'intact' if arm == 'wt' else 'scrambled'}", x=arm, value=c[key], n=165, k=c[k],
                     source=M.RD.DNA_SWEEP_GENOME_DEPENDENT,
                     key=f"cells.{cell}.{arm}.acc_dep165_rows [PER QUERY: mean over 165 queries]")
        M.record("posttraining_published", "a", model=model, checkpoint=ck, group="165 genome-dependent rows",
                 series="intact - shuffled (plotted y)", x=c["dep_wt"], value=c["dep_wt"] - c["dep_scr"], n=165,
                 k=c["k_wt"] - c["k_scr"], source=M.RD.DNA_SWEEP_GENOME_DEPENDENT,
                 key=f"cells.{cell}.wt.acc_dep165_rows - cells.{cell}.scramble.acc_dep165_rows "
                     f"[PER QUERY: mean over 165 queries]")
        # k is the number of correct queries, as in the 165-query rows above, so k/n with n = 1,449
        # reproduces the value.
        M.record("posttraining_published", "b-d source", model=model, checkpoint=ck, group=c["family"],
                 series="all 1,449 rows, genome intact", x="", value=c["all_wt"], n=1449, k=c["all_k"],
                 ci_low=c["all_ci"][0], ci_high=c["all_ci"][1], source=M.RD.DNA_SWEEP_PER_QUERY,
                 key=f"cells.{cell}.wt.accuracy_per_query_full [PER QUERY: mean over 1,449 queries; "
                     f"the genome-averaged value is "
                     f"genome_dependent_accuracy.json cells.{cell}.wt.acc_all_rows_genome_mean = "
                     f"{c['all_gmean']:.4f}]")
    for cell, e in dep.get("excluded", {}).items():
        print(f"[excluded] {cell}: {e['reason']} ({e['degenerate_rate']:.2f})")
    n_expected = 42 - len(dep.get("excluded", {}))
    if not partial:
        assert len(cells) == n_expected, (len(cells), n_expected)
    pubs = [c for c in cells if c["published"]]
    for c in pubs:
        v = c["vs_released"]
        print(f"[published] {c['cell']}: 165 rows intact {c['dep_wt']:.3f} scrambled {c['dep_scr']:.3f}, all rows "
              f"{c['all_wt']:.3f}; vs released-checkpoint records: {v['same_mapped_answer']}/{v['n_rows']} same answers, "
              f"released 165-row acc {v['released_acc_dep165_rows']:.3f}")
    d = [c["dep_wt"] - c["dep_scr"] for c in cells]
    print(f"[a] {len(cells)} checkpoints; intact - shuffled on 165 rows: mean {np.mean(d):+.3f}, min {min(d):+.3f}, "
          f"max {max(d):+.3f} (>0: {sum(x > 0 for x in d)}, <0: {sum(x < 0 for x in d)}, =0: {sum(x == 0 for x in d)}); "
          f"intact range {min(c['dep_wt'] for c in cells):.3f}-{max(c['dep_wt'] for c in cells):.3f}")
    for tag, lab in (("qwen3_1p7b", "1.7B"), ("qwen3_4b", "4B")):
        t = [c for c in cells if c["tag"] == tag]
        print(f"[b-d] {lab}, {len(t)} checkpoints: accuracy on all 1,449 queries, PER QUERY "
              f"{min(c['all_wt'] for c in t):.4f}-{max(c['all_wt'] for c in t):.4f} "
              f"(genome-averaged: {min(c['all_gmean'] for c in t):.4f}-"
              f"{max(c['all_gmean'] for c in t):.4f})")
    if not partial:
        check_published_per_query(cells)
    return cells


def check_published_per_query(cells):
    """Checks the accuracy per query on all 1,449 queries of the two released checkpoints, as plotted
    (this sweep's generations) and as quoted in Sec. 3.2 (0.823 SFT, 0.847 RL, from the released
    checkpoints' own records in the perturbation run), and prints the difference. The weights are
    identical, but greedy generations diverge between the two runs, so the two accuracies differ."""
    for cell, (k_exp, acc_exp) in PUB_PERQUERY_PLOTTED.items():
        c = find(cells, cell=cell)
        assert c is not None and c["published"], cell
        assert c["all_k"] == k_exp and abs(c["all_wt"] - acc_exp) < 5e-5, (cell, c["all_k"], c["all_wt"])
        rel = M.load(RELEASED_PQ[cell])["per_arm"]["wt"]
        k_rel, acc_rel = PUB_PERQUERY_RELEASED[cell]
        assert rel["n_queries"] == 1449 and abs(rel["accuracy_per_query"] - acc_rel) < 5e-5, (cell, rel)
        assert abs(k_rel / 1449 - acc_rel) < 5e-5, (cell, k_rel)
        print(f"[published, per query] {cell} ({c['published']}): plotted {c['all_k']}/1449 = "
              f"{c['all_wt']:.4f} (this sweep's regeneration) vs released records {k_rel}/1449 = "
              f"{acc_rel:.4f} (quoted in Sec. 3.2); difference "
              f"{c['all_k'] - k_rel:+d} queries, {c['all_wt'] - acc_rel:+.4f}. Genome-averaged, the same "
              f"plotted cell read {c['all_gmean']:.4f}.")


def find(cells, **kw):
    out = [c for c in cells if all(c.get(k) == v for k, v in kw.items())]
    return out[0] if len(out) == 1 else None


# ------------------------------------------------------------------------------------------------
# panel drawing, one function per panel so another figure can call the same code
# ------------------------------------------------------------------------------------------------
# Every length and font below is multiplied by `s`. The standalone figure passes s = S (it draws on
# a 7/5.5 canvas); a figure drawn at its printed width passes s = 1 and gets the same panel with the
# same proportions at print size.
YT = np.arange(0, 1.01, 0.25)


def panel_title(ax, text, s=S):
    ax.set_title(text, fontsize=TICK_FS * s, fontweight="normal", pad=3.0 * s, linespacing=1.15)


def draw_scatter_panel(ax, cells, s=S, lim=(-0.02, 1.02), ticks=None):
    """One marker per checkpoint on the 165 genome-dependent queries: x is accuracy with Z_Evo2 from
    intact DNA, y with both DNA windows shuffled, and a dotted line marks equal accuracy.

    `lim` and `ticks` apply to both axes, so the panel stays square and the equality line stays at
    45 degrees. The defaults are 0-1 axes; sft_vs_rl.py passes 0.25-0.75 for Figure 5b."""
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7 * s, ls=":", zorder=1)   # identity line
    for c in sorted(cells, key=lambda c: -c["rl"]):          # big markers first, small on top
        if c["published"]:
            continue
        ax.plot(c["dep_wt"], c["dep_scr"], marker=SIZE_MARK[c["tag"]][0], ms=rl_size(c["rl"]) * s,
                mfc=sft_tint(c["sft_rank"]), mec="black", mew=0.5 * s, ls="none", zorder=4)
    for c in cells:
        if not c["published"]:
            continue
        y = c["dep_scr"]
        ax.plot(c["dep_wt"], y, marker=SIZE_MARK[c["tag"]][0], ms=rl_size(c["rl"]) * s, mfc=sft_tint(c["sft_rank"]),
                mec="black", mew=PUB_MEW * s, ls="none", zorder=7)
    sns.despine(ax=ax)
    tk = YT if ticks is None else ticks
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xticks(tk); ax.set_yticks(tk); ax.set_aspect("equal")
    ax.set_xlabel("Accuracy, Evo2 intact", fontsize=TICK_FS * s)
    ax.set_ylabel("Accuracy, Evo2 shuffled", fontsize=TICK_FS * s)


def scatter_legend_blocks(s=S):
    """The three titled legend blocks of the scatter panel: fill (SFT LoRA rank), marker size (RL
    epochs) and marker shape (model size)."""
    return [("SFT LoRA rank", [Line2D([], [], marker="o", ms=5.0 * s, mfc=sft_tint(rk), mec="black", mew=0.5 * s,
                                      ls="none", label=str(rk)) for rk in SFT_RANKS]),
            ("RL epochs", [Line2D([], [], marker="o", ms=rl_size(e) * s, mfc="white", mec="black", mew=0.5 * s,
                                  ls="none", label=str(e)) for e in RL_EPOCHS]),
            ("Model size", [Line2D([], [], marker=mk, ms=5.0 * s, mfc="white", mec="black", mew=0.5 * s, ls="none",
                                   label=sl) for mk, sl, _ in SIZE_MARK.values()])]


def series(ax, pts_by_tag, s=S, xform=lambda x: x):
    """The two accuracy series of a sweep-axis panel: 165 genome-dependent rows (DNA fill) over all
    1,449 rows (white on grey), one line per model size."""
    for tag, (mk, _, ls) in SIZE_MARK.items():
        pts = sorted(pts_by_tag.get(tag, []), key=lambda c: c["_x"])
        if not pts:
            continue
        xs = [xform(c["_x"]) for c in pts]
        ax.plot(xs, [c["all_wt"] for c in pts], color=GREY, lw=0.8 * s, ls=ls, zorder=1)
        ax.plot(xs, [c["dep_wt"] for c in pts], color=DNA.model, lw=0.8 * s, ls=ls, zorder=2)
        for c, x in zip(pts, xs):
            ew = (PUB_MEW if c["published"] else MEW) * s
            ax.plot([x], [c["all_wt"]], marker=mk, ms=4.2 * s, mfc="white", mec="black", mew=ew, ls="none", zorder=3)
            ax.plot([x], [c["dep_wt"]], marker=mk, ms=4.2 * s, mfc=DNA.model, mec="black", mew=ew, ls="none", zorder=5)


def tagged(cs, xkey):
    d = {}
    for c in cs:
        c = dict(c); c["_x"] = c[xkey]; d.setdefault(c["tag"], []).append(c)
    return d


def sweep_panel_specs(cells):
    """(title, x label, points by size tag, x ticks, x scale) for the three sweep-axis panels, in the
    order they are drawn. Letters are assigned by the caller, so a figure that drops another panel
    can relabel them without touching the panels themselves."""
    specs = []
    # SFT only vs SFT rank
    b_pts = [c for c in cells if c["family"] == "lora_sft_rank"]
    specs.append(("SFT only", "SFT LoRA rank", tagged(b_pts, "sft_rank"), [16, 64, 256], "log"))
    # GRPO 4 epochs on SFT r256, vs RL rank
    c_main = [c for c in cells if c["family"] == "lora_rl_rank" and c["sft_rank"] == PUB_INIT and c["rl"] == 4]
    specs.append(("RL 4 epochs\non SFT r256", "RL LoRA rank", tagged(c_main, "rl_rank"), [4, 16, 64], "log"))
    # GRPO r16 on SFT r256 vs RL epochs (0 = the SFT r256 checkpoint)
    d_main = [c for c in cells if c["sft_rank"] == PUB_INIT and (c["family"] == "lora_sft_rank" or c["rl_rank"] == PUB_RL_RANK)]
    specs.append(("RL r16\non SFT r256", "RL epochs", tagged(d_main, "rl"), [0, 3, 4], "linear"))
    return specs


def draw_sweep_panel(ax, spec, first, s=S):
    """One sweep-axis panel. `first` carries the shared y label and ticks."""
    ptitle, xlabel, pts, ticks, xscale = spec
    series(ax, pts, s=s)
    sns.despine(ax=ax)
    if xscale == "log":
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
        ax.set_xlim(ticks[0] / 1.55, ticks[-1] * 1.55)
    else:
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(0, 1.0); ax.set_yticks(YT)
    if not first:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.set_ylabel(r"Accuracy $Z_{\mathrm{Evo2}}$ intact", fontsize=TICK_FS * s)
    ax.set_xlabel(xlabel, fontsize=TICK_FS * s)
    panel_title(ax, ptitle, s=s)


def place_sweep_legend_row(fig, axes_bd, x_centre, s=S):
    """Legend of the sweep panels: one row below them, laid out left to right from measured widths
    and centred on `x_centre` inches unless that runs off the right edge. It hangs from the measured
    bottom of the lowest x label, so a label change cannot overlap it."""
    W, H = fig.get_size_inches()
    LF = LEGEND_FS * s
    leg = dict(frameon=False, fontsize=LF, handletextpad=0.3, labelspacing=0.2, columnspacing=0.5,
               borderaxespad=0.0, borderpad=0.2)
    lab = dict(va="center", ha="left", fontsize=LF, color="black")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    xl_bottom = min(a_.xaxis.label.get_window_extent(r).y0 for a_ in axes_bd) / fig.dpi   # inches
    y_row = xl_bottom - 0.09 * s              # hangs from the one-line x labels
    size_hs = [Line2D([], [], color="black", lw=0.8 * s, ls=ls, marker=mk, ms=4.2 * s, mfc="white", mec="black",
                      mew=0.5 * s, label=sl) for mk, sl, ls in SIZE_MARK.values()]
    q_hs = [Line2D([], [], color=DNA.model, lw=0.8 * s, marker="o", ms=4.2 * s, mfc=DNA.model, mec="black",
                   mew=0.5 * s, label="Genome-dependent"),
            Line2D([], [], color=GREY, lw=0.8 * s, marker="o", ms=4.2 * s, mfc="white", mec="black", mew=0.5 * s,
                   label="All")]
    pieces = [fig.text(0, y_row / H, "Model size", **lab),
              fig.legend(handles=size_hs, loc="center left", bbox_to_anchor=(0, y_row / H), ncol=2, handlelength=1.3, **leg),
              fig.text(0, y_row / H, "Queries", **lab),
              fig.legend(handles=q_hs, loc="center left", bbox_to_anchor=(0, y_row / H), ncol=2, handlelength=1.3, **leg)]
    gaps = [0.10 * s, 0.25 * s, 0.10 * s]     # text->handles, handles->next label, text->handles
    fig.canvas.draw()
    widths = [pc.get_window_extent(r).width / fig.dpi for pc in pieces]
    row_w = sum(widths) + sum(gaps)
    x = min(x_centre - row_w / 2, W - 0.06 * s - row_w)   # centred unless that runs off the right edge
    for i, pc in enumerate(pieces):
        if isinstance(pc, matplotlib.legend.Legend):
            pc.set_bbox_to_anchor((x / W, y_row / H), transform=fig.transFigure)
        else:
            pc.set_x(x / W)
        x += widths[i] + (gaps[i] if i < len(gaps) else 0)


def draw(cells, out_dir):
    W = 8.0 + 0.28 * S
    TOP, TIT, AX, BOT = 0.10 * S, 0.30 * S, 1.40 * S, 0.53 * S   # BOT: two-line x label + one legend row
    H = TOP + TIT + AX + BOT
    fig = plt.figure(figsize=(W, H))
    a_x = 0.68 * S                # left gutter: signed tick labels ("-0.10") and a two-line y label
    A_W = 1.40 * S                # square scatter panel
    _unused_A_W = 1.12 * S
    gap_a, p_w = 1.27 * S, 0.90 * S             # gap_a holds a's legend column plus clear space before b
    gap_bc, gap_cd = 0.22 * S, 0.22 * S
    p_x = [a_x + A_W + gap_a, a_x + A_W + gap_a + p_w + gap_bc, a_x + A_W + gap_a + 2 * p_w + gap_bc + gap_cd]
    row_top = TOP + TIT
    LF = LEGEND_FS * S

    # ---- a: accuracy with the intact embedding (x) versus with the shuffled one (y), 165 genome-dependent rows
    ax = ax_in(fig, a_x, row_top, A_W, AX)
    draw_scatter_panel(ax, cells, s=S)
    panel_letter(fig, 0.02 * S, TOP, "a", fontsize=8 * S)
    axes_x = [ax]

    # ---- b-d
    for i, (letter, spec) in enumerate(zip("bcd", sweep_panel_specs(cells))):
        ax = ax_in(fig, p_x[i], row_top, p_w, AX)
        draw_sweep_panel(ax, spec, first=(i == 0), s=S)
        panel_letter(fig, p_x[i] - (0.40 if i == 0 else 0.17) * S, TOP, letter, fontsize=8 * S)   # c, d closer to their axes: clears c's two-line title
        axes_x.append(ax)
    align_xlabels(fig, axes_x, pad_in=0.05 * S)

    # ---- legends: a's legend is a column to its right, three titled blocks stacked (fill, size,
    # shape); b-d's legend is one row below, centred under panel c.
    leg = dict(frameon=False, fontsize=LF, handletextpad=0.3, labelspacing=0.2, columnspacing=0.5, borderaxespad=0.0,
               borderpad=0.2)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    blocks = scatter_legend_blocks(s=S)
    lx = a_x + A_W + 0.09 * S                 # legend column starts just right of a's axes
    legs_a = [fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(lx / W, 1 - TOP / H), ncol=1, title=name,
                         title_fontsize=LF, alignment="left", handlelength=0.9, **leg) for name, hs in blocks]
    fig.canvas.draw()
    hs_a = [lg.get_window_extent(r).height / fig.dpi for lg in legs_a]
    gap_blk = 0.05 * S
    y_top = row_top + AX / 2 - (sum(hs_a) + gap_blk * (len(hs_a) - 1)) / 2   # stack centred on a's axes
    for lg, h in zip(legs_a, hs_a):
        lg.set_bbox_to_anchor((lx / W, 1 - y_top / H), transform=fig.transFigure)
        y_top += h + gap_blk
    # b-d: one row, centred under panel c (= the centre of the b-d group)
    place_sweep_legend_row(fig, axes_x[1:], (p_x[0] + p_x[2] + p_w) / 2, s=S)
    fig.canvas.draw()
    low = fig.get_tightbbox(r).y0          # inches: lowest ink (the third legend row) above the bottom edge
    print(f"[layout] lowest ink {low:.3f} in above the bottom edge (BOT = {BOT:.2f} in)")
    return emit(fig, STEM, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--dep", default=None, help="genome_dependent_accuracy.json to read (default: the sweep run dir)")
    a = ap.parse_args()
    cells = load(a.partial, a.dep)
    init_print_style()
    matplotlib.rcParams.update({k: matplotlib.rcParams[k] * S for k in (
        "font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
        "axes.linewidth", "xtick.major.width", "ytick.major.width", "xtick.major.size", "ytick.major.size",
        "xtick.major.pad", "ytick.major.pad", "axes.labelpad")})
    draw(cells, a.out_dir)
    keys = ["figure", "panel", "model", "checkpoint", "group", "series", "x", "value", "n", "k", "ci_low", "ci_high",
            "sd_over_folds", "source", "key"]
    rows = [{k: r.get(k, "") for k in keys} for r in M.ROWS if r["figure"] == "posttraining_published"]
    p = os.path.join(a.out_dir, f"{STEM}_numbers.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {p} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
