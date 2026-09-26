#!/usr/bin/env python
r"""Scores the BioReason checkpoints of the post-training sweep (RQ2) with intact, shuffled and donor
DNA.

For each checkpoint listed in --manifest, reads records_<checkpoint>.jsonl from --run and maps each
generation to the closed vocabulary of disease labels. Reports accuracy per condition and per split
of network_split.json with bootstrap intervals over genomes, how often the answer changes between
queries that share prompt text but differ in genome (pairs from kegg_pairs.py), and how often the
answer follows the donor genome in `swap_variant_donor`. Writes a JSON summary and a CSV with one
row per checkpoint next to it.

    python -m input_use.modalities.dna.posttrain_score --run results/dna/bioreason/posttraining_sweep \
        --manifest <manifest.json> --pairs results/dna/bioreason/perturbations/pairs.json \
        --out results/dna/bioreason/posttraining_sweep/summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict

from input_use.metrics.dna import (canonical_gold, directional_precision, donor_following_rate,
                                   genomic_utilization, label_of, normalize_label_text,
                                   pair_response_table)

ARMS = ("wt", "scramble", "swap_variant_donor")


# --- reading -------------------------------------------------------------------------------------

# A generation built from a handful of distinct characters is a collapsed policy, not an answer.
# distinct character, repeated to the 2,048-token cap.
DEGENERATE_MAX_DISTINCT_CHARS = 3
DEGENERATE_FRACTION = 0.5


LABELS = []      # closed label vocabulary; set in main() from the ground truth


def load_records(path: str) -> dict:
    """{condition: {example_id: parsed_answer}} plus the per-row metadata the aggregation needs."""
    by_cond = defaultdict(dict)
    variant_key, no_answer, degenerate = {}, defaultdict(int), defaultdict(int)
    n, truncated = 0, 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # a preemptible queue preempts with no grace period, so a job can die between two
            # write() calls and leave a half-written final line.
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                truncated += 1
                continue
            n += 1
            eid, cond = r["example_id"], r["condition"]
            raw = (r.get("output") or {}).get("raw") or ""
            # Always re-derive the prediction from the raw generation with the released checkpoints' own
            # metric: map onto the closed label vocabulary, longest label first. Never trust a
            by_cond[cond][eid] = label_of(raw, LABELS)
            variant_key[eid] = (r.get("input") or {}).get("variant_key")
            if not re.search(r"(?i)answer\s*:", raw):
                no_answer[cond] += 1
            if raw and len(set(raw)) <= DEGENERATE_MAX_DISTINCT_CHARS:
                degenerate[cond] += 1
    n_wt = len(by_cond.get("wt", {}))
    deg_rate = degenerate.get("wt", 0) / n_wt if n_wt else 0.0
    if truncated:
        print(f"  [warn] {os.path.basename(path)}: {truncated} unparseable line(s) skipped "
              f"({'expected: a preemption mid-write' if truncated == 1 else 'MORE THAN ONE -- investigate'})")
    return {"by_cond": dict(by_cond), "variant_key": variant_key,
            "n_records": n, "n_truncated": truncated, "no_answer": dict(no_answer),
            "degenerate_count": dict(degenerate),
            "degenerate_rate": deg_rate, "is_degenerate": deg_rate >= DEGENERATE_FRACTION}


# --- aggregation ---------------------------------------------------------------------------------

def genome_mean(hits: dict, variant_key: dict) -> float:
    """Mean over genomes of the within-genome mean. `hits` is example_id -> 0/1."""
    per = defaultdict(list)
    for eid, h in hits.items():
        per[variant_key.get(eid, eid)].append(h)
    if not per:
        return float("nan")
    return sum(sum(v) / len(v) for v in per.values()) / len(per)


def bootstrap_genome(hits: dict, variant_key: dict, n_boot: int, seed: int):
    """Percentile bootstrap over genomes -- the cluster unit, not the row."""
    per = defaultdict(list)
    for eid, h in hits.items():
        per[variant_key.get(eid, eid)].append(h)
    keys = sorted(per)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        draw = [rng.choice(keys) for _ in keys]
        means.append(sum(sum(per[k]) / len(per[k]) for k in draw) / len(draw))
    means.sort()
    return (means[int(0.025 * n_boot)], means[min(int(0.975 * n_boot), n_boot - 1)])


def bootstrap_youden(pairs, predicted, text_of, n_boot: int, seed: int):
    """Percentile bootstrap for U_genome, resampling question texts."""
    cells = defaultdict(lambda: [0, 0, 0, 0])   # text -> [ch|should, no|should, ch|not, no|not]
    for p in pairs:
        should = bool(p["should_change"])
        pa, pb = predicted.get(p["a"]), predicted.get(p["b"])
        if pa is None or pb is None:
            continue
        did = normalize_label_text(pa) != normalize_label_text(pb)
        idx = (0 if did else 1) if should else (2 if did else 3)
        cells[text_of.get(p["a"], p["a"])][idx] += 1
    texts = sorted(cells)
    if len(texts) < 2:
        return (float("nan"), float("nan"))
    vecs = [cells[t] for t in texts]
    rng = random.Random(seed)
    js = []
    for _ in range(n_boot):
        a = b = c = d = 0
        for _ in texts:
            v = vecs[rng.randrange(len(vecs))]
            a += v[0]; b += v[1]; c += v[2]; d += v[3]
        if not (a + b) or not (c + d):
            continue                      # empty margin: J undefined for this resample, not -1
        js.append(a / (a + b) + d / (c + d) - 1.0)
    if len(js) < max(20, n_boot // 10):   # too few valid resamples to quote an interval
        return (float("nan"), float("nan"))
    js.sort()
    lo = js[int(0.025 * len(js))]
    return (lo, js[min(int(0.975 * len(js)), len(js) - 1)])


# --- per-cell scoring ----------------------------------------------------------------------------

def score_cell(rec: dict, pairs_all, gt, donors, net_split, text_of, n_boot, seed) -> dict:
    by_cond, vk = rec["by_cond"], rec["variant_key"]
    wt = by_cond.get("wt", {})

    out = {"n_records": rec["n_records"], "arms_seen": sorted(by_cond),
           "no_answer": rec["no_answer"], "n_wt": len(wt),
           "degenerate_rate": rec["degenerate_rate"], "is_degenerate": rec["is_degenerate"]}

    # --- accuracy, per arm and per network split
    acc = {}
    for arm in ARMS:
        preds = by_cond.get(arm)
        if not preds:
            continue
        hits_all = {eid: int(p is not None and normalize_label_text(p) == canonical_gold(gt[eid]))
                    for eid, p in preds.items() if eid in gt}
        entry = {"all": {"n_rows": len(hits_all), "acc": genome_mean(hits_all, vk)}}
        lo, hi = bootstrap_genome(hits_all, vk, n_boot, seed)
        entry["all"].update({"ci95": [lo, hi],
                             "n_genomes": len({vk.get(e, e) for e in hits_all})})
        for split in ("train", "id_test", "ood_test"):
            h = {e: v for e, v in hits_all.items() if net_split.get(e) == split}
            lo, hi = bootstrap_genome(h, vk, n_boot, seed)
            entry[split] = {"n_rows": len(h), "acc": genome_mean(h, vk), "ci95": [lo, hi],
                            "n_genomes": len({vk.get(e, e) for e in h})}
        acc[arm] = entry
    out["accuracy"] = acc

    # acc(wt) - acc(scramble): the accuracy-side cost of destroying the genomic signal.
    if "wt" in acc and "scramble" in acc:
        out["scramble_delta"] = {s: acc["wt"][s]["acc"] - acc["scramble"][s]["acc"]
                                 for s in ("all", "train", "id_test", "ood_test")}

    # --- U_genome, on all pairs and on the leakage-free subset
    def heldout(p):
        return net_split.get(p["a"]) in ("id_test", "ood_test") and \
               net_split.get(p["b"]) in ("id_test", "ood_test")

    for name, ps in (("all", pairs_all), ("heldout", [p for p in pairs_all if heldout(p)])):
        tab = pair_response_table(ps, wt)
        lo, hi = bootstrap_youden(ps, wt, text_of, n_boot, seed)
        tab["youden_j_ci95"] = [lo, hi]
        tab["defined"] = tab["youden_j"] is not None
        tab["n_pairs_in"] = len(ps)
        tab["n_texts"] = len({text_of.get(p["a"], p["a"]) for p in ps})
        out.setdefault("u_genome", {})[name] = tab

    # U_genome alone cannot say why the answer moved: reading the sequence, or merely receiving a
    # different embedding block.
    out["genomic_utilization"] = genomic_utilization(
        pairs_all, {arm: by_cond[arm] for arm in ("wt", "scramble") if arm in by_cond})

    # A high J is compatible with choosing the direction at chance: within a sensitivity pair the
    # two gold labels are the two plausible answers, so a genome-blind flip is right half the time.
    out["directional_precision"] = directional_precision(
        [p for p in pairs_all if p["should_change"]], wt)

    # --- donor following, on the swap_variant_donor arm
    if "swap_variant_donor" in by_cond:
        donor_answers = {e: d["donor_answer"] for e, d in donors.items()}
        out["donor_swap"] = donor_following_rate(by_cond["swap_variant_donor"], donor_answers, gt)
        sens_rows = {e: p for e, p in by_cond["swap_variant_donor"].items()
                     if e in {q for p in pairs_all if p["should_change"] for q in (p["a"], p["b"])}}
        out["donor_swap_sensitivity_only"] = donor_following_rate(sens_rows, donor_answers, gt)
    return out


# --- main ----------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir holding records_<cell>.jsonl + examples.jsonl")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pairs", required=True, help="pairs.json of the BioReason perturbation run")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    manifest = json.load(open(a.manifest))
    cells = {c["cell"]: c for c in manifest["cells"]}
    pj = json.load(open(a.pairs))
    pairs_all = pj["strata"]["sensitivity"]["pairs"] + pj["strata"]["stability"]["pairs"]
    donors = pj["donors"]
    net_split = json.load(open(os.path.join(a.run, "network_split.json")))

    # ground truth + question text, read once from the wt arm of the shared example set
    gt, text_of = {}, {}
    with open(os.path.join(a.run, "examples.jsonl")) as fh:
        for line in fh:
            e = json.loads(line)
            if e["condition"] != "wt":
                continue
            gt[e["example_id"]] = e["ground_truth"]["answer"]
            text_of[e["example_id"]] = e["payload"]["question"]
    global LABELS
    LABELS = sorted({v for v in gt.values() if v})
    print(f"[posttrain-score] {len(gt)} rows | {len(pairs_all)} pairs "
          f"({sum(p['should_change'] for p in pairs_all)} should-change) | "
          f"{len(set(net_split.values()))} network splits")
    print(f"[posttrain-score] all {sum(p['should_change'] for p in pairs_all)} should-change pairs "
          f"are scorable: distinct entries of the {len(LABELS)}-label vocabulary are distinct by "
          f"construction, so no undecidable-pair exclusion is needed")

    results, missing = {}, []
    for path in sorted(glob.glob(os.path.join(a.run, "records_*.jsonl"))):
        cell = os.path.basename(path)[len("records_"):-len(".jsonl")]
        if cell not in cells:
            print(f"  [skip] {cell}: not in the manifest")
            continue
        rec = load_records(path)
        s = score_cell(rec, pairs_all, gt, donors, net_split, text_of, a.n_boot, a.seed)
        s["cell_meta"] = {k: cells[cell][k] for k in
                          ("family", "model_tag", "kind", "sft_epochs_budget", "rl_epochs",
                           "rl_frac", "grpo_steps", "saved_epochs", "best_val_loss", "src")
                          if k in cells[cell]}
        results[cell] = s
        u = s["u_genome"]["all"]; uh = s["u_genome"]["heldout"]
        acc = s["accuracy"].get("wt", {})
        flag = f"  DEGENERATE ({s['degenerate_rate']:.0%} of wt generations are <=3 distinct chars)" \
               if s["is_degenerate"] else ""
        fmt = lambda v: f"{v:+.3f}" if isinstance(v, float) else "  n/a "
        gu = s.get("genomic_utilization", {})
        dp = s.get("directional_precision", {})
        extra = ""
        if gu.get("content_attributable") is not None:
            extra = (f"  U_scr={fmt(gu.get('u_genome_content_control'))}"
                     f" U_content={fmt(gu.get('content_attributable'))}")
        if dp.get("directional_precision") is not None:
            extra += f" dirP={dp['directional_precision']:.3f}/{dp['n_changed']}"
        print(f"  {cell:26s} n={rec['n_records']:5d}  U_all={fmt(u['youden_j'])} "
              f"U_held={fmt(uh['youden_j'])}{extra}  "
              f"acc id={acc.get('id_test', {}).get('acc', float('nan')):.3f} "
              f"ood={acc.get('ood_test', {}).get('acc', float('nan')):.3f}{flag}")
    for cell in cells:
        if cell not in results:
            missing.append(cell)
    if missing:
        print(f"[posttrain-score] {len(missing)} cells have no records yet: {', '.join(missing)}")

    payload = {"run": a.run, "n_boot": a.n_boot, "seed": a.seed,
               "n_cells_scored": len(results), "cells_missing": missing,
               "n_pairs": len(pairs_all),
               # Under the old lenient matcher this field held the 61 should-change pairs whose two
               # gold labels it could not tell apart.
               "n_undecidable_should_change": 0,
               "label_vocabulary_size": len(LABELS), "metric": "closed-vocabulary label match",
               "results": results}
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[posttrain-score] -> {a.out}")

    # tidy long table, one row per (cell, metric), for the plotting stage
    csv_path = os.path.splitext(a.out)[0] + ".csv"
    with open(csv_path, "w") as fh:
        fh.write("cell,family,model_tag,sft_epochs,rl_epochs,rl_frac,grpo_steps,"
                 "u_genome_all,u_genome_all_lo,u_genome_all_hi,"
                 "u_genome_heldout,u_genome_heldout_lo,u_genome_heldout_hi,"
                 "sens_all,spec_all,acc_train,acc_id_test,acc_ood_test,"
                 "acc_id_lo,acc_id_hi,acc_ood_lo,acc_ood_hi,"
                 "u_genome_scramble,u_content_attributable,frac_surviving_scramble,"
                 "directional_precision,n_changed,"
                 "scramble_delta_ood,follows_donor_rate,no_answer_wt\n")
        for cell, s in sorted(results.items()):
            m, u, uh = s["cell_meta"], s["u_genome"]["all"], s["u_genome"]["heldout"]
            acc = s["accuracy"].get("wt", {})
            g = lambda d, k, dflt=float("nan"): d.get(k, dflt) if isinstance(d, dict) else dflt
            fh.write(",".join(str(x) for x in [
                cell, m.get("family"), m.get("model_tag"), m.get("sft_epochs_budget"),
                m.get("rl_epochs"), m.get("rl_frac"), m.get("grpo_steps"),
                u["youden_j"], u["youden_j_ci95"][0], u["youden_j_ci95"][1],
                uh["youden_j"], uh["youden_j_ci95"][0], uh["youden_j_ci95"][1],
                u["sensitivity"], u["specificity"],
                g(acc.get("train", {}), "acc"), g(acc.get("id_test", {}), "acc"),
                g(acc.get("ood_test", {}), "acc"),
                g(acc.get("id_test", {}), "ci95", [float('nan')] * 2)[0],
                g(acc.get("id_test", {}), "ci95", [float('nan')] * 2)[1],
                g(acc.get("ood_test", {}), "ci95", [float('nan')] * 2)[0],
                g(acc.get("ood_test", {}), "ci95", [float('nan')] * 2)[1],
                s.get("genomic_utilization", {}).get("u_genome_content_control", ""),
                s.get("genomic_utilization", {}).get("content_attributable", ""),
                s.get("genomic_utilization", {}).get("fraction_surviving_scramble", ""),
                s.get("directional_precision", {}).get("directional_precision", ""),
                s.get("directional_precision", {}).get("n_changed", ""),
                s.get("scramble_delta", {}).get("ood_test", float("nan")),
                s.get("donor_swap", {}).get("follows_donor_rate", float("nan")),
                s["no_answer"].get("wt", 0),
            ]) + "\n")
    print(f"[posttrain-score] -> {csv_path}")


if __name__ == "__main__":
    main()
