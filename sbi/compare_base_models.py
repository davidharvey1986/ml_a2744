#!/usr/bin/env python
'''
Compare two sets of base models (stage 1 of train_seed.sh) on the clusters each seed did not
train on: its own base-stage test split of data/100/shear (create_balanced_test_set with the
model's seed) plus the files it ignored (bahamas_1). The inputs are the noise-free shear maps
the base stage trains on, normalised as BalancedClusterDataset.

Per model and domain: AUC of P(SIDM) for CDM against all SIDM clusters, and the mean P(SIDM)
per cross-section. Then the same with every shear rotated by 45 deg (E -> B): a network that
reads the lensing signal should lose the CDM / SIDM separation; the E/B-blind models trained
before the spin-2 augmentation fix will not. With --shift-kpc, also with every map moved off
centre by dataset.CentreJitter (same offsets for both sets): centre-jittered models should
hold their AUC there.

    astro-env/bin/python sbi/compare_base_models.py \
        --old /Users/davidharvey/Work/Dark_Matter_DA/code/models/base_models --old-pattern '*_nob1_final.pth' \
        --new models/base_models_spin2 --new-pattern '*_nob1_spin2_final.pth'
'''
import argparse
import glob
import json
import os
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from dataset import FOV_KPC, CentreJitter, get_cross_section_from_filename  # noqa: E402
from model import create_model  # noqa: E402

DOMAINS = ["bahamas", "darkskies"]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", required=True)
    p.add_argument("--old-pattern", default="*_nob1_final.pth")
    p.add_argument("--new", required=True)
    p.add_argument("--new-pattern", default="*_nob1_spin2_final.pth")
    p.add_argument("--old-label", default="old")
    p.add_argument("--new-label", default="new")
    p.add_argument("--shift-kpc", type=float, default=0., help="also evaluate on maps shifted by this jitter per axis")
    p.add_argument("--max-per-file", type=int, default=150, help="held-out clusters per file and model (speed)")
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "compare_base_models"))
    return p.parse_args()


def load(paths, args, device):
    models = []
    for path in paths:
        args.checkpoint = path
        m = create_model(args).to(device).eval()
        m.checkpoint_name = os.path.basename(path)
        models.append(m)
    return models


def rotate45(t):
    '''Every shear turned by 45 deg: (e1, e2) -> (-e2, e1), E-mode -> B-mode.'''
    out = t.clone()
    out[:, 0], out[:, 1] = -t[:, 1], t[:, 0]
    return out


@torch.no_grad()
def evaluate(model, domain, shear, device, max_per_file, rng, shift_kpc=0.):
    files = sorted(shear)
    counts = {f: shear[f].shape[0] for f in files}
    test = ms.training_test_set(f"100/shear/{domain}*", counts, model.args)
    ignored = set(getattr(model.args, "ignore_dataset", []))
    res = {}
    variants = [("E", lambda t: t), ("B", rotate45)]
    if shift_kpc > 0:
        jitter = CentreJitter(shift_kpc / (FOV_KPC / 100))
        variants.append(("shift", lambda t: jitter(t[0])[None]))
    for lab, tf in variants:
        torch.manual_seed(0)                                     # the same offsets for every model
        p, sig = [], []
        for f in files:
            idx = np.array(sorted(i for i in range(counts[f]) if f in ignored or (f, i) in test))
            if len(idx) > max_per_file:
                idx = np.sort(rng.choice(idx, max_per_file, replace=False))
            imgs = torch.stack([sens.normalise(tf(torch.tensor(shear[f][i:i + 1, :2], dtype=torch.float32))[0])[0] for i in idx])
            p.append(sens.model_outputs([model], imgs, device, 128)[0])
            sig.append(np.full(len(idx), get_cross_section_from_filename(f)))
        p, sig = np.concatenate(p), np.concatenate(sig)
        res[lab] = {"auc": float(roc_auc_score(sig > 0, p)),
                    "p_mean": {f"{s:g}": float(p[sig == s].mean()) for s in np.unique(sig)}}
    return res


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = ms.get_device()
    args = ms.setup_args("concat")
    sets = {"old": sorted(glob.glob(os.path.join(opts.old, opts.old_pattern))),
            "new": sorted(glob.glob(os.path.join(opts.new, opts.new_pattern)))}
    labels = {"old": opts.old_label, "new": opts.new_label}
    print({labels[k]: len(v) for k, v in sets.items()})
    results = {}
    for domain in DOMAINS:
        shear = {}
        for path in sorted(glob.glob(str(REPO / "data" / "100" / "shear" / f"{domain}*.pkl"))):
            _, d = pkl.load(open(path, "rb"))
            shear[os.path.basename(path)] = d[:, :2]
        for name, paths in sets.items():
            rows = []
            for path in paths:
                m = load([path], args, device)[0]
                r = evaluate(m, domain, shear, device, opts.max_per_file, np.random.default_rng(m.args.seed),
                             opts.shift_kpc)
                rows.append({"seed": int(m.args.seed), **r})
            results.setdefault(domain, {})[name] = rows
            auc_e = np.array([r["E"]["auc"] for r in rows])
            auc_b = np.array([r["B"]["auc"] for r in rows])
            pm = {s: np.mean([r["E"]["p_mean"][s] for r in rows]) for s in rows[0]["E"]["p_mean"]}
            shifted = (f" | shifted {opts.shift_kpc:g} kpc {np.mean([r['shift']['auc'] for r in rows]):.3f} ± "
                       f"{np.std([r['shift']['auc'] for r in rows]):.3f}") if opts.shift_kpc > 0 else ""
            print(f"{domain:9s} {labels[name]}: {len(rows)} models | AUC (CDM vs SIDM) E {auc_e.mean():.3f} ± {auc_e.std():.3f}"
                  f" | after 45 deg rotation {auc_b.mean():.3f} ± {auc_b.std():.3f}{shifted} | mean P(SIDM) "
                  + " ".join(f"σ={s}: {v:.3f}" for s, v in pm.items()))
        del shear
    json.dump(results, open(out / "compare_base_models.json", "w"), indent=1)

    plt = sens.style()
    fig, axes = plt.subplots(1, 2, figsize=(13 if opts.shift_kpc > 0 else 11, 4.2), constrained_layout=True)
    for ax, domain in zip(axes, DOMAINS):
        for name, c, off in [("old", "#9a9a9a", -0.12), ("new", "#0072B2", 0.12)]:
            rows = results[domain][name]
            for k, (lab, x) in enumerate([("E", 0), ("B", 1)] + ([("shift", 2)] if opts.shift_kpc > 0 else [])):
                v = [r[lab]["auc"] for r in rows]
                ax.scatter(np.full(len(v), x + off) + np.random.default_rng(k).uniform(-0.04, 0.04, len(v)), v, s=12,
                           color=c, alpha=0.7, label=f"{labels[name]} base models ({len(v)})" if k == 0 else None)
                ax.errorbar(x + off, np.mean(v), np.std(v), fmt="s", color="k", ms=5, capsize=3)
        ax.axhline(0.5, color="#6b6b6b", lw=1, ls=":")
        ax.set_xticks([0, 1] + ([2] if opts.shift_kpc > 0 else []), ["lensing shear (E)", "shears rotated 45° (B)"]
                      + ([f"shifted, {opts.shift_kpc:g} kpc jitter"] if opts.shift_kpc > 0 else []))
        ax.set_ylabel("AUC, CDM vs SIDM (held-out clusters)")
        ax.set_title(domain, fontsize=10)
        ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"compare_base_models.{ext}", dpi=150)


if __name__ == "__main__":
    main()
