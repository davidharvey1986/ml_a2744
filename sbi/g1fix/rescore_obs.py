#!/usr/bin/env python
'''
Re-score only the A2744 map (data/100/a2744/obs_data_<filter>.pkl) with an ensemble and write obs.npz
into each summaries directory given, leaving the simulation summaries alone: for a change to the data
catalogue that does not touch the simulations (sbi/g1fix_after.sh).

    astro-env/bin/python sbi/g1fix/rescore_obs.py --model tweakedalign_k0p9_ctr_ba_best --include-str spin2_cx_ft_ \
        --dirs sbi/summaries_spin2cx_best_n5 sbi/summaries_spin2cx_best_n5_flamingo
'''
import argparse, sys
from pathlib import Path
import numpy as np
import torch

LAUNCH = Path.cwd()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--include-str", default="spin2_cx_ft_")
p.add_argument("--filter", default="concat")
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--dirs", nargs="+", required=True)
opts = p.parse_args()
sys.argv = sys.argv[:1]
import make_summaries as ms           # noqa: E402  (chdirs to notebooks/)

args = ms.default_args               # as ms.setup_args, without the redshift map the A2744 map does not need
args.jwst_filter, args.apply_intrinsic_ell, args.ignore_dataset, args.verbose, args.num_workers = opts.filter, 1.0, [""], False, 0
dev = ms.get_device()
models = ms.load_models(argparse.Namespace(filter=opts.filter, model=opts.model, include_str=opts.include_str, nmodels=None), args, dev)
obs_file = f"../data/100/a2744/obs_data_{opts.filter}.pkl"
_, obs = ms.pkl.load(open(obs_file, "rb"))
latents, softmax = ms.run_models(models, torch.tensor(obs, dtype=torch.float32), dev, opts.batch_size)
for d in opts.dirs:
    d = LAUNCH / d
    old = np.load(d / "obs.npz")["softmax"][:, 0] if (d / "obs.npz").exists() else None
    np.savez(d / "obs.npz", latents=latents, softmax=softmax)
    print(f"{d.name}: A2744 P(SIDM) = {softmax[:, 0].mean():.4f} +/- {softmax[:, 0].std():.4f}"
          + ("" if old is None else f" (was {old.mean():.4f})"))
