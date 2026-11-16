#!/usr/bin/env python
'''Re-evaluate base checkpoints (stage 1, noiseless shear maps) on the test split that main.py builds for
their seed: source = BAHAMAS, target = DARKSKIES. Same seed -> same split for the CDAN and no-DA models.

    astro-env/bin/python notebooks/eval_base_models.py models_base_cmp/base_models/*_final.pth
'''
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
import os
os.chdir(REPO)
from dataset import prepare_dataloaders  # noqa: E402
from model import create_model  # noqa: E402


@torch.no_grad()
def accuracy(model, loaders):
    model.eval()
    ok = n = 0
    per_cs = {}
    for batch in loaders[0]:
        x, cs, y = batch[:3]
        pred = model([x])["classification"].argmax(1)
        ok += (pred == y).sum().item(); n += len(y)
        for c in np.unique(cs.numpy()):
            m = cs.numpy() == c
            a = per_cs.setdefault(float(c), [0, 0]); a[0] += (pred[m] == y[m]).sum().item(); a[1] += m.sum()
    return ok / n, {k: v[0] / v[1] for k, v in sorted(per_cs.items())}


out = {}
for f in sys.argv[1:]:
    ck = torch.load(f, weights_only=False, map_location="cpu")
    args = ck["args"]
    args.data_dir = str(REPO / "data") + "/"
    args.device = torch.device("cpu")
    args.num_workers = 0
    args.checkpoint = f
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    loaders = prepare_dataloaders(args)
    model = create_model(args).eval()
    src, src_cs = accuracy(model, loaders["source_val"])
    tgt, tgt_cs = accuracy(model, loaders["target_test"])
    name = Path(f).name
    out[name] = {"seed": args.seed, "adaptation": args.adaptation, "source_acc": src, "target_acc": tgt,
                 "target_acc_by_cs": tgt_cs, "source_acc_by_cs": src_cs}
    print(f"{args.adaptation:5s} seed {args.seed} {name.split('_')[-1]:10s} source {src:.4f} target {tgt:.4f}", flush=True)
json.dump(out, open(REPO / "notebooks/results/paper_numbers/base_model_accuracy.json", "w"), indent=1)
