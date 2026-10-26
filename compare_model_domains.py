#!/usr/bin/env python
"""
Compare the outputs of trained models on the bahamas and darkskies obs data.

For every model in --models and every data/100/obs/<filter>/{bahamas,darkskies}*.pkl
file, compute P(SIDM) for each cluster using the same loading, test-time
transforms and normalisation as training (dataset.BalancedClusterDataset), then
summarise per cross-section and domain:

  - mean P(SIDM) per class (median and 16-84% range over models)
  - the same, reweighted so both domains have the bahamas_cdm mass distribution
  - CDM-vs-SIDM AUC per domain
  - the CDM gap between domains (darkskies - bahamas)

Per-cluster probabilities are saved so the plots can be remade without
re-running the models.

Usage:
    python compare_model_domains.py
    python compare_model_domains.py --models "models/concat/*seed_1*.pth" --max_z 0.4
"""
import argparse
import glob
import os
import pickle as pkl

import numpy as np
import torch
from torchvision import transforms

from dataset import (BalancedClusterDataset, Downsample, apply_intrinsic_ell,
                     apply_shape_measurement_bias, cluster_member_contamination,
                     rescale_lens_source_configuration)
from model import create_model


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="models/concat/*.pth", help="glob of model checkpoints")
    p.add_argument("--data_dir", default="data/")
    p.add_argument("--jwst_filter", default="concat")
    p.add_argument("--max_z", type=float, default=0.4, help="drop clusters at or above this redshift")
    p.add_argument("--mass_bins", type=float, nargs=3, default=[14.3, 15.0, 0.05],
                   help="log10 M range and bin width used for the mass reweighting")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0, help="seed for the shape-noise realisation")
    p.add_argument("--output", default="results/model_domain_comparison.pkl")
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(paths, device):
    models = []
    for path in paths:
        args = torch.load(path, weights_only=False, map_location="cpu")["args"]
        args.checkpoint = path
        args.verbose = False
        args.device = device
        model = create_model(args).to(device).eval()
        models.append(model)
    return models


def eval_args(model_args, opts):
    '''Training args from the checkpoint, pointed at the obs data with test-time defaults filled in.'''
    args = model_args
    args.data_dir = opts.data_dir
    args.jwst_filter = opts.jwst_filter
    args.verbose = False
    args.ignore_dataset = ['']        # evaluate every file, including any ignored in training
    for name, default in [("downsample", 1), ("cluster_member_contamination", 0.),
                          ("med_norm", -1), ("use_normalization", False), ("use_log_transform", False),
                          ("log_mass_cut", 0.)]:
        if not hasattr(args, name):
            setattr(args, name, default)
    if not hasattr(args, "shape_measurement_bias"):
        args.shape_measurement_bias = {'e1': {'c': 0., 'm': 0.}, 'e2': {'c': 0., 'm': 0.}}
    return args


def test_transform(args):
    # Same as prepare_dataloaders' test_transform
    return transforms.Compose([
        rescale_lens_source_configuration(args),
        cluster_member_contamination(args),
        apply_shape_measurement_bias(args.shape_measurement_bias),
        apply_intrinsic_ell(args),
        Downsample(args),
    ])


def log_mass(meta):
    if 'mass' in meta:
        return np.asarray(meta['mass'], dtype=np.float64)
    return np.log10(np.asarray(meta['Mass_200crit'], dtype=np.float64)*1e10)


def evaluate_file(path, models, args, transform, device, batch_size):
    '''Return per-model P(SIDM) (n_models, N) and the matching meta indices.'''
    rel = os.path.relpath(path, args.data_dir)
    dataset = BalancedClusterDataset(args.data_dir, rel, transform=transform, args=args, dtype='image')

    # Build the inputs once (one shape-noise realisation) and reuse them for every model
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    img_idx = np.array([dataset.file_indices[i][1] for i in range(len(dataset))])

    probs = np.zeros((len(models), len(dataset)), dtype=np.float32)
    with torch.no_grad():
        for m, model in enumerate(models):
            for s in range(0, len(images), batch_size):
                x = images[s:s+batch_size].to(device)
                out = model([x])["classification"]
                probs[m, s:s+batch_size] = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    return probs, img_idx


def weighted_mean(values, weights):
    return np.sum(values*weights, axis=-1)/np.sum(weights)


def auc(neg, pos):
    '''ROC AUC that a positive scores above a negative (Mann-Whitney).'''
    allv = np.concatenate([neg, pos])
    ranks = np.argsort(np.argsort(allv)) + 1
    return (ranks[len(neg):].sum() - len(pos)*(len(pos)+1)/2)/(len(neg)*len(pos))


def main():
    opts = parse()
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    device = get_device()

    paths = sorted(glob.glob(opts.models))
    assert paths, f"No models match {opts.models}"
    print(f"{len(paths)} models | device {device}")
    models = load_models(paths, device)
    args = eval_args(models[0].args, opts)
    transform = test_transform(args)

    files = sorted(glob.glob(f"{opts.data_dir}/100/obs/{opts.jwst_filter}/bahamas*.pkl")) + \
            sorted(glob.glob(f"{opts.data_dir}/100/obs/{opts.jwst_filter}/darkskies*.pkl"))
    files = [f for f in files if '_vd' not in f]

    results = {}
    for path in files:
        name = os.path.basename(path).replace('.pkl', '')
        meta, _ = pkl.load(open(path, 'rb'))
        probs, idx = evaluate_file(path, models, args, transform, device, opts.batch_size)
        results[name] = dict(probs=probs, redshift=np.asarray(meta['redshift'], float)[idx],
                             log_mass=log_mass(meta)[idx], cross_section=dataset_cross_section(name))
        print(f"  {name:16s} N={probs.shape[1]:5d}  mean P(SIDM) over models {probs.mean():.3f}")

    # Mass reweighting to the bahamas_cdm mass distribution
    lo, hi, width = opts.mass_bins
    bins = np.arange(lo, hi + width/2, width)
    ref = results['bahamas_cdm']
    ref_ok = (ref['redshift'] < opts.max_z) & (ref['log_mass'] >= lo) & (ref['log_mass'] < hi)
    target = np.histogram(ref['log_mass'][ref_ok], bins)[0].astype(float)
    target /= target.sum()

    summary = {}
    for name, r in results.items():
        ok = r['redshift'] < opts.max_z
        in_range = ok & (r['log_mass'] >= lo) & (r['log_mass'] < hi)
        b = np.clip(np.digitize(r['log_mass'], bins) - 1, 0, len(bins) - 2)
        have = np.histogram(r['log_mass'][in_range], bins)[0]
        w = np.where(in_range, target[b]/np.maximum(have[b], 1), 0.)
        per_model = r['probs'][:, ok].mean(axis=1)
        per_model_w = weighted_mean(r['probs'], w)
        summary[name] = dict(n=int(ok.sum()), n_matched=int(in_range.sum()),
                             mean=per_model, mean_mass_matched=per_model_w,
                             log_mass=np.median(r['log_mass'][ok]))

    print_summary(summary, results, opts)

    os.makedirs(os.path.dirname(opts.output) or '.', exist_ok=True)
    pkl.dump(dict(models=paths, results=results, summary=summary, options=vars(opts)), open(opts.output, 'wb'))
    print(f"\nSaved per-cluster probabilities and summary to {opts.output}")


def dataset_cross_section(name):
    tag = name.split('_', 1)[1]
    return 0. if tag == 'cdm' else float(tag)


def q(v):
    return f"{np.median(v):.3f} [{np.percentile(v, 16):.3f}-{np.percentile(v, 84):.3f}]"


def print_summary(summary, results, opts):
    print(f"\nMean P(SIDM) per class: median over models [16-84%], z < {opts.max_z}")
    print(f"{'dataset':16s} {'N':>5s} {'logM':>5s}  {'all clusters':>22s}  {'mass-matched to bahamas_cdm':>28s}")
    for name, s in summary.items():
        print(f"{name:16s} {s['n']:5d} {s['log_mass']:5.2f}  {q(s['mean']):>22s}  {q(s['mean_mass_matched']):>28s}")

    print("\nCDM vs SIDM separation per domain (AUC of P(SIDM), median over models)")
    for domain in ['bahamas', 'darkskies']:
        cdm = results.get(f'{domain}_cdm')
        if cdm is None:
            continue
        sidm = [r for n, r in results.items() if n.startswith(domain) and r['cross_section'] > 0]
        aucs = []
        for m in range(cdm['probs'].shape[0]):
            neg = cdm['probs'][m][cdm['redshift'] < opts.max_z]
            pos = np.concatenate([r['probs'][m][r['redshift'] < opts.max_z] for r in sidm])
            aucs.append(auc(neg, pos))
        print(f"  {domain:10s} AUC {q(np.array(aucs))}")

    print("\nCDM gap, darkskies - bahamas mean P(SIDM) (median over models)")
    for key, label in [('mean', 'all clusters'), ('mean_mass_matched', 'mass-matched')]:
        gap = summary['darkskies_cdm'][key] - summary['bahamas_cdm'][key]
        print(f"  {label:13s} {q(gap)}")


if __name__ == "__main__":
    main()
