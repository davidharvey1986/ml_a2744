#!/usr/bin/env python
'''
Every figure in the A2744 paper (aa_a2744.tex), from one place.

Each figure is one function in plots_for_paper.py; this script runs them, each in its own process (matplotlib/LaTeX
state and the catalogue side effects of the imports stay contained), and collects the results under the names the
paper uses in paper_figures/<mass study>/. --push also copies them into the Overleaf clone's Figures/ (no commit).

Figures that read the A2744 catalogue (CAT below) rebuild the shared filtered catalogue, so they run one at a time
and never while another job is using it. The others are run with the catalogue blocked and the source redshift fixed
to its fiducial value, so they can run at any time.

    astro-env/bin/python notebooks/paper_figures.py --list
    astro-env/bin/python notebooks/paper_figures.py                       # all figures, Jauzac mass
    astro-env/bin/python notebooks/paper_figures.py --only model_weighting output_to_model --push
    astro-env/bin/python notebooks/paper_figures.py --no-catalogue        # skip the catalogue figures
    astro-env/bin/python notebooks/paper_figures.py --mass-study harvey_g1v2 --out-dir paper_figures_harvey

Heavy computation is not redone here: the summaries (sbi/make_summaries.py), the sensitivity maps
(sbi/pair_sensitivity.py), the systematics run (sbi/systematics.py) and the shear-ratio scan (sbi/ratio_scan.py; or
--recompute-ratio) must exist. Order matters for one dependency: output_to_model writes plots/gp_posterior_ba_<tag>.json, which
particle_physics and the model systematics read (it is run first).
'''
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PY = sys.executable
OVERLEAF = Path("/Users/davidharvey/Papers/jwst_a2744/aa_draft/Figures")
ZS_FIDUCIAL = 2.2484412345861355            # mean source redshift of the fiducial catalogue (get_source_redshift)

# name: (call in plots_for_paper, needs the catalogue, {file it writes in notebooks/: name in the paper}, in the tex)
FIGURES = {
    "data_examples": ("figure1()", True, {"plots/data_examples.png": "data_examples.png"}, True),
    "model_weighting": ('figure2_summaries(paper_paths(TAG)["summaries"], {d: paper_paths(TAG)["flamingo"] for d in ("flamingo", "tng")}, '
                '2, f"ba_{TAG}", False)', False,
                {"plots/final_model_weighting_with_data_ba_{tag}.pdf": "final_model_weighting_with_data.pdf"}, True),
    "p_vs_mass": ("figure_p_vs_mass(TAG)", False, {"plots/p_vs_mass_ba_{tag}.pdf": "p_vs_mass.pdf"}, True),
    "sensitivity_map": ("figure_sensitivity_map()", True, {"plots/sensitivity_map_pairs.pdf": "sensitivity_map.pdf"}, True),
    "sensitivity_relation": ("figure4_sensitivity(TAG)", False,
                             {"plots/Sensitivty_relation_ba_{tag}.pdf": "Sensitivty_relation.pdf"}, True),
    "shear_ratio": ("figure_shear_ratio(recompute=RECOMPUTE_RATIO, tag=TAG)", False,
                    {"plots/shear_ratio_scatter_50-100.pdf": "shear_ratio_scatter_50-100.pdf"}, True),
    "output_to_model": ("figure5_summaries(TAG, 2, True)", False, {"plots/output_to_model_ba_{tag}.pdf": "output_to_model.pdf"}, True),
    "particle_physics": ("figure6_summaries(TAG, merger=True)", False, {"plots/particle_physics.pdf": "particle_physics.pdf"}, True),
    "intrinsic_ell": ("figureA1()", True, {"plots/intrinsic_ell.pdf": "intrinsic_ell.pdf"}, True),
    "correlated_models": ("figureA2_summaries(TAG)", False,
                          {"plots/correlated_models_ba_{tag}.pdf": "correlated_models.pdf"}, True),
    "systematics": ("figure_systematics(TAG)", False,
                    {"plots/systematics_model_paper_ba_{tag}.pdf": "systematics_model.pdf",
                     "plots/systematics_data_paper_ba_{tag}.pdf": "systematics_data.pdf"}, True),
    "pooled_tail": ("figure_pooled_tail(TAG)", False, {"plots/cdm_pooled_tail_{tag}_nc2.pdf": "cdm_pooled_tail.pdf"}, False),
    "empirical_posterior": ("figure_empirical_posterior(TAG)", False,
                            {"plots/empirical_posterior_ba_{tag}.pdf": "empirical_posterior.pdf"}, False),
}
# Kept as they are on Overleaf unless asked for by name (--only): the stamp examples have not changed with the
# model, and the sensitivity relation has not been redrawn for the current pipeline.
NOT_BY_DEFAULT = {"data_examples", "sensitivity_relation"}

RUNNER = '''
import os, sys
sys.path[:0] = [{here!r}, {repo!r}]
os.chdir({here!r})
if {nocat}:
    import add_shear_to_data as asd
    asd.get_source_redshift = lambda *a, **k: {zs!r}
    def _block(*a, **k):
        raise RuntimeError("this figure reads the catalogue: run it without --no-catalogue")
    asd.get_obs_data = _block
from plots_for_paper import *
TAG, RECOMPUTE_RATIO = {tag!r}, {recompute!r}
{call}
'''


def run(name, opts):
    call, needs_cat, outputs, _ = FIGURES[name]
    nocat = not needs_cat and not (name == "shear_ratio" and opts.recompute_ratio)
    code = RUNNER.format(here=str(HERE), repo=str(REPO), nocat=nocat, zs=ZS_FIDUCIAL, tag=opts.tag,
                         recompute=opts.recompute_ratio, call=call)
    t0 = time.time()
    env = {**os.environ, "MASS_STUDY": opts.mass_study, "MPLBACKEND": "Agg"}
    r = subprocess.run([PY, "-c", code], cwd=HERE, env=env, capture_output=True, text=True)
    log = opts.out_dir / "logs" / f"{name}.log"
    log.write_text(r.stdout + r.stderr)
    if r.returncode:
        print(f"  {name}: FAILED ({time.time() - t0:.0f} s), see {log}")
        print("    " + "\n    ".join((r.stderr.strip().splitlines() or ["?"])[-4:]))
        return False
    for src, dst in outputs.items():
        src = HERE / src.format(tag=opts.tag)
        if not src.exists():
            print(f"  {name}: {src} not written")
            return False
        shutil.copy(src, opts.out_dir / dst)
        if src.suffix == ".pdf" and src.with_suffix(".png").exists():
            shutil.copy(src.with_suffix(".png"), opts.out_dir / Path(dst).with_suffix(".png"))
        if opts.push:
            shutil.copy(src, OVERLEAF / dst)
    print(f"  {name}: {', '.join(outputs.values())} ({time.time() - t0:.0f} s)")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(FIGURES), help="these figures (default: all but %s)" % ", ".join(sorted(NOT_BY_DEFAULT)))
    p.add_argument("--mass-study", default="jauzac", help="mass_weighting study for the A2744 mass (MASS_STUDY)")
    p.add_argument("--tag", default="final", help="model checkpoint tag (summaries_wb1_<tag>_n5)")
    p.add_argument("--no-catalogue", action="store_true", help="skip the figures that read the A2744 catalogue")
    p.add_argument("--recompute-ratio", action="store_true", help="rerun the shear-ratio scan (reads the catalogue)")
    p.add_argument("--out-dir", default=None, help="default notebooks/paper_figures/<mass study>")
    p.add_argument("--push", action="store_true", help=f"also copy into {OVERLEAF} (commit and push yourself)")
    p.add_argument("--list", action="store_true")
    opts = p.parse_args()
    if opts.list:
        for n, (call, cat, outs, tex) in FIGURES.items():
            flags = ("catalogue " if cat else "") + ("" if tex else "(not in the tex) ") + ("(only by name)" if n in NOT_BY_DEFAULT else "")
            print(f"{n:22s} {', '.join(outs.values()):55s} {flags}")
        return
    opts.out_dir = Path(opts.out_dir or HERE / "paper_figures" / opts.mass_study).resolve()
    (opts.out_dir / "logs").mkdir(parents=True, exist_ok=True)
    names = opts.only or [n for n in FIGURES if n not in NOT_BY_DEFAULT]
    if opts.no_catalogue:
        names = [n for n in names if not FIGURES[n][1] and not (n == "shear_ratio" and opts.recompute_ratio)]
    if "output_to_model" in names:            # particle_physics and the model systematics read its GP posterior
        names.remove("output_to_model")
        names.insert(0, "output_to_model")
    print(f"mass study {opts.mass_study}, tag {opts.tag}: {len(names)} figures -> {opts.out_dir}")
    failed = [n for n in names if not run(n, opts)]
    print("done" + (f"; failed: {' '.join(failed)}" if failed else ""))
    sys.exit(bool(failed))


if __name__ == "__main__":
    main()
