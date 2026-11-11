'''ratio_scan on its own copy of the A2744 data (the systematics run is rewriting the catalogue), matched to harvey_g1v2.'''
import os, sys, runpy
from pathlib import Path
REPO = Path('/Users/davidharvey/Work/ml_check')
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]
import mass_weighting, sensitivity as sens
_g = sens.get_obs_data
pass  # real catalogue (run only when no other job uses it)
_m = mass_weighting.a2744_mass_gaussian
mass_weighting.a2744_mass_gaussian = lambda study=None: _m(os.environ.get("MASS_STUDY", "jauzac"))
sys.argv = ["ratio_scan.py"] + sys.argv[1:]
runpy.run_path(str(REPO / "sbi" / "ratio_scan.py"), run_name="__main__")
