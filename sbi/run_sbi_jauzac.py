# run_sbi.py with another A2744 mass prior (env MASS_STUDY, default jauzac): python sbi/run_sbi_jauzac.py <run_sbi args>
import os, runpy, sys
STUDY = os.environ.get("MASS_STUDY", "jauzac")
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "sbi")]
import mass_weighting
_g, _l = mass_weighting.a2744_mass_gaussian, mass_weighting.a2744_log_mass
mass_weighting.a2744_mass_gaussian = lambda study=None: _g(STUDY)
mass_weighting.a2744_log_mass = lambda nsigma=2, study=None: _l(nsigma, STUDY)
sys.argv[0] = str(REPO / "sbi" / "run_sbi.py")
runpy.run_path(sys.argv[0], run_name="__main__")
