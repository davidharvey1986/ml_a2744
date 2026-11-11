'''pairs_obs_panel on its own copy of the A2744 data (other jobs use the catalogue): every get_obs_data call,
including the import-time ones, is redirected to data/100/a2744_ratio_copy.'''
import sys, runpy
from pathlib import Path
REPO = Path('/Users/davidharvey/Work/ml_check')
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]
import add_shear_to_data as asd
COPY = str(REPO / "data" / "100" / "a2744_ratio_copy")
_g = asd.get_obs_data
def _redirect(*a, **k):
    k["data_dir"] = COPY
    return _g(*a, **k)
asd.get_obs_data = _redirect
sys.argv = ["pairs_obs_panel.py"] + sys.argv[1:]
runpy.run_path(str(REPO / "sbi" / "pairs_obs_panel.py"), run_name="__main__")
