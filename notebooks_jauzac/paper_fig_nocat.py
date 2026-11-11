# As paper_fig.py, but never rebuilds the A2744 catalogue (source redshift fixed to its value), so it can run
# while another job uses the catalogue: python paper_fig_nocat.py <function> [args...]
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path[:0] = [os.path.join(HERE, "..", "notebooks"), os.path.join(HERE, "..")]
import add_shear_to_data as asd
asd.get_source_redshift = lambda *a, **k: 2.2484412345861355
def _block(*a, **k):
    raise RuntimeError("this figure needs the catalogue: use paper_fig.py")
asd.get_obs_data = _block
import plots_for_paper as p
os.chdir(HERE)
getattr(p, sys.argv[1])(*[eval(a) for a in sys.argv[2:]])
