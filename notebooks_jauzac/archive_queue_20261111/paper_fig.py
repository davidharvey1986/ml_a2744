# Paper figures with the Jauzac A2744 mass prior, final model: python paper_fig.py <function> [args...]
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path[:0] = [os.path.join(HERE, "..", "notebooks"), os.path.join(HERE, "..")]
import mass_weighting
_g = mass_weighting.a2744_mass_gaussian
mass_weighting.a2744_mass_gaussian = lambda study="jauzac": _g("jauzac")
import plots_for_paper as p
os.chdir(HERE)
getattr(p, sys.argv[1])(*[eval(a) for a in sys.argv[2:]])
