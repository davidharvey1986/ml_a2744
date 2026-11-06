# Paper figures 2, 5, 6 and A3 with the Jauzac A2744 mass prior instead of Harvey's. Run from this directory
# (a sibling of notebooks/, so plots_for_paper's relative paths resolve); outputs go to ./plots/.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path[:0] = [os.path.join(HERE, "..", "notebooks"), os.path.join(HERE, "..")]
import mass_weighting
_g = mass_weighting.a2744_mass_gaussian
mass_weighting.a2744_mass_gaussian = lambda study="jauzac": _g("jauzac")
print("A2744 mass prior:", mass_weighting.a2744_mass_gaussian())
import plots_for_paper as p
ONLY = sys.argv[1:]          # e.g. "rest": skip what the first run already made (figures 2 and 5 for final)
for t in ("final", "best"):
    P = p.paper_paths(t)
    if not (ONLY and t == "final"):
        p.figure2_summaries(P["summaries"], {d: P["flamingo"] for d in ("flamingo", "tng")}, tag=f"ba_{t}")
    ls, le = p.figure5_summaries(t)
    if t == "final":
        p.figure6(ls, le)
        p.figureA3_summaries(t)
