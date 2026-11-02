'''
Switch the pipeline to the 2-D shear susceptibility G1(size, S/N) (shear_susceptibility.py):
add_shear_to_data.get_obs_data applies it after calc_shear, and Figure A1 panel 4 shows it.
Run by sbi/g1fix_after.sh once nothing reads the catalogues. Idempotent.
'''
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]

edits = {
    REPO / "add_shear_to_data.py": [
        ("from pyRRG.calc_shear import calc_shear\n",
         "from pyRRG.calc_shear import calc_shear\nfrom shear_susceptibility import correct_shears\n"),
        ("SHEAR_CUTS = {",
         "G1_2D = True       # shear susceptibility G1(size, S/N) (shear_susceptibility.py) instead of pyRRG's S/N arctan; from 2026-11-02\n"
         "SHEAR_CUTS = {"),
        ("        calc_shear(fits.open(shear_cat)[1].data, filtered_name, **filter_cuts)\n",
         "        calc_shear(fits.open(shear_cat)[1].data, filtered_name, **filter_cuts)\n"
         "        if G1_2D:\n"
         "            correct_shears(filtered_name, verbose=False)\n"),
    ],
}

# Figure A1 panel 4: the fiducial cuts, binned G1 against size per S/N tercile, the G1 used (solid)
# and pyRRG's (dotted)
p = REPO / "notebooks" / "plots_for_paper.py"
s = p.read_text()
new_panel = '''    from shear_susceptibility import fit_g1
    from scipy.stats import binned_statistic as _bs
    color = {'f115w': 'b', 'f150w': 'r'}
    for ifilter in ['f115w', 'f150w']:
        cat = fits.open(f"../data/100/a2744/abell2744clu-grizli-v5.4-{ifilter}-clear_drc_sci_clean.shears")[1].data
        calc_shear(cat, "test.fits", **{**fid_cut, **SHEAR_CUTS[ifilter]})
        data = fits.open("test.fits")[1].data
        model, _ = fit_g1(data, verbose=False)
        ok = np.isfinite(data['g1_gal'])
        size, snr, g1 = data['gal_size'][ok], data['snr'][ok], data['g1_gal'][ok]
        g1_2d, g1_pyrrg = model(size, snr), data['g1_model'][ok]
        xb = np.quantile(size, np.linspace(0, 1, 21))
        xc = _bs(size, size, 'median', bins=xb)[0]
        ax.errorbar(xc, _bs(size, g1, 'mean', bins=xb)[0], _bs(size, g1, std_err, bins=xb)[0],
                    capsize=2, fmt='o', color=color[ifilter], label=ifilter)
        ax.plot(xc, _bs(size, g1_2d, 'mean', bins=xb)[0], '-', color=color[ifilter])
        ax.plot(xc, _bs(size, g1_pyrrg, 'mean', bins=xb)[0], ':', color=color[ifilter])
    ax.plot([], [], 'k-', label=r'$G_1$(size, S/N) used')
    ax.plot([], [], 'k:', label=r'S/N-only fit')
    ax.set_xscale('log')
    ax.set_xlabel('Galaxy size (pixels)', fontsize=15)
    ax.set_ylabel('Lensing susceptibility tensor',fontsize=15)
    ax.legend(fontsize=12)
'''
if "from shear_susceptibility import fit_g1" not in s:
    start = s.index("    cuts= {'f115w':{'signal_noise_cut':4, 'stat_type':'snr'}, ")
    end = s.index("    ax.set_ylabel('Lensing susceptibility tensor',fontsize=15)\n    ax.legend(fontsize=15)\n")
    end += len("    ax.set_ylabel('Lensing susceptibility tensor',fontsize=15)\n    ax.legend(fontsize=15)\n")
    s = s[:start] + new_panel + s[end:]
    p.write_text(s)
    print(f"patched {p.name} (Figure A1 panel 4)")

for path, reps in edits.items():
    t = path.read_text()
    for a, b in reps:
        if b in t:
            continue
        assert t.count(a) == 1, f"{path.name}: {a[:50]!r} found {t.count(a)} times"
        t = t.replace(a, b)
    path.write_text(t)
    print(f"patched {path.name}")
