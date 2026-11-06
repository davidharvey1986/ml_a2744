# Figure 5 (combined BAHAMAS + DARKSKIES response and sigma/m posterior) with the Harvey and Jauzac A2744 mass priors
# overlaid; same functions as plots_for_paper.figure5_summaries, copied so nothing that rewrites the catalogue is imported.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from mass_weighting import a2744_mass_gaussian, gaussian_mass_weights

SBI = os.path.join(HERE, "..", "sbi")
def err_mean(a, rho=0.283):
    return np.std(a, 0) * np.sqrt((1 - rho) / a.shape[0] + rho)
def summary_curve(S, dom, mu, sig, min_ncomp=2):
    P = np.load(f"{S}/params_{dom}.npz"); soft = np.load(f"{S}/softmax_{dom}.npy")
    cs, mass, nc = P["cross_section"], P["log10_mass_750"], P["ncomponents"]
    sig_ = np.unique(cs); out = np.zeros((soft.shape[0], len(sig_)))
    for j, s in enumerate(sig_):
        rows = (cs == s) & (nc >= min_ncomp) & np.isfinite(mass)
        w = gaussian_mass_weights(mass[rows], mu, sig)
        out[:, j] = (soft[:, :, rows] * w).sum(-1).mean(1) / w.sum()
    return sig_, out
def combined_curve(S, mu, sig):
    c = {d: summary_curve(S, d, mu, sig) for d in ("bahamas", "darkskies")}
    sg = np.unique(np.concatenate([v[0] for v in c.values()])); out = np.zeros((next(iter(c.values()))[1].shape[0], len(sg)))
    for j, s in enumerate(sg):
        out[:, j] = np.mean([v[1][:, np.isclose(v[0], s)][:, 0] for v in c.values() if np.isclose(v[0], s).any()], 0)
    return sg, out
def hpd(x, post, mass):
    """Highest-posterior-density interval holding `mass`: the region above the density level that encloses it."""
    order = np.argsort(post)[::-1]; dx = np.gradient(x)
    keep = order[:np.searchsorted(np.cumsum(post[order] * dx[order]), mass) + 1]
    return x[keep].min(), x[keep].max()
def posterior(sg, curve, p_obs, cdm=1e-3):
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.3, length_scale_bounds=(1e-2, 10), nu=1.5) + \
        WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    gp = GaussianProcessRegressor(kernel=k, alpha=1e-4).fit(np.log10(np.where(sg == 0, cdm, sg))[:, None], curve.mean(0))
    x = np.linspace(-3, 1, 4001); m, sd = gp.predict(x[:, None], return_std=True)
    post = norm.pdf(np.mean(p_obs), m, np.sqrt(sd ** 2 + float(err_mean(p_obs)) ** 2)); post /= np.trapz(post, x)
    cdf = cumulative_trapezoid(post, x, initial=0); cdf /= cdf[-1]
    return gp, x, post, cdf, tuple(np.interp([0.5, 0.16, 0.84], cdf, x))

for tag in ("final", "best"):
    S = f"{SBI}/summaries_wb1_{tag}_n5"
    p_obs = np.load(f"{S}/obs.npz")["softmax"][:, 0]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    pm, pe = p_obs.mean(), err_mean(p_obs[:, None])[0]
    ax.axhspan(pm - pe, pm + pe, color="b", alpha=0.2, lw=0); ax.axhline(pm, color="b", lw=1.5, label=f"A2744 {pm:.3f}")
    print(f"== ba_{tag}: A2744 {pm:.3f} ± {pe:.3f}")
    for study, col, off in (("harvey", "0.35", -0.04), ("jauzac", "#D95F02", 0.04)):
        mu, sig = a2744_mass_gaussian(study)
        sg, curve = combined_curve(S, mu, sig)
        gp, x, post, cdf, (med, lo, hi) = posterior(sg, curve, p_obs)
        xs = np.log10(np.where(sg == 0, 1e-3, sg))
        ax.errorbar(xs + off, curve.mean(0), err_mean(curve), fmt="o", color=col, capsize=2)
        xp = np.linspace(-3.2, 0.5, 200); g, s = gp.predict(xp[:, None], return_std=True)
        ax.plot(xp, g, "--", color=col, label=f"{study.capitalize()} mass {mu:.2f} ± {sig:.2f}")
        ax.fill_between(xp, g - s, g + s, color=col, alpha=0.15, lw=0)
        best = 10 ** med
        p_cdm = np.interp(-2.0, x, cdf)
        ml = x[np.argmax(post)]; h68, h95 = hpd(x, post, 0.68), hpd(x, post, 0.95)
        for (a, b), alpha in ((h95, 0.15), (h68, 0.3)):
            m = (x >= a) & (x <= b); bx.fill_between(x[m], post[m], color=col, alpha=alpha, lw=0)
        bx.plot(x, post, color=col, label=rf"{study.capitalize()}: ML $\sigma/m={10**ml:.2f}^{{+{10**h68[1]-10**ml:.2f}}}_{{-{10**ml-10**h68[0]:.2f}}}$")
        bx.axvline(ml, color=col, lw=1.2)
        print(f"  {study:7s} ML log sigma {ml:.2f} (sigma/m {10**ml:.2f}); 68% HPD [{h68[0]:.2f}, {h68[1]:.2f}] -> [{10**h68[0]:.2f}, {10**h68[1]:.2f}]; "
              f"95% HPD [{h95[0]:.2f}, {h95[1]:.2f}] -> [{10**h95[0]:.3f}, {10**h95[1]:.2f}]")
        print(f"  {study:7s} mass {mu:.3f}±{sig:.3f}  curve " + ", ".join(f"{a:g}:{b:.3f}" for a, b in zip(sg, curve.mean(0)))
              + f"  | log sigma {med:.2f} [{lo:.2f}, {hi:.2f}], sigma/m {best:.2f}, P(log sigma < -2) {p_cdm:.3f}")
    ax.set_xticks([-3, -2, -1, 0], ["CDM", "-2", "-1", "0"])
    for a in (ax, bx): a.set_xlabel(r"$\log_{10}[(\sigma_{\rm DM}/m)/({\rm cm^2\,g^{-1}})]$")
    ax.set_ylabel("Mean model output (BAHAMAS + DARKSKIES)"); ax.legend(fontsize=8, loc=2)
    bx.set_ylabel("Posterior"); bx.set_title("vertical line: ML; shaded: 68, 95% HPD", fontsize=9); bx.set_xlim(-3, 1); bx.legend(fontsize=8, loc=2)
    fig.suptitle(f"wb1 ba_{tag}: response with the Harvey vs Jauzac A2744 mass prior", fontsize=11)
    fig.savefig(f"{HERE}/plots/response_harvey_vs_jauzac_ba_{tag}.png", dpi=170)
